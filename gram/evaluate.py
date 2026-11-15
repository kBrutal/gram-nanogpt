"""Evaluation harness (plan §8), part 1: held-out loss matrix + suppression / leakage metrics.

    python -m gram.evaluate runs/gram_s0/ckpt.pt --data-dir data/processed
    python -m gram.evaluate runs/*/ckpt.pt --data-dir data/processed --max-tokens 500000   # quicker

Every checkpoint is scored on every domain under every profile it supports, over the whole split in
non-overlapping block_size windows (deterministic, no sampling). Outputs:
    <ckpt dir>/eval_<split>.json   matrix {profile: {domain: loss, stderr, tokens}} + derived metrics
    --csv (results/metrics/loss.csv) long-format rows appended for tables and plots

Derived metrics for each auxiliary domain d (losses; positive = worse; stderr is paired over windows):
    dL_access         L(other aux on)[d] - L(d on)[d]   a user without d's module vs a user with it
    dL_vs_full        L(other aux on)[d] - L(full)[d]   plan §8.5 definition (full as reference)
    dL_core_only      L(core only)[d]    - L(d on)[d]   removing d's module from its own access profile
    full_access_cost  L(full)[d]         - L(d on)[d]   interference from enabling every module
    cross_module_leakage  share of d's module gain over core-only that the other modules recover anyway
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gram.model import profiles
from gram.train import load_checkpoint, pick_device


@torch.no_grad()
def window_losses(model, arr, block_size: int, batch_size: int, device: str, max_tokens: int | None = None):
    """Mean next-token loss of each non-overlapping block_size window in `arr` (1-D tensor, on CPU)."""
    n_tok = min(len(arr), max_tokens) if max_tokens else len(arr)
    n_win = (n_tok - 1) // block_size
    out = []
    for w in range(0, n_win, batch_size):
        starts = range(w * block_size, min(w + batch_size, n_win) * block_size, block_size)
        chunk = torch.from_numpy(np.stack([arr[s : s + block_size + 1] for s in starts]).astype(np.int64)).to(device)
        logits, _ = model(chunk[:, :-1])
        out.append(F.cross_entropy(logits.transpose(1, 2), chunk[:, 1:], reduction="none").mean(1).float().cpu())
    return torch.cat(out)


def _stderr(t: torch.Tensor) -> float:
    return (t.std() / len(t) ** 0.5).item() if len(t) > 1 else float("nan")


def derive(losses: dict, aux) -> dict:
    """Suppression / leakage metrics from per-window losses {profile: {domain: tensor}}."""
    by_set = {frozenset(a): name for name, a in profiles(aux).items()}

    def L(active, d):
        return losses[by_set[frozenset(active)]][d]

    def diff(a, b):
        t = a - b  # same windows in every profile -> paired difference
        return {"value": t.mean().item(), "stderr": _stderr(t)}

    out, full = {}, set(aux)
    for d in aux:
        own, others = {d}, full - {d}
        if not all(frozenset(s) in by_set for s in (own, others, full, set())) or d not in losses["full"]:
            continue
        gain = (L(set(), d) - L(own, d)).mean().item()
        recovered = (L(set(), d) - L(others, d)).mean().item()
        out[d] = {
            "dL_access": diff(L(others, d), L(own, d)),
            "dL_vs_full": diff(L(others, d), L(full, d)),
            "dL_core_only": diff(L(set(), d), L(own, d)),
            "full_access_cost": diff(L(full, d), L(own, d)),
            "cross_module_leakage": {"value": recovered / gain if abs(gain) > 1e-6 else float("nan"), "stderr": None},
        }
    if aux and "core" in losses["full"]:
        out["core"] = {"core_only_vs_full": diff(L(set(), "core"), L(full, "core")),
                       "max_shift_across_profiles": {"value": max(r["core"].mean().item() for r in losses.values())
                                                     - min(r["core"].mean().item() for r in losses.values()),
                                                     "stderr": None}}
    return out


def evaluate_checkpoint(ckpt_path, data_dir, split: str = "test", batch_size: int = 32,
                        max_tokens: int | None = None, device: str = "cpu") -> dict:
    data_dir = Path(data_dir)
    model, ckpt = load_checkpoint(ckpt_path, device)
    meta = json.loads((data_dir / "meta.json").read_text())
    trained_on, given = ckpt["meta"].get("tokenizer_sha256"), meta.get("tokenizer_sha256")
    if trained_on != given:  # a different tokenizer makes every token id mean something else
        raise ValueError(f"tokenizer mismatch: checkpoint was trained with {trained_on}, {data_dir} has {given}")
    block = model.cfg.block_size
    data = {d: np.memmap(data_dir / f"{d}_{split}.bin", dtype=np.uint16, mode="r") for d in meta["domains"]}
    losses = {}
    for name, active in profiles(model.cfg.aux).items():
        model.set_profile(active)
        losses[name] = {d: window_losses(model, arr, block, batch_size, device, max_tokens) for d, arr in data.items()}
    matrix = {p: {d: {"loss": t.mean().item(), "stderr": _stderr(t), "tokens": len(t) * block}
                  for d, t in row.items()} for p, row in losses.items()}
    return {"ckpt": str(ckpt_path), "run": Path(ckpt_path).parent.name, "step": ckpt["step"], "split": split,
            "max_tokens": max_tokens, "arch_aux": list(model.cfg.aux), "train_args": ckpt["args"],
            "matrix": matrix, "derived": derive(losses, model.cfg.aux)}


def write_csv(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    keys = ("run", "step", "split", "profile", "domain", "metric", "value", "stderr", "tokens")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new:
            w.writeheader()
        base = {k: result[k] for k in ("run", "step", "split")}
        for p, row in result["matrix"].items():
            for d, m in row.items():
                w.writerow({**base, "profile": p, "domain": d, "metric": "loss", "value": m["loss"],
                            "stderr": m["stderr"], "tokens": m["tokens"]})
        for d, ms in result["derived"].items():
            for name, m in ms.items():
                w.writerow({**base, "profile": "", "domain": d, "metric": name, "value": m["value"],
                            "stderr": m["stderr"], "tokens": ""})


def print_result(r: dict) -> None:
    domains = list(next(iter(r["matrix"].values())))
    print(f"\n== {r['run']}  step {r['step']}  split={r['split']}")
    print(f"   {'profile':8s}" + "".join(f"{d:>16s}" for d in domains))
    for p, row in r["matrix"].items():
        print(f"   {p:8s}" + "".join(f"{row[d]['loss']:9.3f} ±{row[d]['stderr']:.3f}" for d in domains))
    for d, ms in r["derived"].items():
        print(f"   {d:6s} " + "  ".join(f"{k} {m['value']:+.3f}" for k, m in ms.items()))


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=None, help="cap tokens per domain (quick checks)")
    p.add_argument("--device", default="auto")
    p.add_argument("--csv", default="results/metrics/loss.csv")
    a = p.parse_args(argv)
    device, results = pick_device(a.device), []
    for path in a.ckpts:
        r = evaluate_checkpoint(path, a.data_dir, a.split, a.batch_size, a.max_tokens, device)
        (Path(path).parent / f"eval_{a.split}.json").write_text(json.dumps(r, indent=2))
        write_csv(r, Path(a.csv))
        print_result(r)
        results.append(r)
    return results


if __name__ == "__main__":
    main()
