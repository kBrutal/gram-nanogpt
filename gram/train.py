"""Domain-routed training (plan §6).

Data layout (produced by the step-7 pipeline, or by hand for toy runs):
    <data-dir>/meta.json             {"vocab_size": int, "domains": [...], + tokenizer/dataset identifiers}
    <data-dir>/<domain>_<split>.bin  uint16 token ids; split in {train, val}

    python -m gram.train --data-dir data/processed --out runs/gram_s0                        # GRAM
    python -m gram.train --data-dir data/processed --arch standard --out runs/std_s0         # Baseline A
    python -m gram.train --data-dir data/processed --arch standard --domains core,sql ...    # Baseline B (no Python)
    python -m gram.train --resume runs/gram_s0/ckpt.pt                                       # continue a run
"""
import argparse
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from gram.model import GPT, GPTConfig, profiles
from gram.routing import OptimizerManager, Router, train_step

MODEL_FLAGS = ("block_size", "n_layer", "n_head", "n_embd", "d_ff", "d_aux")


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to min_ratio * max_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def load_split(data_dir: Path, domains, split: str, block_size: int) -> dict[str, np.memmap]:
    out = {}
    for d in domains:
        arr = np.memmap(data_dir / f"{d}_{split}.bin", dtype=np.uint16, mode="r")
        if len(arr) <= block_size + 1:
            raise ValueError(f"{d}_{split}.bin has {len(arr)} tokens; need more than block_size+1={block_size + 1}")
        out[d] = arr
    return out


def get_batch(arr, batch_size: int, block_size: int, device: str, gen: torch.Generator):
    ix = torch.randint(len(arr) - block_size - 1, (batch_size,), generator=gen).tolist()
    chunk = torch.from_numpy(np.stack([arr[i : i + block_size + 1] for i in ix]).astype(np.int64))
    return chunk[:, :-1].to(device), chunk[:, 1:].to(device)


@torch.no_grad()
def eval_matrix(model, val: dict, batch_size: int, block_size: int, iters: int, device: str, seed: int) -> dict:
    """Validation loss for every profile x domain. Every cell sees the same batches for its domain."""
    model.eval()
    out = {}
    for name, active in profiles(model.cfg.aux).items():
        model.set_profile(active)
        out[name] = {}
        for d, arr in val.items():
            gen = torch.Generator().manual_seed(seed)
            out[name][d] = sum(model(*get_batch(arr, batch_size, block_size, device, gen))[1].item()
                               for _ in range(iters)) / iters
    model.train()
    return out


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_checkpoint(path, device: str = "cpu"):
    """-> (model in eval mode, full checkpoint dict). For evaluation scripts."""
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", help="directory with meta.json and <domain>_<split>.bin")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--resume", help="checkpoint to continue from (its saved settings are reused)")
    p.add_argument("--arch", choices=("gram", "standard"), default="gram")
    p.add_argument("--domains", default="core,python,sql", help="training domains; 'core' required")
    p.add_argument("--mix", default="", help="domain sampling weights, e.g. core=0.6,python=0.2,sql=0.2 "
                                             "(default: equal); this is where p_af is applied")
    p.add_argument("--p-as", type=float, default=0.3, help="P(core group updates) on an auxiliary batch")
    p.add_argument("--p-cr", type=float, default=0.5, help="P(one auxiliary active) on a core batch")
    p.add_argument("--p-xa", type=float, default=0.0, help="variant: P(each other auxiliary co-active, "
                                                           "forward only, never stepped) on an auxiliary batch")
    p.add_argument("--xa-freeze-core", action="store_true", help="ablation: never step the core on a batch "
                                                                 "with a co-active auxiliary (needs --p-xa)")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--stop-after", type=int, default=0, help="save and exit after N steps this session (0 = off)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    defaults = GPTConfig()
    for f in MODEL_FLAGS:
        p.add_argument("--" + f.replace("_", "-"), type=int, default=getattr(defaults, f))
    args = p.parse_args(argv)
    if not args.resume and not args.data_dir:
        p.error("--data-dir is required unless --resume is given")
    return args


def main(argv=None) -> dict:
    args = parse_args(argv)
    resume = torch.load(args.resume, map_location="cpu") if args.resume else None
    if resume:  # reuse the run's settings; only session-level flags may change
        session = {k: getattr(args, k) for k in ("resume", "device", "stop_after")}
        defaults = vars(parse_args(["--resume", args.resume]))  # flags added after the run was started
        args = argparse.Namespace(**{**defaults, **resume["args"], **session})
    device = pick_device(args.device)
    if device == "cuda":  # TF32 matmuls on Ampere+ (A100/L4): near-fp32 accuracy, several times faster
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((data_dir / "meta.json").read_text())
    domains = args.domains.split(",")
    assert "core" in domains, "GRAM and the baselines always train the core domain"
    aux = tuple(d for d in domains if d != "core") if args.arch == "gram" else ()
    cfg = GPTConfig(vocab_size=meta["vocab_size"], aux=aux, **{f: getattr(args, f) for f in MODEL_FLAGS})
    train = load_split(data_dir, domains, "train", cfg.block_size)
    val = load_split(data_dir, meta["domains"], "val", cfg.block_size)  # evaluate held-out domains too

    mix = dict.fromkeys(domains, 1.0)
    for part in filter(None, args.mix.split(",")):
        k, v = part.split("=")
        assert k in mix, f"--mix domain {k!r} not in --domains"
        mix[k] = float(v)
    weights = torch.tensor([mix[d] for d in domains])

    torch.manual_seed(args.seed)
    model = GPT(cfg).to(device)
    # Distinct seeds: same-seeded generators would produce identical streams and correlate
    # domain sampling with routing decisions.
    router = Router(aux, args.p_as, args.p_cr, seed=args.seed + 1, p_xa=args.p_xa,
                    xa_freeze_core=args.xa_freeze_core)
    data_gen = torch.Generator().manual_seed(args.seed + 2)
    eval_seed = args.seed + 3
    optim = OptimizerManager(model.param_groups(), args.lr, args.weight_decay)
    stats = {"tokens": Counter(), "batches": Counter(), "routes": Counter()}
    start = 0
    if resume:
        model.load_state_dict(resume["model"])
        optim.load_state_dict(resume["optim"])
        router.load_state_dict(resume["router"])
        data_gen.set_state(resume["rng"]["data"])
        torch.set_rng_state(resume["rng"]["torch"])
        stats = {k: Counter(v) for k, v in resume["stats"].items()}
        start = resume["step"]
    else:
        (out / "config.json").write_text(json.dumps({
            "args": vars(args), "model": asdict(cfg), "meta": meta, "mix": mix, "device": device,
            "params": {k: sum(p.numel() for p in ps) for k, ps in model.param_groups().items()},
            "params_total": model.num_params(), "git": git_commit(), "torch": torch.__version__,
            "python": platform.python_version(), "platform": platform.platform()}, indent=2))

    def save(step, last_eval):
        ckpt = {"model": model.state_dict(), "optim": optim.state_dict(), "router": router.state_dict(),
                "rng": {"data": data_gen.get_state(), "torch": torch.get_rng_state()},
                "step": step, "stats": {k: dict(v) for k, v in stats.items()}, "args": vars(args),
                "config": asdict(cfg), "meta": meta, "eval": last_eval}
        torch.save(ckpt, out / "ckpt.pt.tmp")
        os.replace(out / "ckpt.pt.tmp", out / "ckpt.pt")  # atomic: a killed session never leaves a torn file

    mode = "a" if resume else "w"
    train_log, eval_log = open(out / "train.jsonl", mode), open(out / "eval.jsonl", mode)
    print(f"device={device} arch={args.arch} aux={list(aux)} params={model.num_params():,} start={start}")
    history, last_eval, t0 = [], None, time.time()
    step = start
    while True:
        if (step % args.eval_every == 0 or step == args.steps) and not (resume and step == start):
            em = eval_matrix(model, val, args.batch_size, cfg.block_size, args.eval_iters, device, eval_seed)
            last_eval = {"step": step, "tokens": dict(stats["tokens"]), "updates": dict(optim.updates),
                         "val": em, "elapsed_s": round(time.time() - t0, 1)}
            history.append(last_eval)
            eval_log.write(json.dumps(last_eval) + "\n"); eval_log.flush()
            cells = "  ".join(f"{p}[" + " ".join(f"{d}={v:.3f}" for d, v in row.items()) + "]" for p, row in em.items())
            print(f"step {step:6d} | {cells} | {last_eval['elapsed_s']}s", flush=True)
            save(step, last_eval)
        if step == args.steps:
            break
        if args.stop_after and step - start == args.stop_after:
            save(step, last_eval)
            print(f"stopped at step {step}; resume with --resume {out / 'ckpt.pt'}")
            break

        domain = domains[int(torch.multinomial(weights, 1, generator=data_gen))]
        x, y = get_batch(train[domain], args.batch_size, cfg.block_size, device, data_gen)
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        optim.set_lr(lr)
        log = step % args.log_every == 0
        rec = train_step(model, router, optim, x, y, domain, args.grad_clip, log)
        stats["tokens"][domain] += x.numel()
        stats["batches"][domain] += 1
        stats["routes"][f"{domain}|active={'+'.join(rec['active']) or '-'}|step={'+'.join(rec['stepped'])}"] += 1
        if log:
            rec.update(step=step, lr=lr, loss=rec["loss"].item(), updates=dict(optim.updates))
            train_log.write(json.dumps(rec) + "\n")
        step += 1
    train_log.close(); eval_log.close()
    return {"eval": history, "stats": stats, "updates": dict(optim.updates), "model": model}


if __name__ == "__main__":
    main()
