"""Training loop.

Step 3 scope: single-domain, byte-level sanity run (train -> checkpoint -> reload -> generate).
Domain-labelled batches, gradient routing and per-group optimizers arrive in steps 5-6.

    python -m gram.train --data data/raw/tinyshakespeare.txt --out runs/sanity
"""
import argparse
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch

from gram.model import GPT, GPTConfig
from gram.routing import make_optimizer

BYTE_VOCAB = 256  # ponytail: byte-level tokens until the shared BPE tokenizer (step 7)
NON_CLI = ("vocab_size", "bias", "aux")  # GPTConfig fields not exposed as int CLI flags


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


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str, gen: torch.Generator):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, args, device, gen) -> float:
    model.eval()
    losses = [model(*get_batch(data, args.batch_size, args.block_size, device, gen))[1].item()
              for _ in range(args.eval_iters)]
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: Path, model, optimizer, cfg: GPTConfig, step: int, metrics: dict):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(cfg), "step": step, "metrics": metrics}, path)


def load_checkpoint(path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="plain-text file")
    p.add_argument("--out", default="runs/sanity")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--sample-tokens", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    for f in fields(GPTConfig):  # model dims: --n-layer, --n-embd, ...
        if f.name not in NON_CLI:
            p.add_argument("--" + f.name.replace("_", "-"), type=int, default=f.default)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = torch.frombuffer(bytearray(Path(args.data).read_bytes()), dtype=torch.uint8).long()
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = GPTConfig(vocab_size=BYTE_VOCAB, **{f.name: getattr(args, f.name) for f in fields(GPTConfig)
                                              if f.name not in NON_CLI})
    model = GPT(cfg).to(device)
    opt = make_optimizer(model.parameters(), args.lr, args.weight_decay)
    (out / "config.json").write_text(json.dumps({"args": vars(args), "model": asdict(cfg), "device": device,
                                                 "torch": torch.__version__, "params": model.num_params()}, indent=2))
    print(f"device={device} params={model.num_params():,} train_tokens={len(train_data):,}")

    batch_gen = torch.Generator().manual_seed(args.seed)
    eval_gen = lambda: torch.Generator().manual_seed(args.seed + 1)  # same val batches every eval
    history, log = [], open(out / "metrics.jsonl", "w")
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            val = estimate_loss(model, val_data, args, device, eval_gen())
            rec = {"step": step, "val_loss": val, "elapsed_s": round(time.time() - t0, 1)}
            history.append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step:5d} | val {val:.4f} | {rec['elapsed_s']}s")
            save_checkpoint(out / "ckpt.pt", model, opt, cfg, step, rec)
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
        _, loss = model(*get_batch(train_data, args.batch_size, cfg.block_size, device, batch_gen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
    log.close()

    # Prove the checkpoint round-trips: fresh model from disk, same val loss, then generate.
    reloaded, _ = load_checkpoint(out / "ckpt.pt", device)
    reloaded_val = estimate_loss(reloaded, val_data, args, device, eval_gen())
    prompt = torch.tensor([[ord("\n")]], device=device)
    sample = bytes(reloaded.generate(prompt, args.sample_tokens, top_k=20)[0].tolist()).decode("utf-8", "replace")
    (out / "sample.txt").write_text(sample)
    print(f"reloaded val {reloaded_val:.4f}\n--- sample ---{sample}\n--------------")
    return {"history": history, "reloaded_val": reloaded_val, "sample": sample}


if __name__ == "__main__":
    main()
