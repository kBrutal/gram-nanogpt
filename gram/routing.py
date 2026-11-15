"""Gradient routing (plan §4): which auxiliaries run and which parameter groups update, per batch.

| batch  | active aux                                   | optimizers stepped           |
|--------|----------------------------------------------|------------------------------|
| core   | none, or one uniformly w.p. p_cr             | core (+ the selected aux)    |
| aux k  | k (+ each other aux w.p. p_xa, forward only) | k always; core w.p. p_as     |

A group that is not selected still gets its gradients *computed* (the core is never
detached, so the active aux learns through it); its optimizer is simply not stepped,
so its weights and AdamW state (incl. weight decay) do not change.

p_xa (variant, default 0 = plan §4): co-activating the other auxiliaries on an aux batch
teaches the core and aux k to tolerate them, so the full-access profile is in-distribution.
The co-active auxiliaries are never stepped, so they still never learn from domain k.
"""
import torch
from torch import nn


def make_optimizer(params, lr: float, weight_decay: float):
    """AdamW with decay on matrices/embeddings only (not biases or LayerNorm), as in nanoGPT."""
    params = [p for p in params if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


class Router:
    """Samples routing decisions from its own generator so they are reproducible and resumable."""

    def __init__(self, aux, p_as: float, p_cr: float, seed: int = 0, p_xa: float = 0.0):
        assert all(0.0 <= p <= 1.0 for p in (p_as, p_cr, p_xa))
        self.aux, self.p_as, self.p_cr, self.p_xa = tuple(aux), p_as, p_cr, p_xa
        self.gen = torch.Generator().manual_seed(seed)

    def _uniform(self) -> float:
        return torch.rand((), generator=self.gen).item()

    def route(self, domain: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """-> (active auxiliaries, parameter groups whose optimizer steps)."""
        if not self.aux:  # standard / data-filtered baseline: one group, always updated
            return (), ("core",)
        if domain == "core":
            if self._uniform() < self.p_cr:  # each aux w.p. p_cr / K, none w.p. 1 - p_cr
                k = self.aux[int(torch.randint(len(self.aux), (), generator=self.gen))]
                return (k,), ("core", k)
            return (), ("core",)
        if domain not in self.aux:
            raise ValueError(f"unknown domain {domain!r}; expected 'core' or one of {self.aux}")
        stepped = ("core", domain) if self._uniform() < self.p_as else (domain,)
        # No draws when p_xa == 0, so plan-§4 runs keep exactly the same random stream.
        co_active = {k for k in self.aux if k != domain and self.p_xa > 0 and self._uniform() < self.p_xa}
        return tuple(k for k in self.aux if k == domain or k in co_active), stepped

    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self.gen.set_state(state["gen"])


class OptimizerManager:
    """One AdamW per parameter group; only selected groups are stepped."""

    def __init__(self, groups: dict[str, list[nn.Parameter]], lr: float, weight_decay: float):
        self.groups = groups
        self.opts = {k: make_optimizer(ps, lr, weight_decay) for k, ps in groups.items()}
        self.updates = dict.fromkeys(groups, 0)

    def zero_grad(self) -> None:
        for o in self.opts.values():
            o.zero_grad(set_to_none=True)

    def set_lr(self, lr: float) -> None:
        # ponytail: one global-step schedule for every group (documented choice; per-group schedules if needed)
        for o in self.opts.values():
            for g in o.param_groups:
                g["lr"] = lr

    def grad_norms(self) -> dict[str, float]:
        """L2 norm of the *computed* gradient per group (0.0 if none), before clipping."""
        out = {}
        for k, ps in self.groups.items():
            grads = [p.grad.norm() for p in ps if p.grad is not None]
            out[k] = torch.linalg.vector_norm(torch.stack(grads)).item() if grads else 0.0
        return out

    def step(self, names, grad_clip: float | None = 1.0, log: bool = False) -> dict:
        """Clip and step only `names`. With log=True, also return grad norms and parameter-change norms
        for every group (stepped or not), which is how routing correctness is audited."""
        rec, before = {}, {}
        if log:
            rec["grad_norm"] = self.grad_norms()
            before = {k: [p.detach().clone() for p in ps] for k, ps in self.groups.items()}
        for k in names:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.groups[k], grad_clip)  # per group: core grads never scale aux
            self.opts[k].step()
            self.updates[k] += 1
        if log:
            rec["delta"] = {k: torch.linalg.vector_norm(torch.stack(
                [(p.detach() - b).norm() for p, b in zip(ps, before[k])])).item()
                for k, ps in self.groups.items()}
        return rec

    def state_dict(self) -> dict:
        return {"opts": {k: o.state_dict() for k, o in self.opts.items()}, "updates": dict(self.updates)}

    def load_state_dict(self, state: dict) -> None:
        for k, o in self.opts.items():
            o.load_state_dict(state["opts"][k])
        self.updates = dict(state["updates"])


def train_step(model, router: Router, optim: OptimizerManager, x, y, domain: str,
               grad_clip: float | None = 1.0, log: bool = False) -> dict:
    """One routed update. Leaves the model in the routed profile; set the profile explicitly before eval."""
    # ponytail: no gradient accumulation; if added, route once per optimizer step and keep the window single-domain
    active, stepped = router.route(domain)
    optim.zero_grad()
    model.set_profile(active)
    _, loss = model(x, targets=y)
    loss.backward()
    rec = optim.step(stepped, grad_clip, log)
    return {"domain": domain, "active": list(active), "stepped": list(stepped), "loss": loss.detach(), **rec}
