from collections import Counter
from dataclasses import replace

import pytest
import torch

from gram.model import GPT, GPTConfig, GRAM_AUX
from gram.routing import OptimizerManager, Router, train_step

CFG = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32, d_ff=64, aux=GRAM_AUX, d_aux=16)


def setup(p_as=0.3, p_cr=0.5, seed=0, lr=1e-2, wd=0.1, cfg=CFG):
    torch.manual_seed(seed)
    model = GPT(cfg).train()
    return model, Router(cfg.aux, p_as, p_cr, seed=seed), OptimizerManager(model.param_groups(), lr, wd)


def batch(seed=0, B=2, T=16):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, CFG.vocab_size, (B, T + 1), generator=g)
    return idx[:, :-1], idx[:, 1:]


def snapshot(model):
    return {k: [p.detach().clone() for p in ps] for k, ps in model.param_groups().items()}


def changed(before, model):
    return {k: any(not torch.equal(b, p) for b, p in zip(before[k], ps)) for k, ps in model.param_groups().items()}


@pytest.mark.parametrize("domain,other", [("python", "sql"), ("sql", "python")])
@pytest.mark.parametrize("p_as", [0.0, 1.0])
def test_aux_batch_routing(domain, other, p_as):
    model, router, optim = setup(p_as=p_as, lr=1.0, wd=1.0)  # huge lr/decay: any stray step would show
    before = snapshot(model)
    for s in range(10):
        rec = train_step(model, router, optim, *batch(s), domain, log=True)
        assert rec["active"] == [domain]
        assert rec["grad_norm"][domain] > 0 and rec["grad_norm"][other] == 0.0  # other aux: no grads at all
        assert rec["grad_norm"]["core"] > 0  # core grads are computed even when core is not stepped
        assert rec["delta"][other] == 0.0
        assert (rec["delta"]["core"] > 0) == (p_as == 1.0)
    assert changed(before, model) == {"core": p_as == 1.0, domain: True, other: False}
    assert optim.opts[other].state == {}  # inactive optimizer state never advances
    assert (optim.opts["core"].state == {}) == (p_as == 0.0)
    assert optim.updates == {"core": 10 * int(p_as), domain: 10, other: 0}


def test_core_batch_without_aux():
    model, router, optim = setup(p_cr=0.0)
    before = snapshot(model)
    rec = train_step(model, router, optim, *batch(), "core", log=True)
    assert rec["active"] == [] and rec["stepped"] == ["core"]
    assert rec["grad_norm"]["python"] == rec["grad_norm"]["sql"] == 0.0
    assert changed(before, model) == {"core": True, "python": False, "sql": False}


def test_core_batch_with_one_aux():
    model, router, optim = setup(p_cr=1.0)
    before = snapshot(model)
    rec = train_step(model, router, optim, *batch(), "core", log=True)
    (k,) = rec["active"]
    assert rec["stepped"] == ["core", k]
    assert changed(before, model) == {"core": True, "python": k == "python", "sql": k == "sql"}


def test_router_frequencies():
    n, p_as, p_cr = 20_000, 0.3, 0.5
    r = Router(GRAM_AUX, p_as, p_cr, seed=0)
    core = Counter(r.route("core")[0] for _ in range(n))
    assert set(core) == {(), ("python",), ("sql",)}  # at most one aux per core batch
    assert abs(core[()] / n - (1 - p_cr)) < 0.02
    assert abs(core[("python",)] / n - p_cr / 2) < 0.02 and abs(core[("sql",)] / n - p_cr / 2) < 0.02
    for d in GRAM_AUX:
        decisions = [r.route(d) for _ in range(n)]
        assert all(a == (d,) and d in s for a, s in decisions)  # own aux always active and stepped
        assert abs(sum("core" in s for _, s in decisions) / n - p_as) < 0.02


def test_router_edge_cases():
    assert Router((), 0.3, 0.5).route("python") == ((), ("core",))  # baseline: single group
    with pytest.raises(ValueError):
        Router(GRAM_AUX, 0.3, 0.5).route("rust")


def test_gradients_cleared_between_batches():
    model, router, optim = setup(p_as=1.0)
    train_step(model, router, optim, *batch(0), "python")
    train_step(model, router, optim, *batch(1), "sql")
    groups = model.param_groups()
    assert all(p.grad is None for p in groups["python"])  # stale python grads gone after an sql step
    assert all(p.grad is not None for p in groups["sql"])


def test_update_counts_match_routing_log():
    model, router, optim = setup()
    tally = Counter()
    for s in range(60):
        rec = train_step(model, router, optim, *batch(s), ("core", "python", "sql")[s % 3])
        tally.update(rec["stepped"])
    assert optim.updates == {k: tally[k] for k in ("core", "python", "sql")}


def test_resume_is_bit_identical(tmp_path):
    domains = ["core", "python", "sql"] * 7
    model, router, optim = setup(seed=0)
    for s, d in enumerate(domains[:10]):
        train_step(model, router, optim, *batch(s), d)
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "router": router.state_dict()},
               tmp_path / "ckpt.pt")
    for s, d in enumerate(domains[10:], start=10):
        train_step(model, router, optim, *batch(s), d)

    ck = torch.load(tmp_path / "ckpt.pt")
    model2, router2, _ = setup(seed=999)  # different init and routing seed: load must override both
    model2.load_state_dict(ck["model"])
    optim2 = OptimizerManager(model2.param_groups(), 1e-2, 0.1)
    optim2.load_state_dict(ck["optim"])
    router2.load_state_dict(ck["router"])
    for s, d in enumerate(domains[10:], start=10):
        train_step(model2, router2, optim2, *batch(s), d)

    assert optim2.updates == optim.updates
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), model2.state_dict().values()))


def test_baseline_single_optimizer():
    model, router, optim = setup(cfg=replace(CFG, aux=()))
    rec = train_step(model, router, optim, *batch(), "python")
    assert rec["stepped"] == ["core"] and set(optim.opts) == {"core"}
