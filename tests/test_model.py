import math

import torch

from gram.model import GPT, GPTConfig

TINY = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32, d_ff=64)


def make(cfg=TINY, seed=0):
    torch.manual_seed(seed)
    return GPT(cfg).eval()


def rand_idx(cfg=TINY, B=2, T=16, seed=1):
    return torch.randint(0, cfg.vocab_size, (B, T), generator=torch.Generator().manual_seed(seed))


def test_shapes_and_initial_loss():
    m, idx = make(), rand_idx()
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (2, 16, TINY.vocab_size)
    # Near-uniform predictions at init -> loss ~ ln(vocab).
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.5


def test_causal_no_future_leak():
    m, idx = make(), rand_idx()
    t = 8
    idx2 = idx.clone()
    idx2[:, t:] = (idx2[:, t:] + 1) % TINY.vocab_size
    a, _ = m(idx)
    b, _ = m(idx2)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)
    assert not torch.allclose(a[:, t:], b[:, t:])


def test_weight_tying():
    m = make()
    assert m.lm_head.weight is m.wte.weight


def test_param_count_default_config():
    # Recorded so architecture drift is noticed (plan §3.5 target: 5-10M).
    assert GPT(GPTConfig()).num_params() == 5_273_088


def test_save_load_roundtrip(tmp_path):
    m, idx = make(seed=0), rand_idx()
    torch.save(m.state_dict(), tmp_path / "ckpt.pt")
    m2 = make(seed=123)
    m2.load_state_dict(torch.load(tmp_path / "ckpt.pt"))
    assert m2.lm_head.weight is m2.wte.weight  # tying survives reload
    assert torch.equal(m(idx)[0], m2(idx)[0])


def test_generate_length_and_determinism():
    m, prompt = make(), rand_idx(B=1, T=4)
    out = m.generate(prompt, max_new_tokens=40, top_k=1)  # greedy; crosses block_size
    assert out.shape == (1, 44)
    assert torch.equal(out[:, :4], prompt)
    assert torch.equal(out, m.generate(prompt, max_new_tokens=40, top_k=1))


def test_can_overfit_one_batch():
    m, idx = make().train(), rand_idx(B=4, T=32)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = m(idx[:, :-1], targets=idx[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


# ---------------- GRAM (step 4) ----------------
from dataclasses import replace

from gram.model import GRAM_AUX, PROFILES

GRAM_TINY = replace(TINY, aux=GRAM_AUX, d_aux=16)


def test_every_layer_has_core_and_aux():
    m = make(GRAM_TINY)
    for b in m.blocks:
        assert set(b.mlp.aux) == {"python", "sql"}
        assert b.mlp.core.fc.out_features == 64 and b.mlp.aux["python"].fc.out_features == 16


def test_default_profile_is_full_and_unknown_rejected():
    m = make(GRAM_TINY)
    assert m.active == frozenset(GRAM_AUX)
    try:
        m.set_profile(["rust"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_profiles_activate_intended_branches():
    m, idx = make(GRAM_TINY), rand_idx()
    calls = []
    for i, b in enumerate(m.blocks):
        for k, branch in b.mlp.aux.items():
            branch.register_forward_hook(lambda mod, inp, out, k=k, i=i: calls.append((i, k)))
    for name, active in PROFILES.items():
        calls.clear()
        m.set_profile(name)
        logits, _ = m(idx)
        assert logits.shape == (2, 16, TINY.vocab_size)
        assert sorted(calls) == sorted((i, k) for i in range(TINY.n_layer) for k in active), name


def test_core_profile_equals_model_without_aux():
    gram, idx = make(GRAM_TINY), rand_idx()
    base = GPT(TINY).eval()
    missing, unexpected = base.load_state_dict(gram.state_dict(), strict=False)
    assert not missing and all(".mlp.aux." in k for k in unexpected)
    gram.set_profile("core")
    assert torch.equal(gram(idx)[0], base(idx)[0])


def test_disabled_aux_changes_nothing_and_full_differs():
    m, idx = make(GRAM_TINY), rand_idx()
    m.set_profile("python")
    ref = m(idx)[0]
    with torch.no_grad():  # scramble the disabled SQL branch: output must not move
        for b in m.blocks:
            for p in b.mlp.aux["sql"].parameters():
                p.add_(torch.randn_like(p) * 10)
    assert torch.equal(m(idx)[0], ref)
    m.set_profile("full")
    assert not torch.allclose(m(idx)[0], ref)


def test_switching_profiles_mutates_nothing():
    m, idx = make(GRAM_TINY), rand_idx()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    full = m(idx)[0]
    for name in ("core", "python", "sql", "full"):
        m.set_profile(name)
        m(idx)
    assert torch.equal(m(idx)[0], full)  # full output unchanged after visiting other profiles
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_inactive_aux_gets_no_gradient():
    m, idx = make(GRAM_TINY).train(), rand_idx()
    m.set_profile("python")
    m(idx, targets=idx)[1].backward()
    g = m.param_groups()
    assert all(p.grad is not None for p in g["python"])
    assert all(p.grad is None for p in g["sql"])
    assert all(p.grad is not None for p in g["core"])


def test_param_groups_partition():
    m = make(GRAM_TINY)
    g = m.param_groups()
    assert set(g) == {"core", "python", "sql"}
    ids = [id(p) for ps in g.values() for p in ps]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    core = {id(p) for p in g["core"]}
    assert id(m.wte.weight) in core and id(m.lm_head.weight) in core and id(m.wpe.weight) in core
    for b in m.blocks:
        for mod in (b.attn, b.ln1, b.ln2, b.mlp.core):
            assert all(id(p) in core for p in mod.parameters())
        for k in GRAM_AUX:
            assert {id(p) for p in b.mlp.aux[k].parameters()} <= {id(p) for p in g[k]}
    assert len(GPT(TINY).param_groups()) == 1  # baseline: core only


def test_full_profile_equals_wide_baseline():
    """GRAM full access == one standard MLP with d_ff = core + sum(aux) (plan's parameter-matched Baseline A)."""
    gram, idx = make(GRAM_TINY), rand_idx()
    wide = GPT(replace(TINY, d_ff=64 + 2 * 16)).eval()
    sd = {k: v for k, v in gram.state_dict().items() if ".mlp." not in k}
    for i, b in enumerate(gram.blocks):
        parts = [b.mlp.core, *b.mlp.aux.values()]
        p = f"blocks.{i}.mlp.core."
        sd[p + "fc.weight"] = torch.cat([m.fc.weight for m in parts], 0)
        sd[p + "fc.bias"] = torch.cat([m.fc.bias for m in parts], 0)
        sd[p + "proj.weight"] = torch.cat([m.proj.weight for m in parts], 1)
        sd[p + "proj.bias"] = sum(m.proj.bias for m in parts)
    wide.load_state_dict(sd)
    assert torch.allclose(gram(idx)[0], wide(idx)[0], atol=1e-5)


def test_gram_param_count_default_config():
    m = GPT(GPTConfig(aux=GRAM_AUX))
    sizes = {k: sum(p.numel() for p in ps) for k, ps in m.param_groups().items()}
    assert sizes == {"core": 5_273_088, "python": 263_680, "sql": 263_680}
    # Parameter-matched baseline (d_ff=1280) differs only by 2 extra proj biases per layer.
    assert m.num_params() - GPT(GPTConfig(d_ff=1280)).num_params() == 2 * 256 * 4
