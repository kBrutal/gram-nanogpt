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
