import json
import math
import random

import numpy as np
import pytest
import torch

from gram.train import lr_at, main

TINY = ["--device", "cpu", "--batch-size", "8", "--block-size", "32", "--n-layer", "2", "--n-head", "2",
        "--n-embd", "32", "--d-ff", "64", "--d-aux", "16", "--eval-iters", "3", "--warmup", "5", "--lr", "3e-3"]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Three byte-level toy domains with distinct surface forms."""
    root, rng = tmp_path_factory.mktemp("data"), random.Random(0)
    words = "the cat sat on a mat and the dog ran far away".split()
    gens = {
        "core": lambda: " ".join(rng.choice(words) for _ in range(12)) + ".\n",
        "python": lambda: f"def f{rng.randint(0, 9)}(x):\n    return x + {rng.randint(0, 9)}\n",
        "sql": lambda: f"SELECT c{rng.randint(0, 9)} FROM t{rng.randint(0, 9)} WHERE id = {rng.randint(0, 99)};\n",
    }
    for d, gen in gens.items():
        for split, n in (("train", 400), ("val", 60)):
            raw = "".join(gen() for _ in range(n)).encode()
            np.frombuffer(raw, dtype=np.uint8).astype(np.uint16).tofile(root / f"{d}_{split}.bin")
    (root / "meta.json").write_text(json.dumps({"vocab_size": 256, "domains": list(gens), "tokenizer": "bytes"}))
    return root


def test_lr_schedule():
    assert lr_at(0, 1.0, 10, 100) == 0.1
    assert lr_at(9, 1.0, 10, 100) == 1.0
    assert math.isclose(lr_at(100, 1.0, 10, 100), 0.1)


def test_gram_training_end_to_end(data_dir, tmp_path):
    out = tmp_path / "run"
    res = main(["--data-dir", str(data_dir), "--out", str(out), "--steps", "60", "--eval-every", "30",
                "--log-every", "1", *TINY])
    first, last = res["eval"][0]["val"], res["eval"][-1]["val"]
    assert set(last) == {"full", "core", "python", "sql"}
    assert all(last["full"][d] < first["full"][d] - 0.5 for d in ("core", "python", "sql"))  # learns every domain
    assert sum(res["stats"]["tokens"].values()) == 60 * 8 * 32

    recs = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert len(recs) == 60
    for r in recs:  # every logged step obeys the update matrix (plan §4.5)
        if r["domain"] == "core":
            assert len(r["active"]) <= 1 and r["stepped"] == ["core", *r["active"]]
        else:
            other = {"python": "sql", "sql": "python"}[r["domain"]]
            assert r["active"] == [r["domain"]] and other not in r["stepped"]
            assert r["delta"][other] == 0.0 and r["grad_norm"][other] == 0.0
    assert recs[-1]["updates"] == res["updates"]

    ckpt = torch.load(out / "ckpt.pt")
    assert set(ckpt["optim"]["opts"]) == {"core", "python", "sql"} and ckpt["step"] == 60
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["params"].keys() == {"core", "python", "sql"} and cfg["git"]


def test_resume_matches_uninterrupted(data_dir, tmp_path):
    common = ["--data-dir", str(data_dir), "--steps", "30", "--eval-every", "10", *TINY]
    full = main([*common, "--out", str(tmp_path / "a")])
    main([*common, "--out", str(tmp_path / "b"), "--stop-after", "13"])  # odd step: not an eval boundary
    resumed = main(["--resume", str(tmp_path / "b" / "ckpt.pt"), "--device", "cpu"])
    assert resumed["updates"] == full["updates"]
    assert resumed["stats"] == full["stats"]
    assert resumed["eval"][-1]["val"] == full["eval"][-1]["val"]
    a, b = full["model"].state_dict(), resumed["model"].state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    steps = [json.loads(l)["step"] for l in (tmp_path / "b" / "eval.jsonl").read_text().splitlines()]
    assert steps == [0, 10, 20, 30]


def test_data_filtered_standard_baseline(data_dir, tmp_path):
    res = main(["--data-dir", str(data_dir), "--out", str(tmp_path / "b"), "--steps", "10", "--eval-every", "10",
                "--arch", "standard", "--domains", "core,sql", *TINY])
    assert res["updates"] == {"core": 10}
    assert "python" not in res["stats"]["tokens"]
    assert list(res["eval"][-1]["val"]) == ["full"]
    assert set(res["eval"][-1]["val"]["full"]) == {"core", "python", "sql"}  # held-out domain still evaluated
