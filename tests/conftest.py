import json
import random

import numpy as np
import pytest

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
        for split, n in (("train", 400), ("val", 60), ("test", 60)):
            raw = "".join(gen() for _ in range(n)).encode()
            np.frombuffer(raw, dtype=np.uint8).astype(np.uint16).tofile(root / f"{d}_{split}.bin")
    (root / "meta.json").write_text(json.dumps({"vocab_size": 256, "domains": list(gens), "tokenizer": "bytes"}))
    return root
