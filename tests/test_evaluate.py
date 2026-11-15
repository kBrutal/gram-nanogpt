import csv
import json
import math
import shutil

import pytest
import torch

from gram.evaluate import evaluate_checkpoint, main, window_losses
from gram.train import load_checkpoint, main as train_main
from tests.conftest import TINY


@pytest.fixture(scope="module")
def ckpt(data_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    train_main(["--data-dir", str(data_dir), "--out", str(out), "--steps", "40", "--eval-every", "40", *TINY])
    return out / "ckpt.pt"


def test_window_losses_match_model_loss(ckpt, data_dir):
    import numpy as np
    model, _ = load_checkpoint(ckpt)
    arr = np.memmap(data_dir / "python_test.bin", dtype=np.uint16, mode="r")
    T = model.cfg.block_size
    w = window_losses(model, arr, T, batch_size=4, device="cpu")
    assert len(w) == (len(arr) - 1) // T  # whole split, non-overlapping windows
    x = torch.from_numpy(arr[2 * T : 3 * T + 1].astype(np.int64))[None]
    assert math.isclose(w[2].item(), model(x[:, :-1], targets=x[:, 1:])[1].item(), rel_tol=1e-5)
    assert len(window_losses(model, arr, T, 4, "cpu", max_tokens=5 * T + 1)) == 5


def test_matrix_and_derived_metrics(ckpt, data_dir, tmp_path):
    (r,) = main([str(ckpt), "--data-dir", str(data_dir), "--device", "cpu", "--csv", str(tmp_path / "m.csv")])
    m, dv = r["matrix"], r["derived"]
    assert set(m) == {"full", "core", "python", "sql"} and all(set(row) == {"core", "python", "sql"} for row in m.values())
    L = lambda p, d: m[p][d]["loss"]
    close = lambda a, b: math.isclose(a, b, abs_tol=1e-5)
    assert close(dv["python"]["dL_access"]["value"], L("sql", "python") - L("python", "python"))
    assert close(dv["sql"]["dL_vs_full"]["value"], L("python", "sql") - L("full", "sql"))
    assert close(dv["sql"]["dL_core_only"]["value"], L("core", "sql") - L("sql", "sql"))
    assert close(dv["python"]["full_access_cost"]["value"], L("full", "python") - L("python", "python"))
    leak = (L("core", "python") - L("sql", "python")) / (L("core", "python") - L("python", "python"))
    assert math.isclose(dv["python"]["cross_module_leakage"]["value"], leak, rel_tol=1e-3)  # ratio of small gaps
    assert close(dv["core"]["core_only_vs_full"]["value"], L("core", "core") - L("full", "core"))
    assert json.loads((ckpt.parent / "eval_test.json").read_text())["step"] == 40

    rows = list(csv.DictReader(open(tmp_path / "m.csv")))
    assert sum(r["metric"] == "loss" for r in rows) == 12
    assert {r["metric"] for r in rows} >= {"loss", "dL_access", "dL_vs_full", "cross_module_leakage"}


def test_evaluation_does_not_change_weights(ckpt, data_dir):
    before = load_checkpoint(ckpt)[0].state_dict()
    evaluate_checkpoint(ckpt, data_dir, max_tokens=200)
    after = load_checkpoint(ckpt)[0].state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_tokenizer_mismatch_is_refused(ckpt, data_dir, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(data_dir, other)
    meta = json.loads((other / "meta.json").read_text())
    (other / "meta.json").write_text(json.dumps({**meta, "tokenizer_sha256": "different"}))
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        evaluate_checkpoint(ckpt, other)
