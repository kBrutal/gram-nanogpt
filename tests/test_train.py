import math

from gram.train import lr_at, main


def test_lr_schedule():
    assert lr_at(0, 1.0, 10, 100) == 0.1
    assert lr_at(9, 1.0, 10, 100) == 1.0
    assert math.isclose(lr_at(100, 1.0, 10, 100), 0.1)


def test_train_checkpoint_reload_generate(tmp_path):
    text = tmp_path / "t.txt"
    text.write_text("the quick brown fox jumps over the lazy dog. " * 200)
    res = main(["--data", str(text), "--out", str(tmp_path / "run"), "--device", "cpu",
                "--steps", "80", "--eval-every", "40", "--eval-iters", "4", "--batch-size", "8",
                "--sample-tokens", "20", "--n-layer", "2", "--n-head", "2", "--n-embd", "32",
                "--d-ff", "64", "--block-size", "32", "--warmup", "10", "--lr", "3e-3"])
    h = res["history"]
    assert h[-1]["val_loss"] < h[0]["val_loss"] - 1.0  # it learns
    assert math.isclose(res["reloaded_val"], h[-1]["val_loss"], rel_tol=1e-5)  # checkpoint round-trips
    assert len(res["sample"].encode("utf-8", "replace")) >= 21
    for f in ("ckpt.pt", "config.json", "metrics.jsonl", "sample.txt"):
        assert (tmp_path / "run" / f).exists()
