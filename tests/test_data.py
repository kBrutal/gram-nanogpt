import gzip
import json
import random
from collections import Counter

import numpy as np
from tokenizers import Tokenizer

from gram.data import (SQL_IN_CODE, assign_split, build, norm_question, norm_sql, parses_as_py3, reindent,
                       schema_key, sql_doc)
from gram.train import main as train_main


def test_split_is_deterministic_and_group_level():
    assert assign_split("python", "org/repo") == assign_split("python", "org/repo")
    c = Counter(assign_split("core", str(i)) for i in range(50_000))
    assert abs(c["val"] / 50_000 - 0.02) < 0.004 and abs(c["test"] / 50_000 - 0.02) < 0.004


def test_python_cleaning():
    method = 'def f(self, x):\n        """Doc."""\n        return x\n'
    assert reindent(method) == 'def f(self, x):\n    """Doc."""\n    return x\n'
    assert parses_as_py3(reindent(method))
    assert not parses_as_py3('def f():\n    print "py2"')
    assert SQL_IN_CODE.search('cur.execute("SELECT id FROM users")')
    assert SQL_IN_CODE.search("q = 'select * from t'")
    assert not SQL_IN_CODE.search('"""Select items from the list."""')


def test_sql_normalisation():
    assert norm_sql('SELECT count(*)  FROM "singer" ;') == norm_sql("select count( * ) from 'singer'")
    assert norm_question("How many singers do we have ?") == norm_question("how many singers do we have")
    assert schema_key("CREATE TABLE a (x INT); INSERT INTO a VALUES (1)") == schema_key("create table a (x int)")


def fake_records():
    rng = random.Random(0)
    words = "the cat sat on a mat and the dog ran far away".split()
    for i in range(400):
        yield {"domain": "core", "source": "fake", "revision": "r", "orig_id": f"c{i}", "group": f"c{i}",
               "text": " ".join(rng.choice(words) for _ in range(30)) + "."}
    for i in range(400):
        yield {"domain": "python", "source": "fake", "revision": "r", "orig_id": f"p{i}", "group": f"repo{i % 200}",
               "text": f"def f{i}(x):\n    if x > {i}:\n        return x * {i % 7}\n    return None"}
    for i in range(400):
        q, s = f"How many rows have id {i}?", f"SELECT COUNT(*) FROM t{i % 30} WHERE id = {i}"
        yield {"domain": "sql", "source": "fake", "revision": "r", "orig_id": f"s{i}", "group": f"schema{i % 200}",
               "question": q, "query": s, "text": sql_doc(f"CREATE TABLE t{i % 30} (id INT)", q, s)}
    yield {"domain": "core", "source": "fake", "revision": "r", "orig_id": "dup", "group": "dup",
           "text": "the cat  sat on a mat and the dog ran far away."}
    yield {"domain": "core", "source": "fake", "revision": "r", "orig_id": "dup2", "group": "dup2",
           "text": "the cat sat on a mat and the dog ran far away."}  # whitespace-only duplicate
    yield {"domain": "sql", "source": "fake", "revision": "r", "orig_id": "leak", "group": "concert",
           "question": "How many singers do we have?", "query": "SELECT count(*) FROM singer",
           "text": sql_doc("CREATE TABLE singer (id INT)", "How many singers do we have?", "SELECT count(*) FROM singer")}


def test_build_end_to_end(tmp_path):
    spider = [{"id": 0, "db_id": "concert_singer", "question": "How many singers do we have?",
               "query": "SELECT count(*) FROM singer"}]
    drops = Counter()
    meta = build(fake_records(), spider, tmp_path, vocab_size=400, tok_sample_chars=10_000, drops=drops)
    assert drops == {"exact_duplicate": 1, "sql_overlaps_spider_dev": 1}

    rows = [json.loads(l) for l in gzip.open(tmp_path / "metadata.jsonl.gz", "rt")]
    assert len(rows) == 1201 and all(r["orig_id"] != "leak" for r in rows)
    splits_per_group = Counter()
    for g in {(r["domain"], r["group"]) for r in rows}:
        splits_per_group[len({r["split"] for r in rows if (r["domain"], r["group"]) == g})] += 1
    assert set(splits_per_group) == {1}  # no group straddles two splits

    tok = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    code = "def f(x):\n    if x > 3:\n        return x  # ok\n\tpass"
    assert tok.decode(tok.encode(code).ids) == code  # byte-level BPE round-trips whitespace exactly
    for d, by_split in meta["counts"].items():
        for split, c in by_split.items():
            ids = np.fromfile(tmp_path / f"{d}_{split}.bin", dtype=np.uint16)
            assert len(ids) == c["tokens"] and (ids == meta["eot_id"]).sum() == c["docs"]
            assert sum(r["n_tokens"] for r in rows if (r["domain"], r["split"]) == (d, split)) == c["tokens"]
    first_py = np.fromfile(tmp_path / "python_train.bin", dtype=np.uint16)
    first_doc = tok.decode(first_py[: list(first_py).index(meta["eot_id"])].tolist())
    assert first_doc.startswith("def f") and "return" in first_doc

    # The trainer consumes the build directly (val only: tiny val splits need a short context).
    res = train_main(["--data-dir", str(tmp_path), "--out", str(tmp_path / "run"), "--device", "cpu",
                      "--steps", "2", "--eval-every", "2", "--eval-iters", "1", "--batch-size", "2",
                      "--block-size", "16", "--n-layer", "1", "--n-head", "1", "--n-embd", "16",
                      "--d-ff", "32", "--d-aux", "8"])
    assert res["updates"]["core"] >= 1
