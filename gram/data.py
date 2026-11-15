"""Data pipeline (plan §5): download -> parse/filter -> dedup -> group-level split -> shared BPE -> packed .bin.

    python -m gram.data --out data/processed                  # full build (~830 MB download, cached in data/raw)
    python -m gram.data --out data/smoke --max-docs 3000      # quick build for local runs

Outputs (read by gram.train):
    meta.json            vocab, domains, token/doc counts per split, sources + pinned revisions + licenses, drops
    tokenizer.json       one byte-level BPE shared by every domain
    <domain>_<split>.bin uint16 token ids, documents separated by <|endoftext|>
    metadata.jsonl.gz    one row per kept document (domain, source, revision, orig_id, group, split, n_tokens)
    spider_dev.jsonl     held-out Spider dev questions/queries for later evaluation (never trained on)
"""
import argparse
import ast
import gzip
import hashlib
import itertools
import json
import random
import re
import shutil
import ssl
import textwrap
import urllib.request
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PREPROCESS_VERSION = 1
EOT = "<|endoftext|>"
DOMAINS = ("core", "python", "sql")
SPLIT_FRACTIONS = {"test": 0.02, "val": 0.02}  # by hashed group; the rest is train
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}"
SOURCES = {
    "tinystories": dict(
        domain="core", repo="roneneldan/TinyStories", rev="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        files=["data/train-00000-of-00004-2d5a1467fff1081b.parquet"], license="CDLA-Sharing-1.0"),
    "codesearchnet_python": dict(
        domain="python", repo="code-search-net/code_search_net", rev="bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
        files=["python/train-00000-of-00001.parquet", "python/validation-00000-of-00001.parquet",
               "python/test-00000-of-00001.parquet"],
        license="other: code from repositories under open-source licenses (see dataset card)"),
    "sql_create_context": dict(
        domain="sql", repo="b-mc2/sql-create-context", rev="9d80a6a118b838d9defc3798d659a54a2ac2ff37",
        files=["sql_create_context_v4.json"], license="CC-BY-4.0"),
    "gretel_text_to_sql": dict(
        domain="sql", repo="gretelai/synthetic_text_to_sql", rev="740ab236e64503fba51be1101df7a1be83bf455d",
        files=["synthetic_text_to_sql_train.snappy.parquet", "synthetic_text_to_sql_test.snappy.parquet"],
        license="Apache-2.0"),
    "spider_dev": dict(  # evaluation only
        domain=None, repo="xlangai/spider", rev="0c350918f3f29ec754f1181c65cdce76cd6c133c",
        files=["spider/validation-00000-of-00001.parquet"], license="CC-BY-SA-4.0"),
}

# ponytail: keyword heuristic, not a SQL parser; catches embedded queries, misses lowercase prose-like SQL
SQL_IN_CODE = re.compile(r"\bSELECT\b[\s\S]{1,200}?\bFROM\b|\bINSERT\s+INTO\b|\bCREATE\s+TABLE\b|\bDELETE\s+FROM\b"
                         r"|(?i:\bselect\s+(?:\*|count\s*\())")


# ---------------- normalisation helpers ----------------

def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_question(q: str) -> str:
    return re.sub(r"\s*([?.,!])\s*", r"\1", collapse_ws(q.lower())).rstrip("?.! ")


def norm_sql(q: str) -> str:
    q = collapse_ws(q.lower().replace('"', "'"))
    return re.sub(r"\s*([(),=<>;*])\s*", r"\1", q).rstrip(";")


def schema_key(context: str) -> str:
    """Split group for SQL: the CREATE TABLE statements (ignores INSERTed rows)."""
    tables = re.findall(r"create\s+table[^;]*", context, flags=re.I)
    return collapse_ws(" ; ".join(tables) if tables else context).lower()


def assign_split(domain: str, group: str) -> str:
    h = int.from_bytes(hashlib.blake2b(f"{domain}|{group}".encode(), digest_size=8).digest(), "big") / 2**64
    if h < SPLIT_FRACTIONS["test"]:
        return "test"
    return "val" if h < SPLIT_FRACTIONS["test"] + SPLIT_FRACTIONS["val"] else "train"


def reindent(code: str) -> str:
    """CodeSearchNet strips a method's `def` line but keeps the body's class-level indent; restore 4 spaces."""
    head, sep, body = code.partition("\n")
    return head + sep + textwrap.indent(textwrap.dedent(body), "    ") if body else head


def parses_as_py3(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def sql_doc(schema: str, question: str, query: str) -> str:
    return f"-- schema: {schema.strip()}\n-- question: {question.strip()}\n{query.strip()}"


# ---------------- download + parse ----------------

def _ssl_context():
    """Verified TLS; python.org macOS builds lack system CAs, so prefer certifi's bundle when present."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(name: str, raw_dir: Path) -> list[Path]:
    src, paths = SOURCES[name], []
    for f in src["files"]:
        path = raw_dir / name / src["rev"][:12] / Path(f).name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {src['repo']}/{f}", flush=True)
            url = HF_URL.format(repo=src["repo"], rev=src["rev"], file=f)
            with urllib.request.urlopen(url, context=_ssl_context()) as r, open(path.with_suffix(".part"), "wb") as out:
                shutil.copyfileobj(r, out, 1 << 20)
            path.with_suffix(".part").rename(path)
        paths.append(path)
    return paths


def read_parquet(path: Path, columns) -> list[dict]:
    return pq.read_table(path, columns=columns).to_pylist()


def parse_source(name: str, paths: list[Path], drops: Counter):
    """Yield {orig_id, group, text} for one source."""
    if name == "tinystories":
        for p in paths:
            for i, r in enumerate(read_parquet(p, ["text"])):
                if text := (r["text"] or "").strip():
                    yield {"orig_id": f"{p.name}:{i}", "group": f"{p.name}:{i}", "text": text}
    elif name == "codesearchnet_python":
        for p in paths:
            for r in read_parquet(p, ["repository_name", "whole_func_string", "func_code_url"]):
                code = reindent(r["whole_func_string"]).strip()
                if SQL_IN_CODE.search(code):
                    drops["python_contains_sql"] += 1
                elif not parses_as_py3(code):
                    drops["python_not_py3"] += 1
                else:
                    yield {"orig_id": r["func_code_url"], "group": r["repository_name"], "text": code}
    elif name == "sql_create_context":
        for i, r in enumerate(json.loads(paths[0].read_text())):
            yield {"orig_id": str(i), "group": schema_key(r["context"]), "question": r["question"],
                   "query": r["answer"], "text": sql_doc(r["context"], r["question"], r["answer"])}
    elif name == "gretel_text_to_sql":
        for p in paths:
            for r in read_parquet(p, ["id", "sql_prompt", "sql_context", "sql"]):
                yield {"orig_id": f"{p.name}:{r['id']}", "group": schema_key(r["sql_context"]),
                       "question": r["sql_prompt"], "query": r["sql"],
                       "text": sql_doc(r["sql_context"], r["sql_prompt"], r["sql"])}
    else:
        raise ValueError(name)


def load_records(raw_dir: Path, max_docs: int | None, drops: Counter):
    for name, src in SOURCES.items():
        if src["domain"] is None:
            continue
        for rec in itertools.islice(parse_source(name, download(name, raw_dir), drops), max_docs):
            yield {"domain": src["domain"], "source": name, "revision": src["rev"], **rec}


def load_spider_dev(raw_dir: Path) -> list[dict]:
    rows = read_parquet(download("spider_dev", raw_dir)[0], ["db_id", "question", "query"])
    return [{"id": i, **r} for i, r in enumerate(rows)]


# ---------------- build ----------------

def train_tokenizer(texts, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    return tok


def build(records, spider_dev: list[dict], out: Path, vocab_size: int = 8000,
          tok_sample_chars: int = 20_000_000, seed: int = 0, drops: Counter | None = None,
          sources: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    drops = drops if drops is not None else Counter()
    spider_q = {norm_question(r["question"]) for r in spider_dev}
    spider_sql = {norm_sql(r["query"]) for r in spider_dev}

    seen, docs = set(), defaultdict(list)  # (domain, split) -> records
    for r in records:
        key = hashlib.blake2b(collapse_ws(r["text"]).encode(), digest_size=16).digest()
        if key in seen:
            drops["exact_duplicate"] += 1
            continue
        seen.add(key)
        question, query = r.pop("question", None), r.pop("query", None)
        if r["domain"] == "sql" and (norm_question(question) in spider_q or norm_sql(query) in spider_sql):
            drops["sql_overlaps_spider_dev"] += 1
            continue
        r["split"] = assign_split(r["domain"], r["group"])
        docs[(r["domain"], r["split"])].append(r)

    # Shared tokenizer, trained on an equal-character sample of each domain's *train* split.
    rng, sample = random.Random(seed), []
    for d in DOMAINS:
        pool, n = docs[(d, "train")][:], 0
        rng.shuffle(pool)
        for r in pool:
            if n >= tok_sample_chars:
                break
            sample.append(r["text"]); n += len(r["text"])
    tok = train_tokenizer(sample, vocab_size)
    tok.save(str(out / "tokenizer.json"))
    eot = tok.token_to_id(EOT)

    counts = {d: {} for d in DOMAINS}
    with gzip.open(out / "metadata.jsonl.gz", "wt") as meta_rows:
        for d, split in itertools.product(DOMAINS, ("train", "val", "test")):  # every file, even if empty
            rs, n_tok = docs[(d, split)], 0
            with open(out / f"{d}_{split}.bin", "wb") as f:
                for i in range(0, len(rs), 4096):
                    chunk = rs[i : i + 4096]
                    encs = tok.encode_batch([r["text"] for r in chunk])
                    ids = [t for e in encs for t in (*e.ids, eot)]
                    np.asarray(ids, dtype=np.uint16).tofile(f)
                    n_tok += len(ids)
                    for r, e in zip(chunk, encs):
                        meta_rows.write(json.dumps({k: r[k] for k in ("domain", "source", "revision", "orig_id", "group",
                                                                       "split")} | {"n_tokens": len(e.ids) + 1,
                                                                                    "preprocess_version": PREPROCESS_VERSION}) + "\n")
            counts[d][split] = {"docs": len(rs), "tokens": n_tok}
            print(f"{d:6s} {split:5s} {len(rs):9,} docs {n_tok:12,} tokens", flush=True)

    with open(out / "spider_dev.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in spider_dev)
    meta = {
        "vocab_size": tok.get_vocab_size(), "domains": list(DOMAINS), "eot_id": eot, "tokenizer": "tokenizer.json",
        "tokenizer_sha256": hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest(),
        "tokenizer_sample_chars_per_domain": tok_sample_chars, "preprocess_version": PREPROCESS_VERSION,
        "seed": seed, "split_rule": f"blake2b(domain|group) fractions {SPLIT_FRACTIONS}, rest train",
        "groups": {"core": "story", "python": "repository", "sql": "CREATE TABLE schema"},
        "counts": counts, "drops": dict(drops), "sources": sources if sources is not None else SOURCES,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--max-docs", type=int, default=None, help="cap documents per source (smoke builds)")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--tok-sample-chars", type=int, default=20_000_000, help="tokenizer training chars per domain")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    drops = Counter()
    raw = Path(a.raw)
    build(load_records(raw, a.max_docs, drops), load_spider_dev(raw), Path(a.out), a.vocab_size,
          a.tok_sample_chars, a.seed, drops)


if __name__ == "__main__":
    main()
