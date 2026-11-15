# GRAM: Gradient-Routed Auxiliary Modules

Selective capability control in a small language model. GRAM is a ~5.8M-parameter GPT-style decoder whose feed-forward
layers hold a **shared core** plus **one auxiliary MLP per specialised domain** (here: Python and SQL). During
training, gradients are *routed* by the domain of each batch, so each auxiliary learns its own domain. At
inference the auxiliaries can be switched on or off, and the question is whether turning a module off removes that
capability while leaving everything else intact.

An independent, small-scale implementation inspired by *Modular Pretraining Enables Access Control*
(arXiv:2607.08077). It does not reproduce the paper's large-scale experiments.

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data](#data)
- [Training](#training)
- [Running on Colab](#running-on-colab)
- [Evaluation](#evaluation)
- [Results so far](#results-so-far)
- [Tests](#tests)

## How it works

### Model

A pre-norm GPT-2-style decoder written from scratch (random init, no pretrained weights):

| | |
|---|---|
| Layers / heads / width | 4 / 4 / 256 |
| Context | 256 tokens |
| Vocabulary | 8,000 (byte-level BPE shared by all domains) |
| Core MLP width | 1,024 |
| Auxiliary MLP width | 128 each (Python, SQL) |
| Parameters | 5,273,088 core + 263,680 per auxiliary = 5,800,448 |

Attention uses `F.scaled_dot_product_attention` (causal), the LM head is tied to the token embedding, and
initialisation follows GPT-2 (residual projections scaled by `1/sqrt(2·n_layer)`). There is no dropout.

Each block's MLP is the core plus whichever auxiliaries are active:

```python
out = core(x) + sum(aux[k](x) for k in active)
```

An inactive auxiliary is skipped entirely, so it contributes nothing and receives no gradient. With every module on,
the MLP is mathematically a single MLP of width 1,024 + 128 + 128 = 1,280, which is what the parameter-matched
baseline uses.

### Inference profiles

| Profile | Active auxiliaries | Intended user |
|---|---|---|
| `full` | Python, SQL | everything enabled |
| `core` | none | general text only |
| `python` | Python | general text + Python |
| `sql` | SQL | general text + SQL |

Switch profiles with `model.set_profile("python")` (or any iterable of module names). Switching never changes
weights.

### Gradient routing

Every batch comes from one domain. The router decides which auxiliaries run in the forward pass and which parameter
groups get an optimizer step:

| Batch domain | Active auxiliaries | Groups stepped |
|---|---|---|
| core text | none, or one auxiliary chosen uniformly with probability `p_cr` | core (+ the chosen auxiliary) |
| auxiliary `k` | `k` (+ each other auxiliary with probability `p_xa`, forward only) | `k` always; core with probability `p_as` |

- The other auxiliary is **never stepped** on a `k` batch, so it never learns from domain `k`.
- A group that isn't stepped still has gradients computed (the core is never detached, so the active auxiliary
  learns through it), but its optimizer doesn't run, so its weights and AdamW state, including weight decay, stay
  untouched.
- Each group (`core`, `python`, `sql`) has its own AdamW and its own gradient clipping. One learning-rate schedule
  (linear warmup, then cosine to 10%) drives all three.
- Router, data sampling and evaluation use separate seeds (`seed+1`, `seed+2`, `seed+3`). Checkpoints store every
  RNG state, so a resumed run is bit-identical to an uninterrupted one.

Defaults: `p_as = 0.3`, `p_cr = 0.5`, `p_xa = 0` (the plain method). `p_xa > 0` is a variant that co-activates the
other auxiliary during training so the `full` profile is in-distribution; `--xa-freeze-core` is an ablation of that
variant (see [Results](#results-so-far)).

## Repository layout

```
gram/
  model.py      GPT, GRAM MLP, inference profiles, parameter groups
  routing.py    Router, per-group optimizers, train_step
  train.py      training loop, checkpointing/resume, validation loss matrix
  data.py       download, filtering, dedup, group-level splits, tokenizer, packed .bin files
  evaluate.py   held-out loss matrix and suppression/leakage metrics
tests/          unit and end-to-end tests (pytest)
colab.ipynb     notebook for training and evaluating on a Colab GPU
```

Generated outputs (`data/`, `runs/`, `results/`, `dist/`, `*.pt`) are gitignored.

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

On macOS with torch 2.2 (built against NumPy 1.x), keep `numpy<2` and `pyarrow<20`; newer pyarrow requires NumPy 2.
Training picks CUDA, then Apple MPS, then CPU automatically (`--device` overrides). TF32 is enabled on CUDA.

## Data

```bash
python -m gram.data --out data/processed                # full build: ~830 MB download (cached in data/raw)
python -m gram.data --out data/smoke --max-docs 3000    # small build for quick local runs
```

Every source is downloaded as parquet straight from the Hugging Face Hub at a **pinned revision**, over verified TLS.

| Domain | Source | Use | License |
|---|---|---|---|
| core | TinyStories (one train shard) | train/val/test | CDLA-Sharing-1.0 |
| python | CodeSearchNet, Python subset | train/val/test | see dataset card |
| sql | `sql-create-context` + Gretel `synthetic_text_to_sql` | train/val/test | CC-BY-4.0 / Apache-2.0 |
| — | Spider dev | evaluation only, never trained on | CC-BY-SA-4.0 |

Processing:

- **SQL** documents are formatted as `-- schema: …` / `-- question: …` / query. Exact duplicates are removed, and any
  example overlapping Spider dev is dropped (570).
- **Python** functions are re-indented. Anything that fails `ast.parse` under Python 3 (4,473) or embeds SQL
  (1,955) is dropped.
- **Splits** are assigned per *group*, not per document, using a hash of `domain|group`: test 2%, val 2%. Groups
  are a story, a repository, or a `CREATE TABLE` schema, so near-duplicates can't straddle train and test.
- A single 8k byte-level BPE is trained on an equal-character sample of each domain's training split.
- Output is `uint16` token ids with `<|endoftext|>` between documents, plus `meta.json` (counts, sources, revisions,
  licenses, drops, tokenizer hash) and per-document metadata.

Full build size (tokens):

| | train | val | test |
|---|---:|---:|---:|
| core | 114.6M | 2.38M | 2.37M |
| python | 128.1M | 3.01M | 3.11M |
| sql | 20.7M | 0.43M | 0.43M |

## Training

```bash
# GRAM (plain method)
python -m gram.train --data-dir data/processed --out runs/gram_s0

# full-access variant and its ablation
python -m gram.train --data-dir data/processed --out runs/gram_xa50_s0 --p-xa 0.5
python -m gram.train --data-dir data/processed --out runs/gram_xa50fc_s0 --p-xa 0.5 --xa-freeze-core

# baselines: standard model (no auxiliaries), with and without Python in the data
python -m gram.train --data-dir data/processed --arch standard --d-ff 1280 --out runs/std_s0
python -m gram.train --data-dir data/processed --arch standard --domains core,sql --out runs/nopy_s0

# continue an interrupted run (all saved settings are reused)
python -m gram.train --resume runs/gram_s0/ckpt.pt
```

Main flags:

| Flag | Default | Meaning |
|---|---|---|
| `--arch` | `gram` | `gram` or `standard` (no auxiliaries) |
| `--domains` | `core,python,sql` | training domains; `core` is required |
| `--mix` | equal | domain sampling weights, e.g. `core=0.6,python=0.2,sql=0.2` |
| `--p-as` / `--p-cr` / `--p-xa` | 0.3 / 0.5 / 0 | routing probabilities (see above) |
| `--xa-freeze-core` | off | never step the core on a batch with a co-active auxiliary |
| `--steps` | 10,000 | total steps (batch 64 × 256 tokens ≈ 164M tokens) |
| `--stop-after` | 0 | save and exit after N steps this session |
| `--lr` / `--weight-decay` / `--warmup` | 1e-3 / 0.1 / 200 | AdamW (β = 0.9, 0.95); decay on matrices only |
| `--eval-every` / `--eval-iters` | 500 / 20 | validation loss matrix cadence and size |
| `--seed` | 0 | model init; router, data and eval seeds derive from it |
| `--n-layer --n-head --n-embd --d-ff --d-aux --block-size` | 4 / 4 / 256 / 1024 / 128 / 256 | model size |

Each run directory contains:

- `ckpt.pt`: model, optimizers, router, RNG states, stats. Written atomically every eval.
- `config.json`: all arguments, parameter counts per group, git commit and environment.
- `train.jsonl`: every `--log-every` steps, with the route, the groups stepped, per-group gradient norms and update
  sizes.
- `eval.jsonl`: the validation loss for every profile × domain.

A 10,000-step run takes about 40 minutes on a Colab GPU.

## Running on Colab

`colab.ipynb` keeps everything persistent in `MyDrive/gram/`:

1. Build the code archive locally: `git archive -o dist/gram_code.zip HEAD`. Upload it to `MyDrive/gram/`.
2. Open the notebook in Colab with a GPU runtime and run the cells in order:
   1. Mount Drive.
   2. Unzip the code and run the tests.
   3. Build the data once and cache it as `data_processed.tar` on Drive. Later sessions restore that cached copy,
      so every run trains on byte-identical data.
   4. Train the run named by `RUN` / `EXTRA`.
3. If the session dies, re-run the cells: training resumes from the last checkpoint on Drive.
4. The later cells show:
   - a go/no-go readout for one run
   - a comparison of all runs
   - held-out test evaluation, written to `MyDrive/gram/results/loss_test.csv`

## Evaluation

```bash
python -m gram.evaluate runs/*/ckpt.pt --data-dir data/processed               # whole test split
python -m gram.evaluate runs/gram_s0/ckpt.pt --data-dir data/processed --max-tokens 500000   # quick check
```

Every checkpoint is scored on every domain under every profile, over the whole split in non-overlapping 256-token
windows, so the result is deterministic. It refuses data whose tokenizer hash differs from the one used in
training.

Outputs:

- `<run>/eval_<split>.json`: the loss matrix plus derived metrics.
- One long-format CSV (`--csv`, default `results/metrics/loss.csv`) with the columns run, step, split, profile,
  domain, metric, value, stderr and tokens.

Derived metrics for each auxiliary domain `d` (losses in nats, positive means worse):

| Metric | Definition | Meaning |
|---|---|---|
| `dL_access` | L(other module on) − L(own module on) | what a user without `d`'s module loses; main suppression measure |
| `dL_vs_full` | L(other module on) − L(full) | same, against the `full` profile |
| `dL_core_only` | L(core) − L(own module on) | value of `d`'s module on its own |
| `full_access_cost` | L(full) − L(own module on) | interference from enabling every module |
| `cross_module_leakage` | (L(core) − L(other on)) / (L(core) − L(own on)) | share of `d`'s gain the other module recovers anyway |

For the core domain the evaluation also reports `core_only_vs_full` and the largest shift across profiles.
Standard errors are over windows, and the differences are paired, so each window is compared with itself under the
other profile. They understate the run-to-run uncertainty; repeat seeds for that.

Generation-based metrics (SQL execution accuracy, Python syntax validity) are planned.

## Results so far

> **Preliminary: one seed per setting, 10,000 steps.** The test split agrees with validation to within about 0.06
> nats, and the per-cell standard errors are at most 0.011. Seed-to-seed variation has not been measured yet.

**Test loss** (nats; lower is better; each domain's own profile is in bold):

| Profile | `gram_s0` core | `gram_s0` Python | `gram_s0` SQL | `gram_xa50_s0` core | `gram_xa50_s0` Python | `gram_xa50_s0` SQL |
|---|---:|---:|---:|---:|---:|---:|
| full | 2.169 | 4.193 | 2.288 | 2.362 | 3.914 | 1.897 |
| core | **2.090** | 6.674 | 5.492 | **2.306** | 7.119 | 5.469 |
| python | 2.102 | **3.227** | 3.949 | 2.325 | **3.923** | 4.788 |
| sql | 2.101 | 5.441 | **1.448** | 2.322 | 6.046 | **1.890** |

| Metric | `gram_s0` Python | `gram_s0` SQL | `gram_xa50_s0` Python | `gram_xa50_s0` SQL |
|---|---:|---:|---:|---:|
| `dL_access` | +2.21 | +2.50 | +2.12 | +2.90 |
| `full_access_cost` | +0.97 | +0.84 | −0.01 | +0.01 |
| `cross_module_leakage` | 0.36 | 0.38 | 0.34 | 0.19 |

Findings:

1. **Routing localises the capabilities.** Without a domain's module, its loss rises by more than 2 nats, while
   core-text loss barely changes across profiles (≤ 0.08).
2. **Full access interferes in the plain method.** The two auxiliaries are never active together in training, and
   each learns a partly generic "code" correction. With both on, those corrections add up and overshoot, so `full`
   is about 0.9 nats worse than each domain's own profile.
3. **The `p_xa = 0.5` variant removes the interference, at a cost.** `full_access_cost` drops to about 0, and SQL
   suppression and leakage improve. But every domain's best loss gets worse: core +0.22, Python +0.69, SQL +0.44.
   - The Python and SQL cost fits each module spending capacity on cancelling the other.
   - The core gap appears within the first ~1,500 steps and then stays flat, and is not yet explained.
4. **Two runs are queued to explain this:**
   - `gram_xa50fc_s0` (`--xa-freeze-core`) tests whether updating the core on co-active batches causes the core
     regression.
   - `gram_s1` (`--seed 1`) measures seed noise.

## Tests

```bash
pytest -q
```

The 44 tests check these properties:

- **Model:** the core profile equals a model with the auxiliaries removed; full access equals a width-1,280 MLP;
  switching profiles never changes weights; inactive modules get no gradients; the parameter groups partition the
  model.
- **Routing:**
  - The update matrix holds under a huge learning rate and weight decay, so any stray step would show.
  - Routing frequencies match `p_as` and `p_cr`.
  - Gradients are cleared between batches, and update counts match the routing log.
  - Resume is bit-identical.
  - The `p_xa` variant is forward-only, and `--xa-freeze-core` changes only core steps.
- **Training:** an end-to-end routing audit on every logged step; resuming from an interrupted run matches an
  uninterrupted one; the data-filtered baseline works.
- **Data:** splits are deterministic; Python cleaning and SQL normalisation work; a full toy build runs through the
  trainer.
- **Evaluation:** window losses equal the model's loss; the derived-metric arithmetic is right; evaluation never
  changes weights; a tokenizer mismatch is refused.
