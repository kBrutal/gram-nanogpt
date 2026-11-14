# GRAM: Gradient-Routed Auxiliary Modules
## Research Implementation Plan

## 1. Project Overview

### Working title

**Modular Capability Access Control in Small Language Models: An Implementation and Evaluation of Gradient-Routed Auxiliary Modules (GRAM)**

### Project objective

Implement and experimentally evaluate a small decoder-only language model that uses a shared core and capability-specific auxiliary MLP modules. During training, route gradients according to the domain of each batch. At inference time, selectively disable auxiliary modules and measure whether their associated capabilities are suppressed while general and retained capabilities remain usable.

This project is an independent, small-scale research implementation inspired by *Modular Pretraining Enables Access Control* (arXiv:2607.08077). It is not intended to reproduce the paper's largest-scale experiments.

### Primary research question

> Can gradient-routed auxiliary modules localize specialized capabilities in a small language model, allowing those capabilities to be selectively suppressed through module ablation while preserving general-purpose performance?

### Selected capability domains

| Component | Training domain | Role |
|---|---|---|
| Core | General-purpose text | Shared, always-active capability |
| Auxiliary 1 | Python programming | Optional specialized capability |
| Auxiliary 2 | SQL and text-to-SQL | Optional specialized capability |

Python and SQL are selected because they support objective, held-out evaluation. The domains are not perfectly independent: general programming, SQL, and natural-language reasoning overlap. This overlap is a useful part of the evaluation rather than something to assume away.

### Scope

The project includes:

- A decoder-only Transformer baseline.
- A GRAM-style architecture with core and auxiliary MLPs.
- Dataset-dependent gradient routing.
- Separate optimizer states for core and auxiliary parameter groups.
- Training and inference-time module configurations.
- Baseline comparisons.
- Capability retention, suppression, and leakage evaluation.
- Adversarial recovery experiments using benign Python and SQL data.
- Hyperparameter sensitivity analysis.
- Reproducible experiments and a technical report.

The project does **not** include instruction-tuning robustness or representation-level interpretability extensions in this scope.

---

## 2. Research Questions and Hypotheses

### RQ1 — Capability localization

Does domain-dependent gradient routing cause Python- and SQL-specific training signals to be concentrated in their corresponding auxiliary modules?

**Hypothesis:** Auxiliary modules trained on their corresponding domains will contribute materially to domain performance, and inactive auxiliary modules will receive no updates on unrelated auxiliary batches.

### RQ2 — Selective capability suppression

Can disabling one auxiliary module reduce performance on its associated domain without materially damaging the other domain or core language modeling?

**Hypothesis:** Disabling the Python module will reduce Python performance more than SQL and core performance; disabling SQL will show the corresponding pattern.

### RQ3 — Comparison with alternative approaches

How does GRAM compare with a standard model trained on all domains, data-filtered models, and a post-hoc unlearning baseline?

**Hypothesis:** GRAM can provide useful capability selectivity while retaining performance on enabled domains. Whether it matches data filtering or outperforms post-hoc unlearning is an empirical question.

### RQ4 — Capability leakage and recovery

How much of a disabled capability remains accessible through the shared core or the other active auxiliary? How easily can the capability be recovered through additional fine-tuning?

**Hypothesis:** Some leakage may remain because capabilities share representations and training signals. Recovery performance will depend on the routing configuration, model capacity, data, and fine-tuning budget.

### RQ5 — Routing hyperparameters

How do the core-gradient probability and core-batch auxiliary activation probability affect the trade-off between retained performance and capability suppression?

**Hypothesis:** Lower shared-core gradient exposure on auxiliary batches may improve isolation but can reduce useful transfer. Core-batch auxiliary activation may improve robustness of the core path to auxiliary ablation.

---

## 3. Architecture

### 3.1 Baseline decoder-only Transformer

Implement a conventional autoregressive language model first. It should include:

- Token embeddings.
- Positional embeddings or a documented alternative.
- Causal self-attention.
- Residual connections.
- Layer normalization.
- Standard feed-forward MLP.
- Language-model output head.
- Next-token prediction objective.

For a pre-normalized Transformer block:

\[
h'_l = h_l + \operatorname{Attention}(\operatorname{LN}(h_l))
\]

\[
h_{l+1} = h'_l + \operatorname{MLP}(\operatorname{LN}(h'_l))
\]

The baseline must train, save checkpoints, reload, and generate text before GRAM changes are introduced.

### 3.2 GRAM MLP

Replace the standard MLP in each Transformer block with:

- One core MLP, \(E_c\).
- One Python auxiliary MLP, \(E_p\).
- One SQL auxiliary MLP, \(E_s\).

For layer input \(x_l\), the conceptual GRAM computation is:

\[
\operatorname{GRAM}(x_l)
=
E_c(x_l)
+
m_p E_p(x_l)
+
m_s E_s(x_l)
\]

where \(m_p,m_s \in \{0,1\}\) are the Python and SQL activation masks.

The Transformer residual update is:

\[
h_{l+1}=x_l+\operatorname{GRAM}(\operatorname{LN}(x_l))
\]

The exact normalization and residual placement must be consistent with the chosen baseline implementation. Do not accidentally change unrelated architecture components between baseline and GRAM.

### 3.3 Parameter ownership

For the initial implementation:

**Shared/core parameter group**

- Token embeddings.
- Positional embeddings, if learned.
- Attention projections in every layer.
- Layer normalization parameters.
- Core MLPs in every layer.
- Final normalization.
- Language-model output head.

**Python auxiliary parameter group**

- Python auxiliary MLP in every Transformer layer.

**SQL auxiliary parameter group**

- SQL auxiliary MLP in every Transformer layer.

This ownership definition is important: the core gradient-routing decision applies to the complete shared/core group, not just the core MLP.

### 3.4 Inference profiles

The model must expose the following configurations:

| Profile | Core | Python auxiliary | SQL auxiliary |
|---|---:|---:|---:|
| Full access | ON | ON | ON |
| Core only | ON | OFF | OFF |
| Python access | ON | ON | OFF |
| SQL access | ON | OFF | ON |

The core remains active in every profile. An inactive auxiliary must not contribute to the forward pass.

### 3.5 Initial model configuration

Use a small model suitable for development and modest GPU training.

| Hyperparameter | Initial value |
|---|---:|
| Transformer layers | 4 |
| Hidden dimension | 256 |
| Attention heads | 4 |
| Core MLP hidden dimension | 1,024 |
| Auxiliary MLP hidden dimension | 128 |
| Auxiliary modules | 2 |
| Context length | 256 |
| Vocabulary size | 8,000 initially |
| Objective | Causal next-token prediction |
| Optimizer | AdamW, separate parameter-group optimizers |

These are proposed starting values, not values that should be represented as the paper's exact configuration. Record the actual parameter count and revise dimensions if needed to fit the chosen compute budget.

Use a tokenizer trained or selected consistently across all domains. Do not train a different tokenizer for each capability, because that would introduce another source of capability separation.

---

## 4. Gradient Routing

Gradient routing is the central mechanism. The implementation must make the update rules explicit and testable.

### 4.1 Routing parameters

Use the following notation:

- \(p_{as}\): probability that the shared/core parameter group receives gradients on an auxiliary-domain batch.
- \(p_{cr}\): probability that an auxiliary module is activated on a core-domain batch.
- \(p_{af,i}\): sampling-frequency multiplier for auxiliary domain \(i\), if needed by the batch sampler.

Initial prototype values:

\[
p_{as}=0.3,\qquad p_{cr}=0.5
\]

Treat these as starting settings. The paper uses different settings in different experiments, so the implementation should make both configurable.

### 4.2 Core-domain batch

For a general-text batch:

1. Activate the core path.
2. With probability \(p_{cr}\), sample one auxiliary uniformly and activate it.
3. Compute the language-model loss.
4. Update the core parameter group.
5. If an auxiliary was activated, update that auxiliary parameter group.
6. Do not update the other auxiliary.

For \(K=2\), sample at most one auxiliary per core batch. The probability of selecting each auxiliary is \(p_{cr}/K\), and the probability of selecting none is \(1-p_{cr}\).

### 4.3 Python-domain batch

For a Python batch:

1. Activate the core path and Python auxiliary.
2. Keep the SQL auxiliary inactive.
3. Compute the language-model loss.
4. Always update the Python auxiliary parameter group.
5. Update the shared/core parameter group with probability \(p_{as}\).
6. Do not update the SQL auxiliary parameter group.

### 4.4 SQL-domain batch

Apply the same procedure as the Python batch, with SQL active and Python inactive.

### 4.5 Update matrix

| Batch domain | Core forward | Python forward | SQL forward | Core update | Python update | SQL update |
|---|---:|---:|---:|---:|---:|---:|
| Core | Yes | Probabilistic | Probabilistic | Always | If selected | If selected |
| Python | Yes | Yes | No | Probabilistic | Always | Never |
| SQL | Yes | No | Yes | Probabilistic | Never | Always |

### 4.6 Optimizer handling

Use separate AdamW optimizer instances, or an equivalently rigorous optimizer manager, for:

- Shared/core parameters.
- Python auxiliary parameters.
- SQL auxiliary parameters.

Only call `step()` for parameter groups selected for the current batch. Do not call `step()` on an inactive optimizer: AdamW weight decay and optimizer state can change parameters even if gradients are zero.

Before each training step:

- Clear gradients for all parameter groups.
- Construct the correct active-module configuration.
- Compute the loss and backpropagate.
- Step only the selected optimizers.
- Record which groups were updated.

For an auxiliary batch where the core is not selected for update, the core can remain in the forward computation graph. Do not detach the core output merely to prevent core parameter updates: doing so can prevent gradients from flowing through the core computation into the active auxiliary. Instead, allow autograd to compute gradients and simply do not step the core optimizer. Clear its gradients before the next step.

This approach preserves the ability of the active auxiliary to learn through its own branch while keeping the shared/core parameters unchanged on that step.

### 4.7 Gradient-routing correctness

The implementation must log:

- Active modules per batch.
- Selected optimizer groups.
- Per-group gradient norms before optimizer steps.
- Per-group parameter changes after optimizer steps.
- Number of updates received by each group.

Distinguish between a parameter group receiving a computed gradient and its optimizer being stepped. The latter determines whether its parameters change.

---

## 5. Data Plan

### 5.1 Domains

Use three labeled data domains:

**Core: general-purpose text**

Purpose:
- General language modeling.
- General instruction-like or explanatory prose only if it is included consistently in the selected corpus.
- General-domain retention evaluation.

**Python: programming**

Purpose:
- Python code modeling.
- Code continuation.
- Simple function implementation.
- Basic debugging or code explanation, if represented in the training format.

**SQL: text-to-SQL and SQL programming**

Purpose:
- SQL query modeling.
- SQL continuation.
- Text-to-SQL generation.
- Query structure and schema-conditioned tasks, if represented consistently.

### 5.2 Candidate datasets

Potential sources include:

- TinyStories: `roneneldan/TinyStories` for a small general-language corpus.
- CodeSearchNet or a carefully filtered Python corpus for Python.
- Spider for text-to-SQL evaluation and suitable training examples.
- Carefully generated synthetic Python and SQL examples for controlled experiments.

Dataset availability, licenses, splits, and suitability must be checked before use. The project should record dataset names, versions, licenses, preprocessing steps, and source URLs.

### 5.3 Important dataset-design caveat

TinyStories is stylistically and semantically different from code datasets. A model may learn domain identity from formatting and token distribution rather than capability boundaries alone. This is acceptable for an initial engineering test, but the report must discuss it.

For a stronger controlled experiment:

- Use comparable sequence-length distributions where practical.
- Keep tokenizer and preprocessing common across domains.
- Deduplicate across all domains.
- Ensure that near-duplicate examples do not cross train/validation/test splits.
- Keep evaluation tasks and templates separate from training templates.
- Include examples that test shared concepts and domain overlap.

### 5.4 Data splits

Create separate train, validation, and test splits for every domain.

Split by underlying task, source, or template where possible, not merely by random rows. This reduces leakage from near-duplicate examples.

Store processed metadata, including:

- Domain label.
- Source dataset and version.
- Original sample identifier where available.
- Split assignment.
- Token count.
- Preprocessing version.

### 5.5 Batch sampling

The training sampler should:

- Emit a domain label with every batch.
- Support configurable domain proportions.
- Apply \(p_{af,i}\) if auxiliary sampling needs to be adjusted.
- Avoid silently oversampling a domain without recording it.
- Log actual tokens and batches consumed per domain.

Report both total training tokens and per-domain training tokens.

---

## 6. Training Pipeline

### 6.1 Training objective

Use causal language modeling:

\[
\mathcal{L}(\theta)
=
-\frac{1}{T}\sum_{t=1}^{T}
\log P_\theta(x_t\mid x_{<t})
\]

Use the same tokenizer, sequence construction, and evaluation conventions across comparable models.

### 6.2 Training loop responsibilities

The trainer must:

1. Load a domain-labeled batch.
2. Select the active auxiliary configuration.
3. Select the optimizer groups that may update.
4. Run the forward pass.
5. Compute next-token loss.
6. Backpropagate.
7. Step only selected optimizers.
8. Record losses, gradient norms, update counts, and routing decisions.
9. Run validation according to a fixed evaluation policy.
10. Save reproducible checkpoints.

### 6.3 Checkpoint contents

Each checkpoint should include:

- Model state dictionary.
- Core optimizer state.
- Python optimizer state.
- SQL optimizer state.
- Training step and token counts.
- Configuration values.
- Random-number-generator states.
- Tokenizer identifier or files.
- Dataset and preprocessing identifiers.
- Validation metrics.

Support resuming training without resetting optimizer state or routing randomness.

### 6.4 Reproducibility

- Set and record Python, NumPy, and PyTorch seeds.
- Record library versions and device information.
- Save configuration files with each run.
- Use multiple seeds for final comparisons.
- Keep training and evaluation splits fixed across comparable experiments.

---

## 7. Baselines and Comparisons

At minimum, implement the following systems.

### Baseline A — Standard Transformer trained on all domains

A conventional Transformer trained on the combined core, Python, and SQL data.

Purpose:
- Establish full-model capability performance.
- Measure the effect of auxiliary ablation relative to an ordinary shared model.

### Baseline B — Data-filtered Transformer

Train a conventional Transformer without the domain intended to be removed.

Examples:
- Core + SQL, with Python data excluded.
- Core + Python, with SQL data excluded.
- Core only.

Purpose:
- Estimate the performance of a model that did not train on the removed domain.

### Baseline C — GRAM

Train the proposed model with domain-dependent gradient routing.

Evaluate all four inference profiles.

### Baseline D — Post-hoc unlearning

Train a conventional model on all domains, then apply a documented unlearning method targeting Python or SQL.

Purpose:
- Compare architectural module ablation with a post-training removal approach.

Document the selected unlearning method and its hyperparameters. Do not describe a simple fine-tuning procedure as a strong unlearning baseline unless its limitations are explicit.

### Fair-comparison requirements

Report:

- Total and active parameter counts.
- Training tokens by domain.
- Training steps.
- Approximate training compute, if available.
- Optimizer and learning-rate settings.
- Number of random seeds.
- Evaluation data and metric definitions.

A GRAM model has additional parameters compared with a standard model if auxiliary branches are simply added. Include parameter-matched or compute-aware comparisons where practical, and state any remaining mismatch.

---

## 8. Evaluation Plan

Evaluate every model on held-out examples from all three domains.

### 8.1 Core language modeling

Metrics:
- Validation negative log-likelihood.
- Perplexity.
- Optionally, a small fixed set of general-language completion evaluations.

Purpose:
- Check whether module ablation damages general language modeling.

### 8.2 Python capability

Metrics:
- Held-out next-token loss/perplexity on Python data.
- Exact-match or unit-test pass rate for generated code tasks.
- Syntax validity, where relevant.

Use sandboxed execution for generated code evaluation. Restrict filesystem, network, time, and resource access. Do not execute untrusted generated code directly on the host.

### 8.3 SQL capability

Metrics:
- Held-out SQL loss/perplexity.
- Exact-match query accuracy.
- Execution accuracy on a controlled database benchmark.

Use isolated, read-only or disposable databases for evaluation. Normalize SQL consistently and document handling of equivalent queries.

### 8.4 Capability-profile matrix

Evaluate:

| Profile | Core test | Python test | SQL test |
|---|---|---|---|
| Full access | Yes | Yes | Yes |
| Core only | Yes | Yes | Yes |
| Python access | Yes | Yes | Yes |
| SQL access | Yes | Yes | Yes |

Testing every domain under every profile is essential. Do not only report the domain that was disabled.

### 8.5 Retention and suppression

For a capability \(d\), define a performance retention ratio:

\[
R_d =
\frac{A_{d,\text{profile}}}
{A_{d,\text{full}}}
\]

where \(A\) is a higher-is-better metric such as accuracy. For loss metrics, report the raw loss and an explicitly defined loss difference instead of using this ratio.

For the disabled capability, report:

\[
\Delta L_d =
L_{d,\text{disabled}} - L_{d,\text{full}}
\]

A positive value indicates increased loss after ablation.

Do not rely on a single metric. A model may have low task accuracy but still retain substantial domain knowledge in its token predictions.

### 8.6 Capability leakage

A simple task-level leakage ratio is:

\[
\operatorname{Leakage}_d =
\frac{A_{d,\text{disabled}}}
{A_{d,\text{full}}}
\]

Lower values indicate greater suppression, assuming the full-model denominator is meaningful.

Also report:
- Disabled-domain loss.
- Disabled-domain task accuracy.
- Performance on the other enabled domain.
- Core performance.
- Confidence intervals or variation across seeds.

Ablation is evidence of reduced access through the tested architecture path, not proof that information has been erased from the model.

---

## 9. Adversarial Recovery Experiment

Use benign Python and SQL tasks to test whether a disabled capability can be recovered through additional training.

### Procedure

For each target capability:

1. Train the GRAM model.
2. Select an inference profile with the target auxiliary disabled.
3. Measure baseline disabled performance.
4. Fine-tune the remaining active model parameters on a small, separate target-domain dataset.
5. Evaluate recovery on a held-out target-domain test set.
6. Repeat with multiple fine-tuning data budgets.
7. Compare with the relevant standard and post-hoc unlearning baselines.

Example fine-tuning budgets can be expressed as numbers of examples or tokens. Choose the actual budgets after inspecting dataset size and compute constraints, and keep them consistent across comparable models.

### Recovery ratio

For a higher-is-better accuracy metric:

\[
\operatorname{RecoveryRatio}
=
\frac{A_{\text{after FT}}-A_{\text{disabled}}}
{A_{\text{full}}-A_{\text{disabled}}}
\]

Interpretation:
- Near 0: little recovery.
- Near 1: recovery to approximately full-model performance.
- Above 1: performance exceeds the original full-model score.

If the denominator is close to zero, do not report this ratio; report raw performance instead.

### Important controls

- Keep the disabled auxiliary physically or logically inactive during evaluation.
- State exactly which parameters are allowed to update during fine-tuning.
- Report whether the auxiliary parameters are frozen, absent, or merely masked.
- Use held-out recovery evaluation examples.
- Compare equal fine-tuning data and compute budgets.
- Do not claim that recovery resistance establishes a security guarantee.

---

## 10. Hyperparameter and Ablation Studies

### 10.1 Routing sensitivity

Evaluate a small grid such as:

\[
p_{as}\in\{0,\ 0.3,\ 0.5,\ 1.0\}
\]

\[
p_{cr}\in\{0,\ 0.2,\ 0.5,\ 0.8\}
\]

These are proposed experiment values. The full grid may be reduced if compute is constrained.

Measure:
- Core validation loss.
- Python performance under full and ablated profiles.
- SQL performance under full and ablated profiles.
- Leakage ratios.
- Gradient norms and update counts.

### 10.2 Auxiliary capacity

Compare a small number of auxiliary hidden dimensions, for example:

\[
d_{\text{aux}}\in\{64,\ 128,\ 256\}
\]

Measure whether increased auxiliary capacity improves domain performance and suppression after ablation, while accounting for parameter count.

### 10.3 Core robustness

Compare different \(p_{cr}\) values to determine whether exposing the core path to randomly selected auxiliary branches during core-domain training improves performance when auxiliaries are later disabled.

### 10.4 Shared-gradient exposure

Compare different \(p_{as}\) values to examine the trade-off between shared learning and capability isolation.

### 10.5 Required ablation checks

- Core only.
- Core + Python.
- Core + SQL.
- Core + both auxiliaries.
- Auxiliary branches present but inactive.
- Auxiliary branches removed from the computation graph, where technically comparable.

---

## 11. Software Design and Repository Structure

Recommended repository:

```text
gram-access-control/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── model.yaml
│   ├── data.yaml
│   ├── training.yaml
│   └── experiments.yaml
│
├── data/
│   ├── download.py
│   ├── preprocess.py
│   ├── tokenizer.py
│   └── dataset.py
│
├── models/
│   ├── attention.py
│   ├── transformer.py
│   ├── gram_mlp.py
│   └── gram_model.py
│
├── training/
│   ├── trainer.py
│   ├── gradient_router.py
│   ├── optimizer_manager.py
│   └── checkpoint.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── capability_eval.py
│   ├── ablation_eval.py
│   ├── leakage_eval.py
│   └── recovery_eval.py
│
├── experiments/
│   ├── baseline/
│   ├── gram/
│   ├── data_filtered/
│   ├── posthoc_unlearning/
│   ├── ablation/
│   └── recovery/
│
├── tests/
│   ├── test_attention.py
│   ├── test_architecture.py
│   ├── test_gradient_routing.py
│   ├── test_optimizer_updates.py
│   ├── test_ablation.py
│   └── test_reproducibility.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiments.py
│
└── results/
    ├── metrics/
    ├── plots/
    └── reports/
```

### Module responsibilities

**`models/gram_mlp.py`**
- Core and auxiliary MLP definitions.
- Module activation interface.
- Clear separation of parameter groups.

**`models/gram_model.py`**
- Full Transformer integration.
- Inference profile selection.
- Consistent auxiliary activation across layers.

**`training/gradient_router.py`**
- Domain-to-routing decisions.
- Sampling of auxiliary modules on core batches.
- Core update probability on auxiliary batches.
- Logging of routing choices.

**`training/optimizer_manager.py`**
- Separate optimizer state.
- Conditional optimizer stepping.
- Gradient clearing and update accounting.

**`evaluation/ablation_eval.py`**
- Runs all inference profiles.
- Produces a domain-by-profile result matrix.

**`evaluation/recovery_eval.py`**
- Runs controlled fine-tuning recovery experiments.
- Tracks which parameters are allowed to update.

---

## 12. Unit Tests and Acceptance Criteria

The implementation should not be considered ready for full experiments until the following tests pass.

### Architecture tests

- The baseline Transformer produces correctly shaped logits.
- Causal attention does not use future tokens.
- GRAM produces correctly shaped logits.
- Every layer contains the expected core and auxiliary modules.
- Each inference profile activates the intended branches.
- The full profile includes all branches.
- The core-only profile excludes both auxiliary contributions.

### Gradient-routing tests

For a Python batch:
- Python auxiliary parameters receive gradients.
- SQL auxiliary parameters receive no gradients.
- Core gradients may be computed, but the core optimizer is stepped only when selected.

For a SQL batch:
- SQL auxiliary parameters receive gradients.
- Python auxiliary parameters receive no gradients.
- Core optimizer stepping follows the routing decision.

For a core batch:
- Core optimizer is always stepped.
- At most one auxiliary is selected.
- Only the selected auxiliary optimizer is stepped.

### Optimizer tests

- Inactive optimizer states do not advance.
- Inactive parameter values remain unchanged.
- AdamW weight decay does not alter inactive parameters.
- Optimizer states are restored correctly from checkpoints.
- Gradients are cleared between batches.

### Ablation tests

- Disabled modules do not contribute to forward outputs.
- Repeated inference with the same configuration is deterministic when sampling and stochastic layers are disabled.
- Switching profiles does not mutate model weights.
- Full-access inference is unchanged by merely evaluating another profile.

### Experimental acceptance criteria

These are project goals, not guaranteed outcomes:

- Core performance remains close to the full GRAM model after auxiliary ablation.
- The disabled domain shows measurable degradation.
- The other enabled domain retains substantially more performance than the disabled domain.
- Results are reproducible across multiple seeds.
- Baselines use documented and comparable data and compute budgets.
- All negative or inconclusive findings are reported.

Do not set success solely by hitting an arbitrary suppression percentage. Report the complete retention-versus-suppression trade-off.

---

## 13. Compute and Engineering Constraints

The initial target is a small model of approximately 5–10M parameters, subject to the actual parameter count after implementation.

Suggested environment split:

| Workload | Environment |
|---|---|
| Architecture development | Local Apple Silicon Mac |
| Unit tests | Local Mac |
| Small synthetic tests | Local Mac |
| Main model training | Colab GPU or equivalent |
| Hyperparameter experiments | GPU environment |
| Evaluation and plotting | Local Mac or GPU environment |

Engineering requirements:

- Use small batches and gradient accumulation where needed.
- Keep sequence length configurable.
- Use mixed precision only when supported and numerically stable.
- Save checkpoints and metrics frequently.
- Profile memory before increasing model size.
- Keep data preprocessing deterministic.
- Avoid relying on a single training run for final claims.

---

## 14. Experiment Tracking and Reporting

Every run should record:

- Experiment identifier.
- Git commit or code version.
- Configuration file.
- Random seed.
- Dataset identifiers and versions.
- Tokenizer identifier.
- Parameter count by group.
- Training tokens by domain.
- Optimizer and learning-rate configuration.
- \(p_{as}\), \(p_{cr}\), and \(p_{af}\).
- Active module frequencies.
- Per-group optimizer update counts.
- Training and validation losses.
- Evaluation metrics for every domain and profile.
- Hardware and software environment.
- Checkpoint location.

CSV and JSON logs are sufficient initially. A tracking platform can be added if it materially improves experiment management.

### Required figures

1. Training and validation loss curves.
2. Core, Python, and SQL performance across inference profiles.
3. Retention-versus-suppression plot.
4. Leakage comparison across model approaches.
5. Recovery versus fine-tuning budget.
6. Performance versus \(p_{as}\).
7. Performance versus \(p_{cr}\).
8. Parameter-group gradient norms and optimizer update counts.

### Required result tables

- Model and parameter-count comparison.
- Full inference-profile evaluation matrix.
- Baseline comparison.
- Routing hyperparameter sensitivity.
- Adversarial recovery results.
- Multi-seed mean and standard deviation.

---

## 15. Risks and Limitations

### Capability overlap

Python and SQL share programming concepts, and both rely on general language understanding. A capability may remain accessible through the core or the other auxiliary. Measure this explicitly.

### Domain and formatting shortcuts

The model may distinguish code from prose through superficial formatting. Include varied examples and held-out task templates.

### Small model capacity

A small model may not learn sufficiently strong capabilities for meaningful ablation. Verify full-model competence before interpreting suppression results.

### Shared-core leakage

Auxiliary-domain examples can update shared attention, embeddings, normalization, and output parameters when the core update is selected. Gradient routing does not guarantee that all domain information is isolated.

### Post-training effects

This scope does not evaluate instruction-tuning robustness. Do not generalize results to instruction-tuned or frontier-scale models.

### Security interpretation

Module ablation is an architectural access-control mechanism under evaluation, not a formal security boundary. A disabled capability may remain partially recoverable from shared parameters or other active modules.

### Compute and statistical uncertainty

Small experiments can be noisy. Report seeds, uncertainty, and compute budgets. Distinguish observed results from hypotheses.

---

## 16. Final Deliverables

The completed project should include:

1. A working standard decoder-only Transformer baseline.
2. A GRAM-style Transformer with core, Python, and SQL MLP branches.
3. A tested gradient-routing implementation.
4. Separate optimizer states and conditional optimizer stepping.
5. A reproducible data pipeline with domain labels.
6. Four inference-time capability profiles.
7. Standard, data-filtered, and post-hoc unlearning baselines.
8. Held-out evaluation for core, Python, and SQL.
9. Capability retention, suppression, and leakage analyses.
10. Controlled adversarial recovery experiments using benign data.
11. Routing and auxiliary-capacity sensitivity experiments.
12. Reproducible configurations, checkpoints, logs, and plots.
13. A technical report describing methodology, results, limitations, and failed experiments.
14. A README with setup, training, evaluation, and reproduction instructions.

---

## 17. Reference

**Primary paper**

*Modular Pretraining Enables Access Control.* arXiv:2607.08077.

- Paper: <https://arxiv.org/pdf/2607.08077>
- arXiv abstract and metadata: <https://arxiv.org/abs/2607.08077>

**Implementation references**

- PyTorch Autograd: <https://pytorch.org/docs/stable/autograd.html>
- PyTorch AdamW: <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>
- nanoGPT: <https://github.com/karpathy/nanoGPT>
- TinyStories: <https://huggingface.co/datasets/roneneldan/TinyStories>
- CodeSearchNet: <https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet>
- Spider benchmark: <https://yale-lily.github.io/spider>

---

## 18. Definition of Done

The project is complete when another researcher can:

1. Install the documented dependencies.
2. Reproduce the baseline and GRAM model configurations.
3. Train the model using the documented domain-routing procedure.
4. Verify gradient-routing behavior through automated tests.
5. Load a checkpoint and select any supported inference profile.
6. Reproduce the evaluation matrix and baseline comparisons.
7. Inspect capability leakage and recovery results.
8. Trace every reported result to a configuration, dataset version, seed, and checkpoint.
9. Understand the limitations without interpreting module ablation as guaranteed erasure or security.

**Final research principle:** The central claim to test is not merely that auxiliary modules can be switched off. It is whether gradient routing creates useful, measurable capability separation while preserving the model's other capabilities—and how robust that separation is to leakage and recovery.
