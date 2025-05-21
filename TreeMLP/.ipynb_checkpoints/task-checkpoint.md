# 📑 Project Instructions – OpenML Regression Benchmark 2025

Welcome! This file walks you through everything you need to reproduce, extend, and maintain the **OpenML Regression Benchmark 2025** workflow.

---
## 1  Overview
The project benchmarks several regression models on every dataset in the `OpenML-CTR23 (8f0ea660163b436bbd4abd49665c7b1d)` suite and logs results to **Weights & Biases (W&B)**. It also supports Bayesian hyper‑parameter sweeps and tracks full data/model lineage via W&B Artifacts.

---
## 2  Prerequisites
| Tool | Version | Notes |
|------|---------|-------|
| Python | ≥ 3.9 | Tested on 3.11 |
| `openml` | ≥ 0.14 | Access OpenML REST API |
| `wandb` | ≥ 0.17 | Experiment tracking + sweeps |
| `scikit‑learn` | ≥ 1.3 | ML algorithms & preprocessing |
| `pandas`/`numpy` | latest | Data wrangling |
| `joblib` | ≥ 1.3 | Caching pipelines |

**Install** (conda or pip):
```bash
pip install openml wandb scikit-learn pandas numpy joblib
```

---
## 3  W&B setup
1. Sign up at <https://wandb.ai> (free tier is fine).
2. Grab your API key (`wandb login` CLI or from the web UI).
3. Either:
   * Set the environment variable `WANDB_API_KEY` **once**:
     ```bash
     export WANDB_API_KEY=xxxxxxxxxxxxxxxx
     ```
   * Or run `wandb.login()` in the notebook and paste the key when prompted.

All runs will appear under the project name **`openml_regression_benchmark_2025`** (configurable at the top of the notebook).

---
## 4  Running the benchmark notebook
1. Open **`openml_regression_benchmark.ipynb`**.
2. _Kernel ▶ Restart & Run All_.
3. The notebook will:
   * Pull the OpenML suite metadata.
   * Iterate over tasks ➜ datasets.
   * Train baseline models with cross‑validation.
   * Log metrics, diagnostic plots, permutation importance, and artifacts to W&B.

**Tip**: Set `suite.tasks[:N]` inside the loop to limit execution during local testing.

---
## 5  Hyper‑parameter sweeps
1. Scroll to **“Define and run a W&B sweep”**.
2. Uncomment the last line and tweak `count` or spawn multiple agents:
   ```python
   wandb.agent(sweep_id, function=sweep_train, count=20)
   ```
3. Monitor sweep progress in the W&B UI ➜ **Sweeps** tab.

**Customising the search** – edit `sweep_config`:
* Add/remove parameters or change priors.
* Expand `dataset_idx` list to sweep across more datasets.

---
## 6  Caching & performance
* All `ColumnTransformer` / encoder fits are cached under `/tmp/sk_cache` via **`joblib.Memory`**. Delete that directory if you change preprocessing logic.
* `cross_validate(..., n_jobs=-1)` uses every CPU core.

---
## 7  Extending the benchmark
| Task | Where to change |
|------|-----------------|
| Add a new baseline model | `baseline_models` dict in the “Benchmark loop” cell |
| Use GPU libraries | Swap estimators for RAPIDS / LightGBM, ensure numeric inputs |
| Change metrics | Append to the global `scoring` dict |
| Add SHAP / Lime explainers | Insert after `log_permutation_importance()` |

---
## 8  Troubleshooting
* **“X contains values that are not numbers”** → Check that `OneHotEncoder(..., sparse_output=False)` is in the preprocessing pipeline.
* **Sweep stalls at 0%** → Your agent isn’t running. Run the `wandb.agent(...)` cell or launch another terminal with `wandb agent <entity/project/sweepid>`.
* **OpenML API quota exceeded** → Cache datasets locally with `openml.config.cache_directory`.

---
## 9  Next steps
* Hook the notebook into CI with `papermill` to ensure it stays green.
* Export the notebook to a script via `jupyter nbconvert --to python` for headless runs.
* Publish results by embedding W&B Reports in your doc/wiki.

---