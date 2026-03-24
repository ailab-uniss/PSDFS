# PSDFS — Paired Signed-Deviation Feature Selection

This repository contains the reference Python implementation of **PSDFS**, an embedded method for **multi-label feature selection (MLFS)**.

PSDFS produces a **single feature ranking per training fold** by combining:

- a **nonnegative reconstruction** template with group sparsity (stable multiplicative updates),
- a **signed-deviation feature lift** to represent *two-sided evidence* under nonnegativity,
- a **paired group-sparsity** penalty to avoid lift-induced split artefacts and rank features in the original space.

## Repository contents

| File | Description |
|---|---|
| `src/psdfs.py` | Core PSDFS algorithm: `PSDFSParams` dataclass and `psdfs()` function |
| `src/mlknn_gpu.py` | GPU-accelerated ML-kNN evaluation utility (requires CuPy) |

## Requirements

- **Python ≥ 3.9**
- **NumPy** (the only dependency for PSDFS itself)
- **CuPy** (optional, only needed for the GPU ML-kNN utility in `mlknn_gpu.py`)

## Quick start

```python
import numpy as np
from psdfs import PSDFSParams, psdfs

# X: (n_samples, n_features) float, nonneg, preferably scaled to [0,1]
# Y: (n_samples, n_labels) in {0, 1}

params = PSDFSParams(
    beta=0.10,
    max_iter=40,
    feature_lift="center_split",
    paired_penalty=True,
)

ranking, W, info = psdfs(X, Y, params)
print("Top-10 features (1-based):", ranking[:10])
```

`ranking` is a **1-based** permutation of feature indices. Use `ranking - 1` for 0-based Python indexing.

## CLI Usage

You can also run PSDFS directly from the command line using the built-in CLI.

**1. Run a quick demo:**
To verify everything is working without needing any dataset files, run the synthetic demo:
```bash
python src/psdfs.py demo
```

**2. Extract rankings from your own datasets:**
You can pass `.mat` or `.npz` files directly to the CLI to extract and save the feature ranking. The `--minmax-01` flag automatically scales your features to `[0,1]` before running PSDFS.
```bash
python src/psdfs.py rank \
  --mat my_dataset.mat \
  --beta 0.10 \
  --minmax-01 \
  --out-ranking results/my_ranking.csv
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `beta` | 0.10 | Sparsity strength (larger ⇒ stronger shrinkage) |
| `max_iter` | 40 | Number of multiplicative update iterations |
| `feature_lift` | `"center_split"` | `"none"` or `"center_split"` |
| `lift_center` | `"mean"` | `"mean"` or `"median"` (used when lift is `"center_split"`) |
| `paired_penalty` | `True` | Couple (j⁺, j⁻) pairs for sparsity and ranking |
| `seed` | 0 | Random seed for W initialisation |

## Reproducibility notes

- **Training-only preprocessing:** all statistics used by PSDFS (lift centering) are computed on the training fold only.
- **Scaling:** PSDFS expects nonneg data. We recommend **min–max scaling to [0, 1] per fold** (fit on train, apply to test).
- **Folds:** for multi-label benchmarks, use **iterative stratification** to preserve label proportions across folds.

## Citation

If you use this code, please cite the accompanying paper.

## License

MIT License. See [LICENSE](LICENSE).
