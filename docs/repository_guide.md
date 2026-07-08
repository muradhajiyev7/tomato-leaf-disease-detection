# Repository Guide

## Folder Contents

- `docs/`: project documentation, methodology, results, and mobile app notes
- `notebooks/`: cleaned public notebooks for the two main project stages
- `notebooks/archive/`: preserved backup notebooks with outputs cleared
- `src/`: reusable Python modules for preprocessing, dataset loading, modeling, evaluation, calibration, TTA, and inference
- `examples/`: command-line inference example and sample image folder
- `assets/`: README images, diagrams, result visuals, and mobile screenshots
- `tests/`: lightweight tests that do not require the full dataset or model weights
- `outputs/`: local output folder kept out of Git except for `.gitkeep`

## Adding Data Locally

Large datasets are excluded from GitHub. Place the dataset locally under:

```text
data/raw/Universal_Tomato_Dataset/
```

## Adding Model Weights Locally

Model weights are excluded from GitHub. Place compatible weights under:

```text
models/
```

Then update `MODEL_PATH` in `src/config.py`.

## Excluded from GitHub

- Raw image dataset
- Trained weights and checkpoints
- Generated outputs
- Archives and compressed files
- Private paths, credentials, API keys, and environment files
- Environment-specific runtime artifacts

## Using Notebooks

The notebooks are cleaned public versions and are not intended to be re-executed on a regular laptop without the dataset, weights, and proper compute environment. Full training and final evaluation were originally run in a high-performance computing environment.
