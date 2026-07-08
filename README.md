# Tomato Leaf Disease Detection System

An end-to-end computer vision system for tomato leaf disease classification using PyTorch and EfficientNet, with calibration, test-time augmentation, confidence-based decision logic, and mobile app integration for greenhouse testing.

## Overview

Tomato leaf diseases can reduce crop yield when they are detected late. This project explores how deep learning can support farmers, greenhouse operators, and agricultural teams by classifying tomato leaf conditions from images and returning confidence-aware predictions.

The full training and final evaluation were performed in a high-performance computing environment. This public repository is a cleaned portfolio version: it includes documentation, notebooks, reusable code structure, and examples, but excludes the large raw dataset, trained model weights, checkpoints, and HPC-specific runtime artifacts.

## Key Highlights

- 36K+ tomato leaf images across train, validation, and test splits
- EfficientNet-based classifier, with EfficientNet-B3 selected for the final disease model
- 94.42% final test accuracy
- 91.99% final macro-F1
- 96.35% accepted-case accuracy
- 94.62% accepted-case macro-F1
- 96.71% accepted-case coverage
- Confidence calibration and test-time augmentation
- Confidence/margin-based accept/reject decision logic
- Mobile application integration with Flutter and Firebase

## Results

| Metric | Value |
|---|---:|
| Test Accuracy | 94.42% |
| Macro-F1 | 91.99% |
| Accepted-case Accuracy | 96.35% |
| Accepted-case Macro-F1 | 94.62% |
| Coverage | 96.71% |

## System Pipeline

Image input -> preprocessing -> EfficientNet classifier -> calibration -> test-time augmentation -> confidence/margin decision -> prediction result -> mobile app display

## Two-Stage Project Design

**Stage 1: Backbone Experiment and Validation**  
The first stage compared model backbone options, dataset loading, training/validation setup, evaluation metrics, and key model-selection takeaways.

**Stage 2: Final Inference Pipeline with Calibration and TTA**  
The second stage built the final confidence-aware inference workflow, including calibration, test-time augmentation, accept/reject logic, final evaluation, and mobile integration notes.

## Repository Structure

```text
tomato-leaf-disease-detection/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── model_card.md
├── dataset_structure.py
├── docs/
├── notebooks/
│   ├── backbone_experiment_and_validation.ipynb
│   ├── final_inference_pipeline_calibration_tta.ipynb
│   └── archive/
├── src/
├── examples/
├── assets/
├── tests/
└── outputs/
```

## Methodology

The project used a structured image-classification workflow: dataset preparation, class-level audit, image augmentation, EfficientNet training, validation/evaluation, confidence calibration, test-time augmentation, and confidence-based acceptance/rejection.

Macro-F1 was used alongside accuracy because disease datasets can have class imbalance. The final accepted-case metrics evaluate how the system behaves when it only returns predictions above the confidence/margin threshold.

## Mobile Application

The model pipeline was connected to a mobile application for greenhouse testing. The mobile side used a Flutter frontend with Firebase integration, supporting a camera/upload workflow and a prediction result screen. Screenshots can be placed under `assets/mobile_app_screenshots/`.

## How to Run

This public repository does not include the full dataset or trained model weights. To run inference locally, place compatible weights under `models/` and update `MODEL_PATH` in `src/config.py`.

```bash
git clone https://github.com/muradhajiyev7/tomato-leaf-disease-detection.git
cd tomato-leaf-disease-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/inference_example.py --image examples/sample_images/sample_leaf.jpg
```

If weights are missing, the example script prints a helpful message instead of failing silently.

## Dataset and Model Weights

Large image datasets and trained model weights are not included in this repository.

Place the dataset locally under:

```text
data/raw/Universal_Tomato_Dataset/
```

Expected split structure:

```text
Universal_Tomato_Dataset/
├── train/
├── val/
└── test/
```

Model weights can be placed under:

```text
models/
```

## Limitations

- Image quality, lighting, blur, and leaf positioning can affect predictions.
- Real greenhouse conditions vary from curated dataset conditions.
- This system is not a replacement for expert agronomist diagnosis.
- More field data can improve robustness.

## Future Work

- Grad-CAM explainability
- Disease severity estimation
- Cloud API deployment
- More real field images
- Continuous learning loop
- Mobile optimization

## Portfolio Note

This repository is a cleaned public portfolio version prepared for demonstration and recruitment purposes.
