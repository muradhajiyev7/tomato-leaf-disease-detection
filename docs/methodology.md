# Methodology

## Data Preparation

The dataset was organized into train, validation, and test splits. Class folders were audited to verify split structure and image counts before training.

## Augmentation

Image augmentation was used to improve robustness to variations in lighting, orientation, scale, and capture conditions. The final public source files include augmentation placeholders that can be adapted for local training.

## EfficientNet

EfficientNet backbones were evaluated for tomato leaf disease classification. EfficientNet-B3 was selected for the final disease model pipeline.

## Backbone Experiment and Validation

The first notebook documents model comparison, dataset loading, training/validation setup, evaluation logic, and the reasoning that moved the project toward the final pipeline.

## Final Inference Pipeline

The second notebook documents the final system workflow: preprocessing, model prediction, calibration, TTA, accept/reject logic, evaluation, and mobile integration notes.

## Calibration

Confidence calibration was used to improve how well predicted confidence values matched observed correctness. This matters for deployment because a high-confidence wrong prediction can be more harmful than a low-confidence uncertain prediction.

## Test-Time Augmentation

Test-time augmentation combines predictions from multiple transformed versions of the same image. This can improve stability when image orientation or crop framing varies.

## Accept/Reject Logic

The final pipeline used confidence and uncertainty-margin thresholds to accept confident predictions and reject uncertain cases. This created a tradeoff between coverage and reliability.
