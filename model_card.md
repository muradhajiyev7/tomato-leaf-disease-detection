# Model Card: Tomato Leaf Disease Classifier

## Model Details

This project uses an EfficientNet-based image classifier for tomato leaf disease classification. The final disease model used an EfficientNet-B3 backbone in a PyTorch workflow.

## Intended Use

The model is intended to support tomato leaf disease screening from image inputs in educational, portfolio, and prototype greenhouse-testing contexts. It should be treated as decision-support tooling, not as a replacement for expert agricultural diagnosis.

## Dataset Summary

The local project dataset contains approximately 40K tomato leaf images organized into train, validation, and test splits. The public GitHub repository excludes the raw dataset because of size and distribution concerns.

## Training and Validation Setup

Training and final evaluation were performed in a high-performance computing environment. The public repository keeps cleaned notebooks and modular source templates for transparency, but does not include checkpoints, trained weights, or environment-specific runtime artifacts.

## Evaluation Metrics

| Metric | Value |
|---|---:|
| Test Accuracy | 94.42% |
| Macro-F1 | 91.99% |
| Accepted-case Accuracy | 96.35% |
| Accepted-case Macro-F1 | 94.62% |
| Coverage | 96.71% |

## Calibration and TTA

The final pipeline included confidence calibration and test-time augmentation. Calibration was used to improve the reliability of confidence scores, while TTA aggregated predictions from multiple transformed views of the same input image.

## Accepted-Case Evaluation

The system used confidence and uncertainty-margin logic to accept high-confidence predictions and reject uncertain cases. Accepted-case metrics report performance on the subset of predictions accepted by that decision rule.

## Limitations

- Performance can degrade under low-quality images, unusual lighting, occlusion, or unseen field conditions.
- Dataset composition may not represent every greenhouse or farm environment.
- Similar diseases can be visually difficult to distinguish.
- The model should not be used as the only basis for crop treatment decisions.

## Ethical and Practical Considerations

Incorrect disease predictions can lead to delayed treatment or unnecessary intervention. Users should validate important decisions with agricultural experts and local field observations.

## Deployment Notes

The public repository excludes model weights. To deploy the model, place compatible weights under `models/` and update `MODEL_PATH` in `src/config.py`.

## Mobile App Integration

The project was connected to a Flutter/Firebase mobile application for greenhouse testing, with image capture/upload and prediction display workflows.
