# Results

## Final Metrics

| Metric | Value |
|---|---:|
| Test Accuracy | 94.42% |
| Macro-F1 | 91.99% |
| Accepted-case Accuracy | 96.35% |
| Accepted-case Macro-F1 | 94.62% |
| Coverage | 96.71% |

## Accepted-Case Evaluation

Accepted-case evaluation measures the model only on predictions that pass confidence and uncertainty-margin thresholds. In this project, accepted-case performance improved relative to the full test set while preserving 96.71% coverage.

## Why Calibration Improves Reliability

Calibration helps make confidence scores more meaningful. For a mobile or greenhouse workflow, this is important because the system should know when to return a prediction and when to mark a case as uncertain.

## Why Macro-F1 Matters

Macro-F1 gives each class equal weight, which is important for disease classification when some categories may have fewer examples than others.

## Why Coverage Matters

Coverage reports how often the system accepts a prediction. A very strict threshold may improve accepted accuracy but reject too many images. The final system balanced reliability with practical usability.
