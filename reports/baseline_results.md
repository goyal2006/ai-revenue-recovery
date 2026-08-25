# Baseline Model Results

## Model

Logistic Regression

## Dataset

Recovery-attempt transactions only.

- Total recovery attempts: 1000
- Training samples: 800
- Test samples: 200
- Random state: 42

## Results

| Metric | Score |
|---|---:|
| Accuracy | 0.5950 |
| ROC-AUC | 0.5580 |

## Confusion Matrix

| | Predicted False | Predicted True |
|---|---:|---:|
| Actual False | 18 | 63 |
| Actual True | 18 | 101 |

## Interpretation

The baseline Logistic Regression model achieves 59.5% accuracy
and a ROC-AUC of 0.5580.

The model has strong recall for successful recoveries (0.85)
but weak recall for unsuccessful recoveries (0.22).

The ROC-AUC indicates that the model currently has only weak
discriminative power. This baseline will be used as the reference
point for subsequent models and feature-engineering experiments.