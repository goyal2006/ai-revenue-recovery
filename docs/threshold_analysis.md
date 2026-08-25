# Threshold Analysis

## Objective

Determine how the recovery decision threshold affects revenue recovery and net business value.

## Method

Recovery candidates were evaluated using 5-fold stratified out-of-fold predictions from the baseline Logistic Regression model.

The threshold determines which transactions are selected for recovery.

## Key Results

| Recovery Action Cost | Best Threshold | Customers Targeted | Net Revenue |
|---:|---:|---:|---:|
| ₹5  | 0.30 | 991 | ₹22,133.18 |
| ₹10 | 0.30 | 991 | ₹17,178.18 |
| ₹20 | 0.40 | 948 | ₹7,438.09 |
| ₹30 | 0.65 | 327 | ₹738.79 |

## Interpretation

Lower recovery costs favor broader recovery targeting because the system can afford to attempt recovery on more customers.

As recovery cost increases, the optimal threshold increases because the system must become more selective and prioritize customers with higher predicted recovery likelihood.

At a recovery cost of ₹30, broad recovery targeting becomes unprofitable. A threshold of 0.65 produces the highest net revenue among the tested thresholds.

## Important Limitation

The recovery cost values are hypothetical because the dataset is synthetic.

The threshold should therefore be treated as an experimental business-policy parameter rather than a production decision.

In a real deployment, the threshold should be optimized using actual recovery-action costs, recovered revenue, customer impact, and operational constraints.

## Model Limitation

The baseline Logistic Regression model has relatively weak discrimination:

ROC-AUC = 0.5831 ± 0.0383

Therefore, the model currently provides modest prioritization ability rather than highly accurate recovery prediction.

The threshold analysis demonstrates the business decision framework, but improving model discrimination remains an important next step.