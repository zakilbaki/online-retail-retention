# Online Retail Retention

Data Science portfolio project based on the UCI Online Retail II dataset.

## Objective

Build a temporally valid churn-risk model for a non-contractual retailer. A customer reaches churn after 60 consecutive days without a valid positive purchase. At each monthly reference date, the model predicts whether an active customer will reach that boundary during the next 30 days.

## Methodology

- Audit and clean two overlapping transaction sheets.
- Preserve cancellations and reconstruct fully reversed invoices conservatively.
- Create monthly customer snapshots using only information available before each reference date.
- Use chronological walk-forward validation with target-aware purging.
- Reserve September through November 2011 as an untouched out-of-time test period.
- Add candidate features incrementally and compare them on both temporal validation folds.

## Current model

The current reference is a standardized logistic regression using:

- `RecencyDays`
- `PurchaseFrequency`
- `HasCancellation`
- `IsInChurnRiskWindow`

The explicit risk-window feature represents the operational boundary `RecencyDays >= 30`. It is preferred over the equivalent recent-purchase count because it is simpler and directly interpretable.

Development validation results:

| Validation period | PR AUC | Recall | Precision |
| --- | ---: | ---: | ---: |
| October-November 2010 | 90.33% | 96.33% | 80.39% |
| January-February 2011 | 97.10% | 94.91% | 92.58% |

The final test set remains locked. These figures are model-selection results, not final test performance.

## Notebook sequence

1. `01_data_audit.ipynb`
2. `02_customer_activity_and_churn_definition.ipynb`
3. `03_snapshot_construction.ipynb`
4. `04_exploratory_data_analysis.ipynb`
5. `05_feature_engineering.ipynb`
6. `06_baseline_model.ipynb`
7. `07_model_comparison.ipynb`

The remaining notebooks are reserved for calibration, ranking, and customer-value analysis.

## Roadmap

### 1. Complete model selection

- Compare logistic regression, decision trees, Random Forest, and stochastic gradient descent on the same purged temporal folds.
- Increase model complexity incrementally and retain a model only when its PR AUC improves consistently across periods.
- Analyze false positives and false negatives by customer feature to identify where each model fails.
- Keep the out-of-time test period locked until the complete modeling pipeline is frozen.

### 2. Select the operating threshold from business costs

- Use precision when unnecessary interventions are expensive or may irritate customers.
- Use recall when missing a valuable future churner represents the larger loss.
- Consider F1 when false-positive and false-negative consequences are comparable.
- Evaluate candidate thresholds with the explicit objective:

```text
TotalCost = CostFalsePositive × FalsePositives
          + CostFalseNegative × FalseNegatives
```

- Extend the calculation with campaign cost, customer value, intervention success probability, and retained margin when those inputs become available.

### 3. Convert predictions into retention decisions

- Combine churn probability with customer value to create risk-value segments.
- Assign tiered actions such as low-cost reminders, targeted offers, or proactive service according to expected value.
- Concentrate limited retention resources on customers with both high churn risk and high expected value.
- Use product and purchasing-pattern associations to generate product-development hypotheses, while avoiding causal claims from observational transactions alone.

### 4. Validate and monitor the system

- Evaluate the frozen pipeline once on the locked out-of-time test period.
- Record predictions, interventions, costs, and customer responses so retention uplift can eventually be measured.
- Monitor feature drift, ranking quality, calibration, contact volume, and realized business value over time.
- Use the collected feedback to update thresholds, retrain models, and adapt retention strategies when customer behavior changes.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `online_retail_II.xlsx` in `data/raw/` before running the notebooks. Raw, interim, and processed datasets are intentionally excluded from Git.
