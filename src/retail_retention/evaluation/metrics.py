"""Evaluation helpers shared by model-selection and business analysis."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_probability, threshold=0.5):
    """Calculate ranking and threshold-dependent binary metrics."""
    y_true = np.asarray(y_true, dtype=bool)
    y_probability = np.asarray(y_probability, dtype=float)
    y_pred = y_probability >= threshold
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[False, True]
    ).ravel()
    return {
        "Threshold": threshold,
        "ChurnRate": y_true.mean(),
        "ContactRate": y_pred.mean(),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "PR_AUC": average_precision_score(y_true, y_probability),
        "ROC_AUC": roc_auc_score(y_true, y_probability),
        "BrierScore": brier_score_loss(y_true, y_probability),
        "LogLoss": log_loss(y_true, y_probability),
        "TrueNegatives": tn,
        "FalsePositives": fp,
        "FalseNegatives": fn,
        "TruePositives": tp,
    }


def total_error_cost(false_positives, false_negatives, fp_cost=1, fn_cost=1):
    """Return the simplified cost of classification errors."""
    return fp_cost * false_positives + fn_cost * false_negatives


def expected_calibration_error(y_true, y_probability, n_bins=10):
    """Return weighted absolute calibration error across equal-width bins."""
    y_true = np.asarray(y_true, dtype=bool)
    y_probability = np.asarray(y_probability, dtype=float)
    bin_ids = np.minimum((y_probability * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += mask.mean() * abs(
                y_true[mask].mean() - y_probability[mask].mean()
            )
    return error

