"""Reference churn-model definition."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "RecencyDays",
    "PurchaseFrequency",
    "HasCancellation",
    "IsInChurnRiskWindow",
]
TARGET_COLUMN = "WillChurnNext30Days"


def build_reference_model():
    """Return the model selected during temporal development validation."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="liblinear"),
    )

