"""Build the fixed, leakage-safe conventional preprocessing pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "step",
    "amount",
    "hour",
    "day",
    "log_amount",
    "is_transfer",
    "is_cash_out",
    "is_merchant_destination",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    """Create the predetermined preprocessing transformer.

    The transformer is always fitted *inside* each cross-validation fold.  This
    prevents category/scaling statistics from the validation fold leaking into
    training. Logistic Regression receives standardised numeric features, while
    tree models retain their original numeric scale because their split rules are
    scale invariant.
    """

    numeric_transformer: Pipeline | str
    if scale_numeric:
        numeric_transformer = Pipeline([("scaler", StandardScaler())])
    else:
        numeric_transformer = "passthrough"

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                # Unknown categories may appear in a validation fold or future
                # data; ignoring them is safer than failing the whole experiment.
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def validate_model_features(columns: list[str]) -> None:
    """Fail before CV when the prepared dataset lacks required model features."""

    missing = sorted(set(MODEL_FEATURES).difference(columns))
    if missing:
        raise ValueError(f"Modeling dataset is missing features: {', '.join(missing)}")
