from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.default import AppConfig, build_default_config
from core.utils import ensure_dir, save_json


FEATURE_COLUMNS = [
    "Order Method",
    "Order Type",
    "Food Quantity",
    "Dessert Quantity",
    "Drink Quantity",
    "Subtotal",
    "Delivery Fee",
    "Total Order",
    "Discount Amount",
    "Gender",
    "Customer Loyalty",
    "Delivery Rating",
]


def _load_training_frame(cfg: AppConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.paths.raw_dir / "food_delivery_data.xlsx")
    df = df[df["Order Status"].isin(["Cancelled", "Completed"])].copy()
    if len(df) < cfg.fit.cancel_train_min_samples:
        raise ValueError("Not enough labeled samples for cancellation model.")
    df["label"] = (df["Order Status"] == "Cancelled").astype(int)
    return df


def fit_cancel_model(cfg: AppConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or build_default_config()
    df = _load_training_frame(cfg)

    categorical = [col for col in FEATURE_COLUMNS if df[col].dtype == object]
    numeric = [col for col in FEATURE_COLUMNS if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(df[FEATURE_COLUMNS], df["label"])

    stats = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "train_samples": int(len(df)),
        "cancel_rate": float(df["label"].mean()),
        "sklearn_version": sklearn.__version__,
    }
    return {"pipeline": pipeline, "stats": stats}


def main() -> None:
    cfg = build_default_config()
    ensure_dir(cfg.paths.fitted_dir)
    result = fit_cancel_model(cfg)
    with (cfg.paths.fitted_dir / "cancel_model.pkl").open("wb") as handle:
        pickle.dump(result["pipeline"], handle)
    save_json(cfg.paths.fitted_dir / "cancel_model_meta.json", result["stats"])
    print("saved", cfg.paths.fitted_dir / "cancel_model.pkl")


if __name__ == "__main__":
    main()
