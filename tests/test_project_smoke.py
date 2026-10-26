"""Kiểm tra nhanh cấu hình, dữ liệu và notebook của dự án."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.data.preprocess import handle_missing_values, handle_outliers, preprocess_pipeline
from src.features.feature_engineering import (
    create_credit_features_german,
    create_credit_features_gmsc,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    config = load_config()

    assert config.get("split.random_state") == 42
    assert config.get("datasets.german_credit.target_col") == "target"
    assert config.get("datasets.give_me_credit.target_col") == "SeriousDlqin2yrs"


def test_missing_values_are_imputed():
    frame = pd.DataFrame({"income": [10.0, None, 30.0], "job": ["A", None, "B"]})

    processed = handle_missing_values(frame)

    assert not processed.isna().any().any()
    assert processed.loc[1, "income"] == 20.0
    assert processed.loc[1, "job"] in {"A", "B"}


def test_outlier_capping_preserves_target():
    frame = pd.DataFrame({"amount": [1, 2, 3, 4, 1000], "target": [0, 0, 1, 0, 1]})

    processed = handle_outliers(frame, exclude_cols=["target"])

    assert processed["amount"].max() < frame["amount"].max()
    assert processed["target"].tolist() == frame["target"].tolist()


def test_feature_engineering_handles_boundaries_and_ties():
    german = pd.DataFrame({
        "age": [18, 25, 40, 56, 90],
        "duration": [1, 12, 24, 48, 120],
        "credit_amount": [100, 100, 2500, 10000, 50000],
        "target": [0, 1, 0, 1, 0],
    })
    transformed_german = create_credit_features_german(german)
    assert transformed_german["age_group"].notna().all()
    assert transformed_german["credit_amount_bin"].notna().all()
    assert transformed_german["duration_group"].notna().all()

    gmsc = pd.DataFrame({"age": [18, 30, 50, 61, 130]})
    transformed_gmsc = create_credit_features_gmsc(gmsc)
    assert transformed_gmsc["age_group"].notna().all()


def test_preprocessing_keeps_test_distribution_and_no_missing_values():
    frame = pd.DataFrame({
        "amount": [10.0, 12.0, None, 15.0, 17.0, 19.0, 22.0, 25.0, 30.0, 10000.0,
                   11.0, 13.0, 16.0, 18.0, 21.0, 24.0, 28.0, 35.0, 40.0, 45.0],
        "purpose": ["A", "B", "A", "C", "A", "B", "C", "A", "B", "C",
                    "C", "A", "B", "C", "A", "B", "A", "C", "B", "A"],
        "target": [0] * 15 + [1] * 5,
    })

    result = preprocess_pipeline(
        frame,
        target_col="target",
        test_size=0.3,
        random_state=3,
        imbalance_method="smote",
        save_processed=False,
    )

    assert len(result["X_test"]) == 6
    assert len(result["y_test"]) == 6
    assert result["X_train"].isna().sum().sum() == 0
    assert result["X_test"].isna().sum().sum() == 0
    assert len(result["X_train"]) >= len(result["X_train_original"])
    assert len(result["y_test"].unique()) == 2


def test_preprocessing_rejects_missing_target():
    frame = pd.DataFrame({"amount": [1, 2, 3], "target": [0, None, 1]})

    with pytest.raises(ValueError, match="giá trị thiếu"):
        preprocess_pipeline(frame, target_col="target", save_processed=False)


def test_notebooks_are_valid_nbformat_json():
    notebooks = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))

    assert len(notebooks) >= 3
    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] == 5
        assert notebook["cells"]
