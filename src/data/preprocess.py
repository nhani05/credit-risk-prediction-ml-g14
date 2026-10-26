"""
preprocess.py - Tiền xử lý dữ liệu cho bài toán đánh giá rủi ro tín dụng.

Các thống kê học được (điền thiếu, ngưỡng ngoại lai, encoding, scaling) được
fit chỉ trên tập train để hạn chế data leakage. SMOTE chỉ được áp dụng cho train.
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def handle_missing_values(
    df: pd.DataFrame,
    strategy_numeric: str = "median",
    strategy_categorical: str = "most_frequent",
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """Điền giá trị thiếu đơn lẻ trong DataFrame (tiện ích EDA/khám phá)."""
    result = df.copy()
    dropped = result.columns[result.isna().mean() > drop_threshold]
    result = result.drop(columns=dropped)
    numeric = result.select_dtypes(include=[np.number]).columns
    categorical = result.select_dtypes(include=["object", "category"]).columns
    if len(numeric):
        result[numeric] = SimpleImputer(strategy=strategy_numeric).fit_transform(result[numeric])
    if len(categorical):
        result[categorical] = SimpleImputer(strategy=strategy_categorical).fit_transform(result[categorical])
    return result


def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    factor: float = 1.5,
    exclude_cols: list = None,
) -> pd.DataFrame:
    """Capping ngoại lai tiện ích; với pipeline, ngưỡng phải lấy từ train."""
    result = df.copy()
    exclude_cols = set(exclude_cols or [])
    for col in result.select_dtypes(include=[np.number]).columns:
        if col in exclude_cols:
            continue
        if method == "iqr":
            q1, q3 = result[col].quantile([0.25, 0.75])
            spread = q3 - q1
            lower, upper = q1 - factor * spread, q3 + factor * spread
        elif method == "zscore":
            mean, std = result[col].mean(), result[col].std()
            lower, upper = mean - factor * std, mean + factor * std
        elif method in (None, "none"):
            continue
        else:
            raise ValueError(f"Phương pháp ngoại lai không hợp lệ: {method}")
        result[col] = result[col].clip(lower=lower, upper=upper)
    return result


def encode_categorical(
    df: pd.DataFrame, method: str = "label", target_col: str = None
) -> Tuple[pd.DataFrame, dict]:
    """Mã hóa biến phân loại cho khám phá; pipeline chính fit encoder trên train."""
    result = df.copy()
    cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in cols:
        cols.remove(target_col)
    encoders = {}
    if method in ("label", "auto"):
        for col in cols:
            encoder = LabelEncoder()
            result[col] = encoder.fit_transform(result[col].astype(str))
            encoders[col] = encoder
    elif method == "onehot":
        result = pd.get_dummies(result, columns=cols, drop_first=True, dtype=int)
    else:
        raise ValueError(f"Phương pháp encoding không hợp lệ: {method}")
    return result, encoders


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "standard"
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit scaler trên train rồi transform train/test."""
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method in (None, "none"):
        return X_train.copy(), X_test.copy(), None
    else:
        raise ValueError(f"Phương pháp scaling không hợp lệ: {method}")
    scaler.fit(X_train)
    return (
        pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index),
        pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index),
        scaler,
    )


def handle_imbalance(
    X: pd.DataFrame, y: pd.Series, method: str = "smote", random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Xử lý mất cân bằng; gọi hàm này trên tập train, không gọi trên test."""
    if method in (None, "none"):
        return X, y
    if y.nunique() < 2:
        raise ValueError("SMOTE/undersampling yêu cầu ít nhất hai lớp trong y.")
    if method == "smote":
        minority_count = y.value_counts().min()
        if minority_count < 2:
            raise ValueError("SMOTE cần ít nhất 2 mẫu ở lớp thiểu số.")
        sampler = SMOTE(random_state=random_state, k_neighbors=min(5, minority_count - 1))
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == "combined":
        minority_count = y.value_counts().min()
        if minority_count < 2:
            raise ValueError("SMOTE cần ít nhất 2 mẫu ở lớp thiểu số.")
        sampler = ImbPipeline([
            ("over", SMOTE(sampling_strategy=0.5, random_state=random_state,
                            k_neighbors=min(5, minority_count - 1))),
            ("under", RandomUnderSampler(sampling_strategy=0.8, random_state=random_state)),
        ])
    else:
        raise ValueError(f"Phương pháp xử lý mất cân bằng không hợp lệ: {method}")
    X_res, y_res = sampler.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def preprocess_pipeline(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    missing_strategy: str = "median",
    outlier_method: str = "iqr",
    encoding_method: str = "label",
    scaling_method: str = "standard",
    imbalance_method: str = "smote",
    save_processed: bool = True,
    drop_missing_threshold: float = 0.5,
    outlier_factor: float = 1.5,
) -> dict:
    """Chia dữ liệu trước, sau đó fit tất cả bước tiền xử lý trên train."""
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy target column: {target_col}")
    if df[target_col].isna().any():
        raise ValueError(f"Cột mục tiêu '{target_col}' có giá trị thiếu.")
    if not 0 < test_size < 1:
        raise ValueError("test_size phải nằm giữa 0 và 1.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Loại cột nhiều missing dựa trên train; giữ cùng một tập cột cho test.
    dropped = X_train.columns[X_train.isna().mean() > drop_missing_threshold]
    X_train, X_test = X_train.drop(columns=dropped), X_test.drop(columns=dropped)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_imputer = SimpleImputer(strategy=missing_strategy)
    if num_cols:
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    if cat_cols:
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # Ngưỡng capping được tính trên train, áp dụng nguyên trạng lên test.
    bounds = {}
    for col in num_cols:
        if outlier_method == "iqr":
            q1, q3 = X_train[col].quantile([0.25, 0.75])
            spread = q3 - q1
            bounds[col] = (q1 - outlier_factor * spread, q3 + outlier_factor * spread)
        elif outlier_method in (None, "none"):
            continue
        else:
            raise ValueError("Pipeline hỗ trợ outlier_method='iqr' hoặc 'none'.")
        lower, upper = bounds[col]
        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    encoders = {}
    if cat_cols:
        if encoding_method in ("label", "auto"):
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].astype(str))
            X_test[cat_cols] = encoder.transform(X_test[cat_cols].astype(str))
            encoders["categorical"] = encoder
        elif encoding_method == "onehot":
            X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        else:
            raise ValueError(f"Phương pháp encoding không hợp lệ: {encoding_method}")

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    X_train, X_test, scaler = scale_features(X_train, X_test, scaling_method)

    # Giữ bản train chưa resample để cross-validation dùng phân bố quan sát được.
    X_train_original, y_train_original = X_train.copy(), y_train.copy()

    # Sampling chỉ áp dụng trên train. Test giữ nguyên phân bố thực tế.
    if imbalance_method not in (None, "none"):
        X_train, y_train = handle_imbalance(X_train, y_train, imbalance_method, random_state)

    if save_processed:
        output_dir = os.path.join(PROCESSED_DIR, str(target_col))
        os.makedirs(output_dir, exist_ok=True)
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        y_train.to_frame(name=target_col).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        y_test.to_frame(name=target_col).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_original": X_train_original,
        "y_train_original": y_train_original,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": X_train.columns.tolist(),
        "imputers": {"numeric": num_imputer, "categorical": cat_imputer},
        "outlier_bounds": bounds,
        "dropped_columns": dropped.tolist(),
    }
