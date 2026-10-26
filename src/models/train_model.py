"""
train_model.py - Module huấn luyện mô hình Machine Learning.

Các mô hình:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting (GBM)
    - XGBoost
    - LightGBM
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Stacking Ensemble

Kỹ thuật:
    - Cross-Validation
    - Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
    - Lưu/tải mô hình với joblib

Tác giả: Nhóm 14
Môn: Học Máy
"""

import os
import time
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, make_scorer
)

# Import XGBoost và LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


# =============================================================================
# ĐỊNH NGHĨA MÔ HÌNH VÀ HYPERPARAMETERS
# =============================================================================

def get_default_models() -> Dict[str, Any]:
    """
    Trả về dictionary chứa các mô hình với tham số mặc định.

    Returns:
        Dict {tên_mô_hình: instance_mô_hình}.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=10
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=42, learning_rate=0.1
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1
        ),
    }

    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, random_state=42, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        )

    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, verbose=-1
        )

    return models


def get_param_grids() -> Dict[str, dict]:
    """
    Trả về dictionary chứa lưới tham số cho hyperparameter tuning.

    Returns:
        Dict {tên_mô_hình: param_grid}.
    """
    param_grids = {
        'Logistic Regression': [
            {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1'],
                'solver': ['liblinear', 'saga'],
            },
            {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['liblinear', 'saga'],
            },
        ],
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'],
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.01, 0.1],
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        },
    }

    if HAS_XGBOOST:
        param_grids['XGBoost'] = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        }

    if HAS_LIGHTGBM:
        param_grids['LightGBM'] = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, -1],
            'num_leaves': [31, 50, 70],
            'subsample': [0.8, 0.9, 1.0],
        }

    return param_grids


# =============================================================================
# HUẤN LUYỆN MÔ HÌNH
# =============================================================================

def train_single_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model"
) -> dict:
    """
    Huấn luyện một mô hình đơn lẻ.

    Args:
        model: Instance mô hình sklearn.
        X_train: Dữ liệu train.
        y_train: Nhãn train.
        model_name: Tên mô hình.

    Returns:
        Dict chứa mô hình đã train và thời gian train.
    """
    logger.info(f"Đang huấn luyện {model_name}...")
    start_time = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    train_acc = accuracy_score(y_train, model.predict(X_train))

    logger.info(f"  ✅ {model_name} - Train Acc: {train_acc:.4f} - Thời gian: {train_time:.2f}s")

    return {
        'model': model,
        'train_time': train_time,
        'train_accuracy': train_acc,
    }


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Dict[str, Any] = None
) -> Dict[str, dict]:
    """
    Huấn luyện tất cả mô hình.

    Args:
        X_train: Dữ liệu train.
        y_train: Nhãn train.
        models: Dict mô hình. Nếu None, sử dụng mặc định.

    Returns:
        Dict {tên_mô_hình: kết_quả_train}.
    """
    if models is None:
        models = get_default_models()

    results = {}

    logger.info("=" * 60)
    logger.info(f"HUẤN LUYỆN {len(models)} MÔ HÌNH")
    logger.info("=" * 60)

    for name, model in models.items():
        try:
            results[name] = train_single_model(model, X_train, y_train, name)
        except Exception as e:
            logger.error(f"  ❌ Lỗi khi huấn luyện {name}: {e}")
            results[name] = {'model': None, 'error': str(e)}

    logger.info("=" * 60)
    logger.info("HOÀN TẤT HUẤN LUYỆN!")
    logger.info("=" * 60)

    return results


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Any] = None,
    cv: int = 5,
    scoring: str = 'f1'
) -> pd.DataFrame:
    """
    Đánh giá mô hình bằng Cross-Validation.

    Args:
        X: Dữ liệu đặc trưng.
        y: Biến mục tiêu.
        models: Dict mô hình.
        cv: Số fold.
        scoring: Metric đánh giá ('accuracy', 'f1', 'roc_auc', 'precision', 'recall').

    Returns:
        DataFrame so sánh kết quả CV của các mô hình.
    """
    if models is None:
        models = get_default_models()

    results = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    logger.info(f"\nCross-Validation ({cv}-fold, scoring='{scoring}'):")
    logger.info("-" * 50)

    for name, model in models.items():
        try:
            start_time = time.time()
            scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
            cv_time = time.time() - start_time

            results.append({
                'Model': name,
                'CV Mean': scores.mean(),
                'CV Std': scores.std(),
                'CV Min': scores.min(),
                'CV Max': scores.max(),
                'Time (s)': cv_time,
            })

            logger.info(
                f"  {name:25s} | {scoring}: {scores.mean():.4f} ± {scores.std():.4f} | "
                f"[{scores.min():.4f}, {scores.max():.4f}] | {cv_time:.1f}s"
            )
        except Exception as e:
            logger.error(f"  {name}: Lỗi - {e}")

    cv_df = pd.DataFrame(results).sort_values('CV Mean', ascending=False).reset_index(drop=True)
    return cv_df


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(
    model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model",
    method: str = 'grid',
    cv: int = 5,
    scoring: str = 'f1',
    n_iter: int = 50
) -> dict:
    """
    Tinh chỉnh siêu tham số cho mô hình.

    Args:
        model: Instance mô hình.
        param_grid: Lưới tham số.
        X_train: Dữ liệu train.
        y_train: Nhãn train.
        model_name: Tên mô hình.
        method: Phương pháp ('grid' hoặc 'random').
        cv: Số fold CV.
        scoring: Metric đánh giá.
        n_iter: Số lần thử (cho RandomizedSearch).

    Returns:
        Dict chứa mô hình tốt nhất và kết quả.
    """
    logger.info(f"\nTinh chỉnh siêu tham số cho {model_name} ({method} search)...")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    if method == 'grid':
        search = GridSearchCV(
            model, param_grid, cv=skf, scoring=scoring,
            n_jobs=-1, verbose=0, refit=True
        )
    elif method == 'random':
        search = RandomizedSearchCV(
            model, param_grid, cv=skf, scoring=scoring,
            n_iter=n_iter, n_jobs=-1, verbose=0, refit=True, random_state=42
        )
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}")

    start_time = time.time()
    search.fit(X_train, y_train)
    tune_time = time.time() - start_time

    logger.info(f"  Tham số tốt nhất: {search.best_params_}")
    logger.info(f"  Điểm tốt nhất ({scoring}): {search.best_score_:.4f}")
    logger.info(f"  Thời gian: {tune_time:.1f}s")

    return {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': pd.DataFrame(search.cv_results_),
        'tune_time': tune_time,
    }


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Dict[str, Any] = None,
    param_grids: Dict[str, dict] = None,
    method: str = 'random',
    cv: int = 5,
    scoring: str = 'f1',
    n_iter: int = 30
) -> Dict[str, dict]:
    """
    Tinh chỉnh siêu tham số cho tất cả mô hình.

    Returns:
        Dict {tên_mô_hình: kết_quả_tuning}.
    """
    if models is None:
        models = get_default_models()
    if param_grids is None:
        param_grids = get_param_grids()

    tuned_results = {}

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    for name, model in models.items():
        if name in param_grids:
            try:
                tuned_results[name] = tune_hyperparameters(
                    model, param_grids[name], X_train, y_train,
                    model_name=name, method=method, cv=cv,
                    scoring=scoring, n_iter=n_iter
                )
            except Exception as e:
                logger.error(f"  ❌ Lỗi tuning {name}: {e}")
        else:
            logger.warning(f"  ⚠️ Không có param_grid cho {name}. Bỏ qua.")

    return tuned_results


# =============================================================================
# STACKING ENSEMBLE
# =============================================================================

def create_stacking_model(
    base_models: Dict[str, Any] = None,
    final_estimator=None
) -> StackingClassifier:
    """
    Tạo mô hình Stacking Ensemble.

    Args:
        base_models: Dict các mô hình cơ sở. Mặc định sử dụng RF, GBM, LR.
        final_estimator: Meta-learner. Mặc định Logistic Regression.

    Returns:
        StackingClassifier instance.
    """
    if base_models is None:
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ]

        if HAS_XGBOOST:
            estimators.append(
                ('xgb', XGBClassifier(n_estimators=100, random_state=42,
                                       use_label_encoder=False, eval_metric='logloss'))
            )
        if HAS_LIGHTGBM:
            estimators.append(
                ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
            )
    else:
        estimators = [(name, model) for name, model in base_models.items()]

    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    logger.info(f"Đã tạo Stacking Ensemble với {len(estimators)} mô hình cơ sở")
    return stacking


# =============================================================================
# LƯU / TẢI MÔ HÌNH
# =============================================================================

def save_model(model, model_name: str, directory: str = None) -> str:
    """
    Lưu mô hình ra file.

    Args:
        model: Mô hình đã train.
        model_name: Tên mô hình (dùng làm tên file).
        directory: Thư mục lưu. Mặc định là models/.

    Returns:
        Đường dẫn file đã lưu.
    """
    if directory is None:
        directory = MODELS_DIR
    os.makedirs(directory, exist_ok=True)

    filename = model_name.lower().replace(' ', '_') + '.joblib'
    filepath = os.path.join(directory, filename)

    joblib.dump(model, filepath)
    logger.info(f"Đã lưu mô hình: {filepath}")

    return filepath


def load_model(model_name: str, directory: str = None):
    """
    Tải mô hình từ file.

    Args:
        model_name: Tên mô hình.
        directory: Thư mục chứa file.

    Returns:
        Mô hình đã tải.
    """
    if directory is None:
        directory = MODELS_DIR

    filename = model_name.lower().replace(' ', '_') + '.joblib'
    filepath = os.path.join(directory, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy mô hình: {filepath}")

    model = joblib.load(filepath)
    logger.info(f"Đã tải mô hình: {filepath}")

    return model


def save_all_models(
    trained_results: Dict[str, dict],
    directory: str = None
) -> Dict[str, str]:
    """
    Lưu tất cả mô hình đã train.

    Args:
        trained_results: Dict kết quả train từ train_all_models().

    Returns:
        Dict {tên_mô_hình: đường_dẫn_file}.
    """
    paths = {}
    for name, result in trained_results.items():
        if result.get('model') is not None:
            paths[name] = save_model(result['model'], name, directory)
    return paths
