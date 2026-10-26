"""
feature_engineering.py - Module trích xuất và lựa chọn đặc trưng.

Bao gồm:
    - Tạo đặc trưng mới (Feature Creation)
    - Đặc trưng đa thức (Polynomial Features)
    - Đặc trưng tương tác (Interaction Features)
    - Lựa chọn đặc trưng (Feature Selection)
    - Phân tích thành phần chính (PCA)

Tác giả: Nhóm 14
Môn: Học Máy
"""

import logging
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, chi2,
    RFE, f_classif
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. TẠO ĐẶC TRƯNG MỚI
# =============================================================================

def create_credit_features_german(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo đặc trưng mới cho bộ dữ liệu German Credit.

    Các đặc trưng mới:
        - credit_per_month: Số tiền tín dụng trên mỗi tháng
        - age_group: Nhóm tuổi
        - credit_amount_bin: Phân nhóm số tiền tín dụng
        - duration_group: Phân nhóm thời hạn vay
        - amount_duration_ratio: Tỷ lệ số tiền / thời hạn

    Args:
        df: DataFrame German Credit.

    Returns:
        DataFrame với các đặc trưng mới.
    """
    df = df.copy()

    # Tỷ lệ tín dụng trên tháng
    if 'credit_amount' in df.columns and 'duration' in df.columns:
        df['credit_per_month'] = df['credit_amount'] / (df['duration'] + 1)

        # Tỷ lệ số tiền / thời hạn
        df['amount_duration_ratio'] = df['credit_amount'] / (df['duration'] + 1)

    # Nhóm tuổi
    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'].clip(lower=0),
            bins=[-np.inf, 25, 35, 45, 55, np.inf],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype(int)

    # Phân nhóm số tiền tín dụng
    if 'credit_amount' in df.columns:
        # Ngưỡng cố định để phép biến đổi không phụ thuộc toàn bộ dữ liệu.
        df['credit_amount_bin'] = pd.cut(
            df['credit_amount'],
            bins=[-np.inf, 1000, 2500, 5000, 10000, np.inf],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype(int)

    # Phân nhóm thời hạn vay
    if 'duration' in df.columns:
        df['duration_group'] = pd.cut(
            df['duration'],
            bins=[-np.inf, 12, 24, 36, 48, np.inf],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype(int)

    logger.info(f"Đã tạo đặc trưng mới cho German Credit. Shape: {df.shape}")
    return df


def create_credit_features_gmsc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo đặc trưng mới cho bộ dữ liệu Give Me Some Credit (GMSC).

    Các đặc trưng mới:
        - total_past_due: Tổng số lần quá hạn
        - has_past_due: Có từng quá hạn không
        - total_lines: Tổng số khoản vay
        - age_group: Nhóm tuổi
        - high_debt_flag: Cờ nợ cao
        - income_per_dependent: Thu nhập trên mỗi người phụ thuộc

    Args:
        df: DataFrame Give Me Some Credit.

    Returns:
        DataFrame với các đặc trưng mới.
    """
    df = df.copy()

    # Tổng số lần quá hạn thanh toán
    past_due_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]
    existing_cols = [c for c in past_due_cols if c in df.columns]
    if existing_cols:
        df['total_past_due'] = df[existing_cols].sum(axis=1)
        df['has_past_due'] = (df['total_past_due'] > 0).astype(int)

    # Nhóm tuổi
    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'].clip(lower=0),
            bins=[-np.inf, 30, 40, 50, 60, np.inf],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype(int)

    # Cờ nợ cao
    if 'DebtRatio' in df.columns:
        df['high_debt_flag'] = (df['DebtRatio'] > 1.0).astype(int)

    # Thu nhập trên mỗi người phụ thuộc
    if 'MonthlyIncome' in df.columns and 'NumberOfDependents' in df.columns:
        df['income_per_dependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)

    # Tổng số khoản vay/tín dụng
    loan_cols = ['NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']
    existing_loan_cols = [c for c in loan_cols if c in df.columns]
    if existing_loan_cols:
        df['total_lines'] = df[existing_loan_cols].sum(axis=1)

    # Tỷ lệ sử dụng tín dụng cao
    if 'RevolvingUtilizationOfUnsecuredLines' in df.columns:
        df['high_utilization'] = (
            df['RevolvingUtilizationOfUnsecuredLines'] > 1.0
        ).astype(int)

    logger.info(f"Đã tạo đặc trưng mới cho GMSC. Shape: {df.shape}")
    return df


# =============================================================================
# 2. ĐẶC TRƯNG ĐA THỨC
# =============================================================================

def create_polynomial_features(
    X: pd.DataFrame,
    degree: int = 2,
    interaction_only: bool = False,
    top_k: int = None
) -> Tuple[pd.DataFrame, PolynomialFeatures]:
    """
    Tạo đặc trưng đa thức.

    Args:
        X: DataFrame đặc trưng.
        degree: Bậc đa thức.
        interaction_only: Chỉ tạo đặc trưng tương tác (không có bậc cao).
        top_k: Chỉ sử dụng top_k đặc trưng quan trọng nhất.

    Returns:
        Tuple (DataFrame đặc trưng mới, PolynomialFeatures object).
    """
    if top_k and top_k < X.shape[1]:
        # Chọn top_k đặc trưng để tránh bùng nổ chiều
        cols = X.columns[:top_k].tolist()
        X_subset = X[cols]
    else:
        X_subset = X

    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )
    X_poly = poly.fit_transform(X_subset)
    feature_names = poly.get_feature_names_out(X_subset.columns)

    df_poly = pd.DataFrame(X_poly, columns=feature_names, index=X.index)

    logger.info(f"Đã tạo đặc trưng đa thức: {X_subset.shape[1]} -> {df_poly.shape[1]} đặc trưng")
    return df_poly, poly


# =============================================================================
# 3. LỰA CHỌN ĐẶC TRƯNG
# =============================================================================

def select_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 10
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lựa chọn đặc trưng bằng Mutual Information.

    Args:
        X: DataFrame đặc trưng.
        y: Series biến mục tiêu.
        k: Số đặc trưng cần chọn.

    Returns:
        Tuple (DataFrame đặc trưng đã chọn, danh sách tên đặc trưng).
    """
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    # Lấy tên đặc trưng được chọn
    mask = selector.get_support()
    selected_features = X.columns[mask].tolist()

    # Hiển thị điểm số
    scores = pd.DataFrame({
        'feature': X.columns,
        'mi_score': selector.scores_
    }).sort_values('mi_score', ascending=False)

    logger.info(f"Top {k} đặc trưng theo Mutual Information:")
    for _, row in scores.head(k).iterrows():
        logger.info(f"  {row['feature']}: {row['mi_score']:.4f}")

    return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features


def select_features_chi2(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 10
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lựa chọn đặc trưng bằng Chi-Square test.

    Lưu ý: Chỉ áp dụng cho đặc trưng không âm.

    Args:
        X: DataFrame đặc trưng (giá trị >= 0).
        y: Series biến mục tiêu.
        k: Số đặc trưng cần chọn.

    Returns:
        Tuple (DataFrame đặc trưng đã chọn, danh sách tên đặc trưng).
    """
    k = min(k, X.shape[1])

    # Đảm bảo giá trị không âm
    X_pos = X.copy()
    X_pos[X_pos < 0] = 0

    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X_pos, y)

    mask = selector.get_support()
    selected_features = X.columns[mask].tolist()

    logger.info(f"Đã chọn {k} đặc trưng bằng Chi-Square test: {selected_features}")
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features


def select_features_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 10,
    estimator=None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lựa chọn đặc trưng bằng Recursive Feature Elimination (RFE).

    Args:
        X: DataFrame đặc trưng.
        y: Series biến mục tiêu.
        n_features: Số đặc trưng cần chọn.
        estimator: Bộ phân loại dùng cho RFE. Mặc định RandomForest.

    Returns:
        Tuple (DataFrame đặc trưng đã chọn, danh sách tên đặc trưng).
    """
    n_features = min(n_features, X.shape[1])

    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_].tolist()

    # Xếp hạng đặc trưng
    ranking = pd.DataFrame({
        'feature': X.columns,
        'ranking': rfe.ranking_
    }).sort_values('ranking')

    logger.info(f"Top {n_features} đặc trưng theo RFE:")
    for _, row in ranking.head(n_features).iterrows():
        logger.info(f"  Hạng {int(row['ranking'])}: {row['feature']}")

    return X[selected_features], selected_features


def get_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'random_forest'
) -> pd.DataFrame:
    """
    Phân tích mức độ quan trọng của đặc trưng.

    Args:
        X: DataFrame đặc trưng.
        y: Series biến mục tiêu.
        method: Phương pháp ('random_forest', 'mutual_info').

    Returns:
        DataFrame chứa tên đặc trưng và điểm quan trọng, sắp xếp giảm dần.
    """
    if method == 'random_forest':
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
    elif method == 'mutual_info':
        importances = mutual_info_classif(X, y, random_state=42)
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}")

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return importance_df


# =============================================================================
# 4. GIẢM CHIỀU DỮ LIỆU (PCA)
# =============================================================================

def apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Giảm chiều dữ liệu bằng PCA.

    Args:
        X_train: Tập train.
        X_test: Tập test.
        n_components: Số thành phần chính hoặc tỷ lệ phương sai cần giữ.
            - Nếu < 1: giữ lại n_components% phương sai.
            - Nếu >= 1: giữ lại n_components thành phần.

    Returns:
        Tuple (X_train_pca, X_test_pca, pca_model).
    """
    pca = PCA(n_components=n_components, random_state=42)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Hiển thị thông tin PCA
    logger.info(f"PCA: {X_train.shape[1]} -> {pca.n_components_} thành phần")
    logger.info(f"Phương sai giải thích: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")

    # Chi tiết từng thành phần
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    for i, (var, cum_var) in enumerate(zip(pca.explained_variance_ratio_, cumulative_var)):
        if i < 10:  # Chỉ in 10 thành phần đầu
            logger.info(f"  PC{i+1}: {var:.4f} (tích lũy: {cum_var:.4f})")

    return X_train_pca, X_test_pca, pca


# =============================================================================
# 5. PIPELINE TỔNG HỢP
# =============================================================================

def feature_engineering_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    dataset_name: str = 'german_credit',
    feature_selection_method: str = 'mutual_info',
    n_select_features: int = 15,
    use_pca: bool = False,
    pca_variance: float = 0.95,
    create_new_features: bool = True
) -> dict:
    """
    Pipeline trích xuất đặc trưng tổng hợp.

    Args:
        X_train, X_test: Dữ liệu train/test.
        y_train: Biến mục tiêu (train).
        dataset_name: Tên bộ dữ liệu ('german_credit' hoặc 'give_me_credit').
        feature_selection_method: Phương pháp lựa chọn đặc trưng.
        n_select_features: Số đặc trưng cần chọn.
        use_pca: Có áp dụng PCA không.
        pca_variance: Tỷ lệ phương sai giữ lại khi dùng PCA.
        create_new_features: Có tạo đặc trưng mới không.

    Returns:
        Dict chứa dữ liệu và metadata.
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU PIPELINE TRÍCH XUẤT ĐẶC TRƯNG")
    logger.info("=" * 60)

    result = {}

    # Phân tích mức độ quan trọng ban đầu
    logger.info("\n[1] Phân tích Feature Importance...")
    importance_df = get_feature_importance(X_train, y_train)
    result['feature_importance'] = importance_df

    # Lựa chọn đặc trưng
    if feature_selection_method and n_select_features < X_train.shape[1]:
        logger.info(f"\n[2] Lựa chọn đặc trưng ({feature_selection_method})...")

        if feature_selection_method == 'mutual_info':
            X_train_sel, selected = select_features_mutual_info(
                X_train, y_train, k=n_select_features
            )
        elif feature_selection_method == 'chi2':
            X_train_sel, selected = select_features_chi2(
                X_train, y_train, k=n_select_features
            )
        elif feature_selection_method == 'rfe':
            X_train_sel, selected = select_features_rfe(
                X_train, y_train, n_features=n_select_features
            )
        else:
            X_train_sel = X_train
            selected = X_train.columns.tolist()

        X_test_sel = X_test[selected]
        result['selected_features'] = selected
    else:
        X_train_sel = X_train
        X_test_sel = X_test
        result['selected_features'] = X_train.columns.tolist()

    # PCA (tùy chọn)
    if use_pca:
        logger.info(f"\n[3] Giảm chiều bằng PCA (giữ {pca_variance*100}% phương sai)...")
        X_train_final, X_test_final, pca_model = apply_pca(
            X_train_sel, X_test_sel, n_components=pca_variance
        )
        result['pca_model'] = pca_model

        # Chuyển về DataFrame
        pca_cols = [f'PC{i+1}' for i in range(X_train_final.shape[1])]
        X_train_final = pd.DataFrame(X_train_final, columns=pca_cols)
        X_test_final = pd.DataFrame(X_test_final, columns=pca_cols)
    else:
        X_train_final = X_train_sel
        X_test_final = X_test_sel

    result['X_train'] = X_train_final
    result['X_test'] = X_test_final

    logger.info(f"\nShape cuối cùng - Train: {X_train_final.shape}, Test: {X_test_final.shape}")
    logger.info("=" * 60)
    logger.info("HOÀN TẤT PIPELINE TRÍCH XUẤT ĐẶC TRƯNG!")
    logger.info("=" * 60)

    return result
