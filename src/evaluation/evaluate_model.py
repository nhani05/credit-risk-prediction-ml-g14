"""
evaluate_model.py - Module đánh giá hiệu năng mô hình.

Các phương pháp đánh giá:
    - Accuracy, Precision, Recall, F1-Score
    - AUC-ROC, AUC-PR
    - Confusion Matrix
    - Classification Report
    - So sánh nhiều mô hình
    - Phân tích lỗi (Error Analysis)
    - Phân tích bias-variance (Learning Curves)
    - Kiểm định thống kê

Tác giả: Nhóm 14
Môn: Học Máy
"""

import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
    log_loss, matthews_corrcoef, cohen_kappa_score,
    brier_score_loss
)
from sklearn.model_selection import (
    cross_val_score, learning_curve, StratifiedKFold
)
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. ĐÁNH GIÁ MỘT MÔ HÌNH
# =============================================================================

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> dict:
    """
    Đánh giá toàn diện một mô hình.

    Args:
        model: Mô hình đã huấn luyện.
        X_test: Dữ liệu test.
        y_test: Nhãn test.
        model_name: Tên mô hình.

    Returns:
        Dict chứa tất cả metric đánh giá.
    """
    # Dự đoán
    y_pred = model.predict(X_test)

    # Xác suất dự đoán (nếu có)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)

    # Tính các metric
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        metrics['avg_precision'] = average_precision_score(y_test, y_proba)
        metrics['log_loss'] = log_loss(y_test, y_proba)
        metrics['brier_score'] = brier_score_loss(y_test, y_proba)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    metrics['true_negatives'] = cm[0, 0]
    metrics['false_positives'] = cm[0, 1]
    metrics['false_negatives'] = cm[1, 0]
    metrics['true_positives'] = cm[1, 1]

    # Tỷ lệ lỗi cụ thể
    metrics['fpr'] = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    metrics['fnr'] = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

    # Lưu dự đoán
    metrics['y_pred'] = y_pred
    metrics['y_proba'] = y_proba

    # In kết quả
    _print_evaluation(metrics)

    return metrics


def _print_evaluation(metrics: dict) -> None:
    """In kết quả đánh giá ra console."""
    print(f"\n{'='*50}")
    print(f"  📊 KẾT QUẢ ĐÁNH GIÁ: {metrics['model_name']}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  MCC:         {metrics['mcc']:.4f}")

    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  Avg Prec:    {metrics['avg_precision']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")

    print(f"\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print(f"\n  FPR (Tỷ lệ dương tính giả): {metrics['fpr']:.4f}")
    print(f"  FNR (Tỷ lệ âm tính giả):    {metrics['fnr']:.4f}")
    print(f"{'='*50}")


# =============================================================================
# 2. SO SÁNH NHIỀU MÔ HÌNH
# =============================================================================

def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    So sánh hiệu năng nhiều mô hình.

    Args:
        models: Dict {tên: mô_hình} hoặc {tên: kết_quả_train}.
        X_test: Dữ liệu test.
        y_test: Nhãn test.

    Returns:
        DataFrame so sánh các metric.
    """
    all_metrics = {}

    for name, model_or_result in models.items():
        # Hỗ trợ cả model trực tiếp và result dict
        if isinstance(model_or_result, dict):
            model = model_or_result.get('model') or model_or_result.get('best_model')
        else:
            model = model_or_result

        if model is None:
            continue

        try:
            metrics = evaluate_model(model, X_test, y_test, model_name=name)
            all_metrics[name] = metrics
        except Exception as e:
            logger.error(f"Lỗi đánh giá {name}: {e}")

    # Tạo bảng so sánh
    comparison_cols = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'mcc', 'brier_score'
    ]

    comparison_data = []
    for name, metrics in all_metrics.items():
        row = {'Model': name}
        for col in comparison_cols:
            row[col] = metrics.get(col, np.nan)
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('f1_score', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("  📋 BẢNG SO SÁNH MÔ HÌNH")
    print("=" * 80)
    print(comparison_df.to_string(index=False, float_format='{:.4f}'.format))
    print("=" * 80)

    return comparison_df


# =============================================================================
# 3. PHÂN TÍCH LỖI
# =============================================================================

def error_analysis(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list = None,
    model_name: str = "Model"
) -> dict:
    """
    Phân tích lỗi chi tiết của mô hình.

    Phân tích:
        - False Positives: dự đoán rủi ro cao nhưng thực tế không
        - False Negatives: dự đoán rủi ro thấp nhưng thực tế rủi ro cao
        - So sánh đặc trưng giữa các nhóm lỗi

    Args:
        model: Mô hình đã train.
        X_test: Dữ liệu test.
        y_test: Nhãn test.
        feature_names: Tên đặc trưng.
        model_name: Tên mô hình.

    Returns:
        Dict chứa phân tích lỗi.
    """
    y_pred = model.predict(X_test)

    X_df = pd.DataFrame(X_test, columns=feature_names) if feature_names else pd.DataFrame(X_test)

    # Phân loại mẫu
    correct_mask = y_pred == y_test.values
    fp_mask = (y_pred == 1) & (y_test.values == 0)  # False Positive
    fn_mask = (y_pred == 0) & (y_test.values == 1)  # False Negative
    tp_mask = (y_pred == 1) & (y_test.values == 1)  # True Positive
    tn_mask = (y_pred == 0) & (y_test.values == 0)  # True Negative

    analysis = {
        'total_samples': len(y_test),
        'correct': correct_mask.sum(),
        'incorrect': (~correct_mask).sum(),
        'accuracy': correct_mask.mean(),
        'false_positives': fp_mask.sum(),
        'false_negatives': fn_mask.sum(),
    }

    print(f"\n{'='*60}")
    print(f"  🔍 PHÂN TÍCH LỖI: {model_name}")
    print(f"{'='*60}")
    print(f"  Tổng mẫu: {analysis['total_samples']}")
    print(f"  Đúng: {analysis['correct']} ({analysis['accuracy']*100:.1f}%)")
    print(f"  Sai:  {analysis['incorrect']} ({(1-analysis['accuracy'])*100:.1f}%)")
    print(f"    - False Positives (FP): {analysis['false_positives']}")
    print(f"      (Dự đoán rủi ro CAO nhưng thực tế THẤP → Từ chối nhầm)")
    print(f"    - False Negatives (FN): {analysis['false_negatives']}")
    print(f"      (Dự đoán rủi ro THẤP nhưng thực tế CAO → Chấp nhận nhầm)")

    # Chi phí lỗi trong tín dụng
    # FN thường nguy hiểm hơn FP vì cho vay nhầm người rủi ro cao
    if analysis['false_negatives'] > 0:
        fn_ratio = analysis['false_negatives'] / (analysis['false_negatives'] + analysis['false_positives'] + 1e-10)
        print(f"\n  ⚠️  Tỷ lệ FN/(FN+FP): {fn_ratio:.2%}")
        if fn_ratio > 0.5:
            print(f"  → Mô hình có xu hướng BỎ LỌT rủi ro (cần cải thiện Recall)")
        else:
            print(f"  → Mô hình có xu hướng TỪ CHỐI nhầm (có thể giảm threshold)")

    # So sánh đặc trưng trung bình giữa FP và FN
    if len(X_df.columns) > 0 and (fp_mask.sum() > 0 or fn_mask.sum() > 0):
        print(f"\n  📊 So sánh đặc trưng trung bình:")
        comparison = pd.DataFrame({
            'TP (mean)': X_df[tp_mask].mean() if tp_mask.sum() > 0 else 0,
            'TN (mean)': X_df[tn_mask].mean() if tn_mask.sum() > 0 else 0,
            'FP (mean)': X_df[fp_mask].mean() if fp_mask.sum() > 0 else 0,
            'FN (mean)': X_df[fn_mask].mean() if fn_mask.sum() > 0 else 0,
        })
        analysis['feature_comparison'] = comparison
        print(comparison.round(3).head(10).to_string())

    print(f"{'='*60}")

    return analysis


# =============================================================================
# 4. LEARNING CURVES (BIAS-VARIANCE)
# =============================================================================

def compute_learning_curves(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    train_sizes: np.ndarray = None,
    scoring: str = 'f1'
) -> dict:
    """
    Tính Learning Curves để phân tích bias-variance.

    Args:
        model: Mô hình cần phân tích.
        X: Toàn bộ dữ liệu đặc trưng.
        y: Toàn bộ nhãn.
        cv: Số fold CV.
        train_sizes: Mảng kích thước tập train (tỷ lệ).
        scoring: Metric đánh giá.

    Returns:
        Dict chứa train_sizes, train_scores, val_scores.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )

    # Tính trung bình và độ lệch chuẩn
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Phân tích bias-variance
    gap = train_mean[-1] - val_mean[-1]

    print(f"\n📈 Learning Curve Analysis:")
    print(f"  Train score (cuối): {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    print(f"  Val score (cuối):   {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
    print(f"  Gap (train - val):  {gap:.4f}")

    if gap > 0.1:
        print(f"  → High Variance (Overfitting): Mô hình quá phức tạp")
        print(f"    Đề xuất: Thêm dữ liệu, regularization, giảm complexity")
    elif val_mean[-1] < 0.6:
        print(f"  → High Bias (Underfitting): Mô hình quá đơn giản")
        print(f"    Đề xuất: Thêm đặc trưng, tăng complexity, bỏ regularization")
    else:
        print(f"  → Mô hình cân bằng tốt giữa bias và variance")

    return {
        'train_sizes': train_sizes_abs,
        'train_mean': train_mean,
        'train_std': train_std,
        'val_mean': val_mean,
        'val_std': val_std,
        'gap': gap,
    }


# =============================================================================
# 5. KIỂM ĐỊNH THỐNG KÊ
# =============================================================================

def statistical_test(
    scores_model_a: np.ndarray,
    scores_model_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    alpha: float = 0.05
) -> dict:
    """
    Kiểm định thống kê so sánh hai mô hình (Paired t-test).

    Args:
        scores_model_a: Điểm CV của mô hình A.
        scores_model_b: Điểm CV của mô hình B.
        model_a_name, model_b_name: Tên mô hình.
        alpha: Mức ý nghĩa.

    Returns:
        Dict chứa kết quả kiểm định.
    """
    t_stat, p_value = stats.ttest_rel(scores_model_a, scores_model_b)

    result = {
        'model_a': model_a_name,
        'model_b': model_b_name,
        'mean_a': scores_model_a.mean(),
        'mean_b': scores_model_b.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
    }

    print(f"\n📊 Kiểm định Paired t-test:")
    print(f"  {model_a_name}: {scores_model_a.mean():.4f} ± {scores_model_a.std():.4f}")
    print(f"  {model_b_name}: {scores_model_b.mean():.4f} ± {scores_model_b.std():.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.6f}")

    if p_value < alpha:
        better = model_a_name if scores_model_a.mean() > scores_model_b.mean() else model_b_name
        print(f"  → Khác biệt CÓ ý nghĩa thống kê (p < {alpha})")
        print(f"  → {better} tốt hơn có ý nghĩa thống kê")
    else:
        print(f"  → Khác biệt KHÔNG có ý nghĩa thống kê (p >= {alpha})")

    return result


# =============================================================================
# 6. TỔNG HỢP BÁO CÁO
# =============================================================================

def generate_report(
    models_metrics: Dict[str, dict],
    dataset_name: str = "Dataset"
) -> str:
    """
    Tạo báo cáo đánh giá tổng hợp dạng text.

    Args:
        models_metrics: Dict {tên_mô_hình: metrics_dict}.
        dataset_name: Tên bộ dữ liệu.

    Returns:
        Chuỗi báo cáo.
    """
    report_lines = [
        "=" * 70,
        f"  BÁO CÁO ĐÁNH GIÁ MÔ HÌNH - {dataset_name}",
        "=" * 70,
        "",
    ]

    # Bảng so sánh
    report_lines.append("1. BẢNG SO SÁNH TỔNG QUAN:")
    report_lines.append("-" * 70)
    header = f"{'Model':25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}"
    report_lines.append(header)
    report_lines.append("-" * 70)

    best_f1 = 0
    best_model = ""

    for name, m in models_metrics.items():
        f1 = m.get('f1_score', 0)
        if f1 > best_f1:
            best_f1 = f1
            best_model = name

        line = (
            f"{name:25s} "
            f"{m.get('accuracy', 0):7.4f} "
            f"{m.get('precision', 0):7.4f} "
            f"{m.get('recall', 0):7.4f} "
            f"{m.get('f1_score', 0):7.4f} "
            f"{m.get('roc_auc', 0):7.4f}"
        )
        report_lines.append(line)

    report_lines.append("-" * 70)
    report_lines.append("")

    # Mô hình tốt nhất
    report_lines.append(f"2. MÔ HÌNH TỐT NHẤT: {best_model} (F1={best_f1:.4f})")
    report_lines.append("")

    # Phân tích lỗi
    report_lines.append("3. PHÂN TÍCH LỖI:")
    for name, m in models_metrics.items():
        fp = m.get('false_positives', 0)
        fn = m.get('false_negatives', 0)
        report_lines.append(f"   {name}: FP={fp}, FN={fn}, FPR={m.get('fpr', 0):.4f}, FNR={m.get('fnr', 0):.4f}")

    report_lines.append("")
    report_lines.append("=" * 70)

    report = "\n".join(report_lines)
    print(report)

    return report
