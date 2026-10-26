"""
visualize.py - Module trực quan hóa dữ liệu và kết quả mô hình.

Bao gồm:
    - EDA: Phân bố, tương quan, boxplot
    - Feature Importance
    - ROC Curves, Precision-Recall Curves
    - Confusion Matrix Heatmap
    - Learning Curves
    - SHAP Values (Điểm cộng: kiến thức mới)
    - Calibration Curves

Tác giả: Nhóm 14
Môn: Học Máy
"""

import os
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')

# Cấu hình matplotlib cho tiếng Việt và đẹp hơn
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
sns.set_style('whitegrid')
sns.set_palette('husl')


def _save_figure(fig, filename: str, save_dir: str = None) -> str:
    """Lưu figure ra file."""
    if save_dir is None:
        save_dir = FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"Đã lưu hình: {filepath}")
    return filepath


# =============================================================================
# 1. EDA - PHÂN TÍCH KHÁM PHÁ DỮ LIỆU
# =============================================================================

def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    title: str = "Phân bố biến mục tiêu",
    save: bool = True
) -> None:
    """Vẽ biểu đồ phân bố biến mục tiêu."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Biểu đồ cột
    counts = df[target_col].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(counts.index.astype(str), counts.values, color=colors)
    axes[0].set_title('Số lượng mỗi lớp')
    axes[0].set_xlabel('Lớp')
    axes[0].set_ylabel('Số lượng')
    for i, (idx, val) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(i, val + len(df)*0.01, str(val), ha='center', fontweight='bold')

    # Biểu đồ tròn
    labels = [f'Lớp {i}\n({v/len(df)*100:.1f}%)' for i, v in zip(counts.index, counts.values)]
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=[0.05]*len(counts))
    axes[1].set_title('Tỷ lệ mỗi lớp')

    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save:
        _save_figure(fig, 'target_distribution.png')
    else:
        plt.show()


def plot_feature_distributions(
    df: pd.DataFrame,
    target_col: str = None,
    max_features: int = 12,
    save: bool = True
) -> None:
    """Vẽ phân bố các đặc trưng số."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    cols_to_plot = numeric_cols[:max_features]
    n_cols = 3
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()

    for i, col in enumerate(cols_to_plot):
        if target_col and target_col in df.columns:
            for label in df[target_col].unique():
                subset = df[df[target_col] == label][col].dropna()
                axes[i].hist(subset, bins=30, alpha=0.6, label=f'Lớp {label}')
            axes[i].legend()
        else:
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='steelblue')
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('')

    # Ẩn axes thừa
    for j in range(len(cols_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Phân bố các đặc trưng số', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save:
        _save_figure(fig, 'feature_distributions.png')
    else:
        plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = 'pearson',
    save: bool = True
) -> None:
    """Vẽ heatmap tương quan giữa các đặc trưng."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, square=True, linewidths=0.5,
        ax=ax, vmin=-1, vmax=1,
        cbar_kws={'shrink': 0.8, 'label': 'Hệ số tương quan'}
    )

    ax.set_title(f'Ma trận tương quan ({method.capitalize()})', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save:
        _save_figure(fig, 'correlation_heatmap.png')
    else:
        plt.show()


def plot_boxplots(
    df: pd.DataFrame,
    target_col: str = None,
    max_features: int = 12,
    save: bool = True
) -> None:
    """Vẽ boxplot để phát hiện ngoại lai."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    cols_to_plot = numeric_cols[:max_features]
    n_cols = 3
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        if target_col and target_col in df.columns:
            df.boxplot(column=col, by=target_col, ax=axes[i])
        else:
            axes[i].boxplot(df[col].dropna())
        axes[i].set_title(col, fontsize=10)

    for j in range(len(cols_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Boxplot - Phát hiện ngoại lai', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save:
        _save_figure(fig, 'boxplots.png')
    else:
        plt.show()


def plot_missing_values(df: pd.DataFrame, save: bool = True) -> None:
    """Vẽ biểu đồ giá trị thiếu."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if len(missing) == 0:
        logger.info("Không có giá trị thiếu để vẽ.")
        return

    missing_pct = missing / len(df) * 100

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
    bars = ax.barh(missing.index, missing_pct.values, color='coral')
    ax.set_xlabel('Tỷ lệ thiếu (%)')
    ax.set_title('Tỷ lệ giá trị thiếu theo cột', fontsize=14, fontweight='bold')

    for bar, pct, count in zip(bars, missing_pct.values, missing.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% ({count})', va='center')

    plt.tight_layout()

    if save:
        _save_figure(fig, 'missing_values.png')
    else:
        plt.show()


# =============================================================================
# 2. FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Mức độ quan trọng của đặc trưng",
    save: bool = True
) -> None:
    """
    Vẽ biểu đồ mức độ quan trọng của đặc trưng.

    Args:
        importance_df: DataFrame với cột 'feature' và 'importance'.
        top_n: Số đặc trưng hiển thị.
        title: Tiêu đề.
    """
    top = importance_df.head(top_n).sort_values('importance')

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top)))[::-1]
    ax.barh(top['feature'], top['importance'], color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save:
        _save_figure(fig, 'feature_importance.png')
    else:
        plt.show()


# =============================================================================
# 3. ROC CURVES
# =============================================================================

def plot_roc_curves(
    models_metrics: Dict[str, dict],
    y_test: pd.Series,
    title: str = "Đường cong ROC - So sánh mô hình",
    save: bool = True
) -> None:
    """
    Vẽ đường cong ROC cho nhiều mô hình trên cùng một biểu đồ.

    Args:
        models_metrics: Dict {tên: metrics_dict} (cần có 'y_proba').
        y_test: Nhãn thực.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(models_metrics)))

    for (name, metrics), color in zip(models_metrics.items(), colors):
        y_proba = metrics.get('y_proba')
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC = {roc_auc:.4f})')

    # Đường ngẫu nhiên
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Ngẫu nhiên (AUC = 0.5)')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Tỷ lệ dương tính giả (FPR)', fontsize=12)
    ax.set_ylabel('Tỷ lệ dương tính thật (TPR)', fontsize=12)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        _save_figure(fig, 'roc_curves.png')
    else:
        plt.show()


# =============================================================================
# 4. PRECISION-RECALL CURVES
# =============================================================================

def plot_precision_recall_curves(
    models_metrics: Dict[str, dict],
    y_test: pd.Series,
    title: str = "Đường cong Precision-Recall",
    save: bool = True
) -> None:
    """Vẽ đường cong Precision-Recall cho nhiều mô hình."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(models_metrics)))

    for (name, metrics), color in zip(models_metrics.items(), colors):
        y_proba = metrics.get('y_proba')
        if y_proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            ax.plot(recall, precision, color=color, lw=2,
                    label=f'{name} (AP = {ap:.4f})')

    # Đường baseline
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
               label=f'Baseline (P = {baseline:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        _save_figure(fig, 'precision_recall_curves.png')
    else:
        plt.show()


# =============================================================================
# 5. CONFUSION MATRIX HEATMAP
# =============================================================================

def plot_confusion_matrices(
    models_metrics: Dict[str, dict],
    save: bool = True
) -> None:
    """Vẽ confusion matrix cho nhiều mô hình."""
    n_models = len(models_metrics)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (name, metrics) in enumerate(models_metrics.items()):
        cm = metrics.get('confusion_matrix')
        if cm is not None:
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Rủi ro thấp', 'Rủi ro cao'],
                yticklabels=['Rủi ro thấp', 'Rủi ro cao']
            )
            axes[i].set_title(name, fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Thực tế')
            axes[i].set_xlabel('Dự đoán')

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save:
        _save_figure(fig, 'confusion_matrices.png')
    else:
        plt.show()


# =============================================================================
# 6. LEARNING CURVES
# =============================================================================

def plot_learning_curves(
    learning_data: dict,
    model_name: str = "Model",
    save: bool = True
) -> None:
    """Vẽ learning curve (bias-variance analysis)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    train_sizes = learning_data['train_sizes']
    train_mean = learning_data['train_mean']
    train_std = learning_data['train_std']
    val_mean = learning_data['val_mean']
    val_std = learning_data['val_std']

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation Score')

    ax.set_xlabel('Số mẫu huấn luyện', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        _save_figure(fig, f'learning_curve_{model_name.lower().replace(" ", "_")}.png')
    else:
        plt.show()


# =============================================================================
# 7. SHAP VALUES (ĐIỂM CỘNG - KIẾN THỨC MỚI)
# =============================================================================

def plot_shap_analysis(
    model,
    X: pd.DataFrame,
    model_name: str = "Model",
    max_display: int = 15,
    save: bool = True
) -> None:
    """
    Trực quan hóa SHAP values cho mô hình (Model Interpretability).

    SHAP (SHapley Additive exPlanations) là phương pháp giải thích mô hình
    dựa trên lý thuyết trò chơi, giúp hiểu đóng góp của từng đặc trưng
    vào kết quả dự đoán.

    Đây là kiến thức bổ sung nằm ngoài chương trình (Điểm cộng).

    Args:
        model: Mô hình đã train.
        X: Dữ liệu đặc trưng.
        model_name: Tên mô hình.
        max_display: Số đặc trưng hiển thị.
    """
    try:
        import shap
    except ImportError:
        logger.warning("Chưa cài đặt thư viện shap. Chạy: pip install shap")
        return

    logger.info(f"Đang tính SHAP values cho {model_name}...")

    # Chọn explainer phù hợp
    model_type = type(model).__name__

    if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier',
                       'XGBClassifier', 'LGBMClassifier']:
        explainer = shap.TreeExplainer(model)
    else:
        # Sử dụng sample nhỏ cho mô hình khác (chậm hơn)
        X_sample = X.sample(n=min(100, len(X)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)

    shap_values = explainer.shap_values(X.head(500))  # Giới hạn để tăng tốc

    # Nếu là list (multi-class), lấy class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X.head(500),
        max_display=max_display, show=False
    )
    plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        _save_figure(plt.gcf(), f'shap_summary_{model_name.lower().replace(" ", "_")}.png')
    else:
        plt.show()

    # Bar Plot (mean absolute SHAP)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X.head(500),
        plot_type='bar', max_display=max_display, show=False
    )
    plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        _save_figure(plt.gcf(), f'shap_bar_{model_name.lower().replace(" ", "_")}.png')
    else:
        plt.show()

    logger.info(f"Đã hoàn tất phân tích SHAP cho {model_name}")


# =============================================================================
# 8. CALIBRATION CURVES
# =============================================================================

def plot_calibration_curves(
    models_metrics: Dict[str, dict],
    y_test: pd.Series,
    n_bins: int = 10,
    save: bool = True
) -> None:
    """Vẽ đường cong Calibration cho nhiều mô hình."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, len(models_metrics)))

    for (name, metrics), color in zip(models_metrics.items(), colors):
        y_proba = metrics.get('y_proba')
        if y_proba is not None:
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)
            axes[0].plot(prob_pred, prob_true, 'o-', color=color, label=name)
            axes[1].hist(y_proba, bins=50, alpha=0.3, color=color, label=name)

    # Đường hoàn hảo
    axes[0].plot([0, 1], [0, 1], 'k--', label='Calibration hoàn hảo')
    axes[0].set_xlabel('Xác suất dự đoán trung bình')
    axes[0].set_ylabel('Tỷ lệ dương tính thực')
    axes[0].set_title('Calibration Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Xác suất dự đoán')
    axes[1].set_ylabel('Số lượng')
    axes[1].set_title('Phân bố xác suất dự đoán', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')

    plt.tight_layout()

    if save:
        _save_figure(fig, 'calibration_curves.png')
    else:
        plt.show()


# =============================================================================
# 9. TỔNG HỢP - VẼ TẤT CẢ
# =============================================================================

def create_all_visualizations(
    df: pd.DataFrame,
    target_col: str,
    models_metrics: Dict[str, dict] = None,
    y_test: pd.Series = None,
    importance_df: pd.DataFrame = None,
    save: bool = True
) -> None:
    """
    Tạo tất cả biểu đồ trực quan.

    Args:
        df: DataFrame gốc (cho EDA).
        target_col: Tên cột mục tiêu.
        models_metrics: Dict metrics của các mô hình.
        y_test: Nhãn test.
        importance_df: DataFrame feature importance.
    """
    logger.info("=" * 60)
    logger.info("TẠO BIỂU ĐỒ TRỰC QUAN")
    logger.info("=" * 60)

    # EDA
    logger.info("\n[1/8] Phân bố biến mục tiêu...")
    plot_target_distribution(df, target_col, save=save)

    logger.info("[2/8] Phân bố đặc trưng...")
    plot_feature_distributions(df, target_col, save=save)

    logger.info("[3/8] Ma trận tương quan...")
    plot_correlation_heatmap(df, save=save)

    logger.info("[4/8] Boxplots...")
    plot_boxplots(df, target_col, save=save)

    logger.info("[5/8] Giá trị thiếu...")
    plot_missing_values(df, save=save)

    # Model evaluation plots
    if models_metrics and y_test is not None:
        logger.info("[6/8] ROC Curves...")
        plot_roc_curves(models_metrics, y_test, save=save)

        logger.info("[7/8] Precision-Recall Curves...")
        plot_precision_recall_curves(models_metrics, y_test, save=save)

        logger.info("[8/8] Confusion Matrices...")
        plot_confusion_matrices(models_metrics, save=save)

    if importance_df is not None:
        plot_feature_importance(importance_df, save=save)

    logger.info("\nĐã tạo xong tất cả biểu đồ!")
    logger.info(f"Thư mục: {FIGURES_DIR}")
