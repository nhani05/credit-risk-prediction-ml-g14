"""
main.py - Pipeline chính cho dự án Đánh giá Rủi ro Tín dụng.

Chạy toàn bộ pipeline từ tải dữ liệu đến đánh giá mô hình.

Sử dụng:
    python main.py                          # Chạy toàn bộ pipeline
    python main.py --step download          # Chỉ tải dữ liệu
    python main.py --step preprocess        # Chỉ tiền xử lý
    python main.py --step train             # Chỉ huấn luyện
    python main.py --step evaluate          # Chỉ đánh giá
    python main.py --step visualize         # Chỉ trực quan hóa
    python main.py --dataset german_credit  # Chỉ chạy trên 1 dataset
    python main.py --tune                   # Bật hyperparameter tuning

Tác giả: Nhóm 14
Môn: Học Máy
"""

import os
import sys
import argparse
import logging
import warnings

import pandas as pd
import numpy as np

# Thêm thư mục gốc vào path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config
from src.data.download_data import download_all
from src.data.load_data import load_all_datasets, dataset_overview
from src.data.preprocess import preprocess_pipeline
from src.features.feature_engineering import (
    create_credit_features_german, create_credit_features_gmsc,
    feature_engineering_pipeline, get_feature_importance
)
from src.models.train_model import (
    train_all_models, cross_validate_models, tune_all_models,
    create_stacking_model, save_all_models, get_default_models
)
from src.evaluation.evaluate_model import (
    evaluate_model, compare_models, error_analysis,
    compute_learning_curves, generate_report
)
from src.visualization.visualize import (
    create_all_visualizations, plot_roc_curves, plot_shap_analysis,
    plot_learning_curves, plot_calibration_curves
)

# Ẩn warning
warnings.filterwarnings('ignore')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description='Pipeline Đánh giá Rủi ro Tín dụng bằng Machine Learning'
    )
    parser.add_argument(
        '--step', type=str, default='all',
        choices=['all', 'download', 'preprocess', 'train', 'evaluate', 'visualize'],
        help='Bước pipeline cần chạy (mặc định: all)'
    )
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['all', 'german_credit', 'give_me_credit'],
        help='Bộ dữ liệu cần xử lý (mặc định: all)'
    )
    parser.add_argument(
        '--tune', action='store_true',
        help='Bật hyperparameter tuning'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Đường dẫn file cấu hình'
    )
    parser.add_argument(
        '--shap', action='store_true',
        help='Bật phân tích SHAP (tốn thời gian)'
    )

    return parser.parse_args()


def run_pipeline_for_dataset(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    config,
    do_tune: bool = False,
    do_shap: bool = False
) -> dict:
    """
    Chạy pipeline cho một bộ dữ liệu.

    Args:
        df: DataFrame dữ liệu.
        target_col: Tên cột mục tiêu.
        dataset_name: Tên bộ dữ liệu.
        config: Config object.
        do_tune: Bật hyperparameter tuning.
        do_shap: Bật phân tích SHAP.

    Returns:
        Dict kết quả pipeline.
    """
    print("\n" + "🔷" * 35)
    print(f"  PIPELINE: {dataset_name}")
    print("🔷" * 35)

    results = {'dataset_name': dataset_name}

    # =========================================================================
    # BƯỚC 1: Tiền xử lý dữ liệu
    # =========================================================================
    print("\n" + "━" * 60)
    print("  📦 BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU")
    print("━" * 60)

    preprocessed = preprocess_pipeline(
        df=df,
        target_col=target_col,
        test_size=config.get('split.test_size', 0.2),
        random_state=config.get('split.random_state', 42),
        missing_strategy=config.get('preprocessing.missing_strategy', 'median'),
        outlier_method=config.get('preprocessing.outlier_method', 'iqr'),
        encoding_method=config.get('preprocessing.encoding_method', 'label'),
        scaling_method=config.get('preprocessing.scaling_method', 'standard'),
        imbalance_method=config.get('preprocessing.imbalance_method', 'smote'),
        drop_missing_threshold=config.get('preprocessing.drop_missing_threshold', 0.5),
        outlier_factor=config.get('preprocessing.outlier_factor', 1.5),
    )

    X_train = preprocessed['X_train']
    X_test = preprocessed['X_test']
    y_train = preprocessed['y_train']
    y_test = preprocessed['y_test']

    results['preprocessed'] = preprocessed

    # =========================================================================
    # BƯỚC 2: Trích xuất đặc trưng
    # =========================================================================
    print("\n" + "━" * 60)
    print("  🔧 BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG")
    print("━" * 60)

    fe_result = feature_engineering_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        dataset_name=dataset_name,
        feature_selection_method=config.get('feature_engineering.feature_selection_method', 'mutual_info'),
        n_select_features=config.get('feature_engineering.n_select_features', 15),
        use_pca=config.get('feature_engineering.use_pca', False),
    )

    X_train = fe_result['X_train']
    X_test = fe_result['X_test']
    results['feature_engineering'] = fe_result

    # =========================================================================
    # BƯỚC 3: Huấn luyện mô hình
    # =========================================================================
    print("\n" + "━" * 60)
    print("  🤖 BƯỚC 3: HUẤN LUYỆN MÔ HÌNH")
    print("━" * 60)

    # Cross-validation phải dùng train gốc, chưa resample để tránh cùng mẫu tổng hợp
    # bị phân tán giữa các fold. Test luôn giữ nguyên phân bố ban đầu.
    X_cv = preprocessed['X_train_original']
    y_cv = preprocessed['y_train_original']
    print("\n--- Cross-Validation (train chưa resample) ---")
    cv_results = cross_validate_models(
        X_cv, y_cv,
        cv=config.get('cross_validation.n_folds', 5),
        scoring=config.get('cross_validation.scoring', 'f1')
    )
    results['cv_results'] = cv_results

    # Huấn luyện tất cả mô hình
    print("\n--- Huấn luyện ---")
    trained = train_all_models(X_train, y_train)

    # Thêm Stacking Ensemble
    print("\n--- Stacking Ensemble ---")
    stacking = create_stacking_model()
    stacking_result = {}
    try:
        import time
        start = time.time()
        stacking.fit(X_train, y_train)
        stacking_result = {
            'model': stacking,
            'train_time': time.time() - start,
        }
        trained['Stacking Ensemble'] = stacking_result
        logger.info(f"  ✅ Stacking Ensemble - Thời gian: {stacking_result['train_time']:.2f}s")
    except Exception as e:
        logger.error(f"  ❌ Stacking Ensemble: {e}")

    results['trained_models'] = trained

    # Hyperparameter Tuning (tùy chọn)
    if do_tune:
        print("\n--- Hyperparameter Tuning ---")
        tuned = tune_all_models(
            X_cv, y_cv,
            method=config.get('tuning.method', 'random'),
            cv=config.get('tuning.cv', 5),
            scoring=config.get('tuning.scoring', 'f1'),
            n_iter=config.get('tuning.n_iter', 30)
        )
        results['tuned_models'] = tuned

        # Cập nhật models với best estimators
        for name, tune_result in tuned.items():
            if 'best_model' in tune_result:
                trained[name + ' (Tuned)'] = {'model': tune_result['best_model']}

    # =========================================================================
    # BƯỚC 4: Đánh giá mô hình
    # =========================================================================
    print("\n" + "━" * 60)
    print("  📊 BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH")
    print("━" * 60)

    # Đánh giá từng mô hình
    all_metrics = {}
    best_model_name = None
    best_model = None
    for name, result in trained.items():
        model = result.get('model')
        if model is not None:
            try:
                metrics = evaluate_model(model, X_test, y_test, model_name=name)
                all_metrics[name] = metrics
            except Exception as e:
                logger.error(f"Lỗi đánh giá {name}: {e}")

    results['all_metrics'] = all_metrics

    # So sánh mô hình
    comparison_df = compare_models(trained, X_test, y_test)
    results['comparison'] = comparison_df

    # Phân tích lỗi cho mô hình tốt nhất
    if all_metrics:
        best_model_name = max(all_metrics, key=lambda k: all_metrics[k].get('f1_score', 0))
        best_model = trained[best_model_name]['model']

        print(f"\n--- Phân tích lỗi: {best_model_name} ---")
        error_result = error_analysis(
            best_model, X_test, y_test,
            feature_names=X_train.columns.tolist(),
            model_name=best_model_name
        )
        results['error_analysis'] = error_result

        # Learning Curves
        print(f"\n--- Learning Curves: {best_model_name} ---")
        try:
            lc_data = compute_learning_curves(
                best_model, preprocessed['X_train_original'][X_train.columns],
                preprocessed['y_train_original'],
                cv=5, scoring='f1'
            )
            results['learning_curves'] = lc_data
            plot_learning_curves(lc_data, model_name=best_model_name)
        except Exception as e:
            logger.warning(f"Không thể tính Learning Curve: {e}")

    # =========================================================================
    # BƯỚC 5: Trực quan hóa
    # =========================================================================
    print("\n" + "━" * 60)
    print("  📈 BƯỚC 5: TRỰC QUAN HÓA")
    print("━" * 60)

    create_all_visualizations(
        df=df, target_col=target_col,
        models_metrics=all_metrics, y_test=y_test,
        importance_df=fe_result.get('feature_importance')
    )

    # Calibration Curves
    plot_calibration_curves(all_metrics, y_test)

    # SHAP Analysis (điểm cộng)
    if do_shap and best_model_name:
        print("\n--- SHAP Analysis (Điểm cộng) ---")
        try:
            plot_shap_analysis(best_model, X_test, model_name=best_model_name)
        except Exception as e:
            logger.warning(f"Không thể tạo SHAP: {e}")

    # =========================================================================
    # BƯỚC 6: Lưu kết quả
    # =========================================================================
    print("\n" + "━" * 60)
    print("  💾 BƯỚC 6: LƯU KẾT QUẢ")
    print("━" * 60)

    # Lưu mô hình
    model_paths = save_all_models(trained)
    results['model_paths'] = model_paths

    # Lưu báo cáo
    report = generate_report(all_metrics, dataset_name)
    report_path = os.path.join(PROJECT_ROOT, 'reports', f'report_{dataset_name}.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Đã lưu báo cáo: {report_path}")

    # Lưu bảng so sánh
    comparison_path = os.path.join(PROJECT_ROOT, 'reports', f'comparison_{dataset_name}.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Đã lưu bảng so sánh: {comparison_path}")

    return results


def main():
    """Hàm chính chạy toàn bộ pipeline."""
    args = parse_args()

    print("=" * 70)
    print("  🏦 MÔ HÌNH ĐÁNH GIÁ RỦI RO TÍN DỤNG")
    print("  📚 Bài tập lớn môn Học Máy - Nhóm 14")
    print("=" * 70)

    # Tải cấu hình
    config = load_config(args.config)
    logger.info(f"Đã tải cấu hình: {config}")

    # =========================================================================
    # Tải dữ liệu
    # =========================================================================
    if args.step in ['all', 'download']:
        print("\n" + "🔶" * 35)
        print("  BƯỚC 0: TẢI DỮ LIỆU")
        print("🔶" * 35)
        download_all()
        if args.step == 'download':
            return {}

    if args.step in ['preprocess', 'train', 'evaluate', 'visualize']:
        logger.error(
            "--step %s chưa được triển khai độc lập; dùng --step all "
            "hoặc chạy các notebook theo từng giai đoạn.", args.step
        )
        sys.exit(2)

    # =========================================================================
    # Nạp dữ liệu
    # =========================================================================
    datasets = load_all_datasets()

    if not datasets:
        logger.error("Không tìm thấy dữ liệu! Chạy: python main.py --step download")
        sys.exit(1)

    # Chỉ chạy dataset được yêu cầu và báo lỗi nếu dataset đó chưa có.
    if args.dataset != 'all' and args.dataset not in datasets:
        logger.error(
            "Không tìm thấy dữ liệu '%s'. Kiểm tra file trong data/raw/.",
            args.dataset,
        )
        sys.exit(1)

    # =========================================================================
    # Chạy pipeline cho từng dataset
    # =========================================================================
    all_results = {}

    dataset_configs = {
        'german_credit': {
            'target_col': 'target',
            'feature_creator': create_credit_features_german,
        },
        'give_me_credit': {
            'target_col': 'SeriousDlqin2yrs',
            'feature_creator': create_credit_features_gmsc,
        },
    }

    for dataset_name, df in datasets.items():
        if args.dataset != 'all' and args.dataset != dataset_name:
            continue

        ds_config = dataset_configs.get(dataset_name, {})
        target_col = ds_config.get('target_col', 'target')

        # Tạo đặc trưng mới
        feature_creator = ds_config.get('feature_creator')
        if feature_creator:
            df = feature_creator(df)

        result = run_pipeline_for_dataset(
            df=df,
            target_col=target_col,
            dataset_name=dataset_name,
            config=config,
            do_tune=args.tune,
            do_shap=args.shap,
        )
        all_results[dataset_name] = result

    # =========================================================================
    # Tổng kết
    # =========================================================================
    print("\n" + "🟢" * 35)
    print("  ✅ PIPELINE HOÀN TẤT!")
    print("🟢" * 35)

    for ds_name, result in all_results.items():
        comparison = result.get('comparison')
        if comparison is not None:
            print(f"\n📋 {ds_name} - Top 3 mô hình:")
            print(comparison.head(3).to_string(index=False))

    print("\n📁 Kết quả được lưu tại:")
    print(f"   - Mô hình: models/")
    print(f"   - Báo cáo: reports/")
    print(f"   - Biểu đồ: reports/figures/")
    print(f"\n{'='*70}")

    return all_results


if __name__ == '__main__':
    main()
