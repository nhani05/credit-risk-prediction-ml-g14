"""
load_data.py - Module nạp và khám phá sơ bộ dữ liệu.

Cung cấp các hàm để đọc dữ liệu từ file CSV và hiển thị
thông tin cơ bản về từng bộ dữ liệu.

Tác giả: Nhóm 14
Môn: Học Máy
"""

import os
import logging

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')


def load_german_credit(data_dir: str = None) -> pd.DataFrame:
    """
    Đọc bộ dữ liệu German Credit.

    Args:
        data_dir: Thư mục chứa file dữ liệu.

    Returns:
        DataFrame chứa dữ liệu German Credit.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    filepath = os.path.join(data_dir, 'german_credit.csv')

    if not os.path.exists(filepath):
        logger.error(f"Không tìm thấy file: {filepath}")
        logger.info("Vui lòng chạy download_data.py trước!")
        raise FileNotFoundError(f"File không tồn tại: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Đã nạp German Credit: {df.shape[0]} mẫu, {df.shape[1]} cột")

    return df


def load_give_me_credit(data_dir: str = None) -> pd.DataFrame:
    """
    Đọc bộ dữ liệu Give Me Some Credit.

    Args:
        data_dir: Thư mục chứa file dữ liệu.

    Returns:
        DataFrame chứa dữ liệu Give Me Some Credit.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    filepath = os.path.join(data_dir, 'cs-training.csv')

    if not os.path.exists(filepath):
        logger.error(f"Không tìm thấy file: {filepath}")
        logger.info("Vui lòng chạy download_data.py hoặc tải thủ công từ Kaggle!")
        raise FileNotFoundError(f"File không tồn tại: {filepath}")

    df = pd.read_csv(filepath)

    # Loại bỏ cột index thừa nếu có
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    logger.info(f"Đã nạp Give Me Some Credit: {df.shape[0]} mẫu, {df.shape[1]} cột")

    return df


def dataset_overview(df: pd.DataFrame, dataset_name: str = "Dataset") -> dict:
    """
    Hiển thị thông tin tổng quan về bộ dữ liệu.

    Args:
        df: DataFrame cần phân tích.
        dataset_name: Tên bộ dữ liệu.

    Returns:
        Dict chứa các thông tin tổng quan.
    """
    print("\n" + "=" * 70)
    print(f"  TỔNG QUAN: {dataset_name}")
    print("=" * 70)

    # Kích thước
    print(f"\n📊 Kích thước: {df.shape[0]} mẫu × {df.shape[1]} đặc trưng")

    # Kiểu dữ liệu
    print(f"\n📋 Kiểu dữ liệu:")
    print(df.dtypes.value_counts().to_string())

    # Giá trị thiếu
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        'Số lượng': missing,
        'Tỷ lệ (%)': missing_pct
    })
    missing_info = missing_info[missing_info['Số lượng'] > 0]

    if len(missing_info) > 0:
        print(f"\n⚠️  Giá trị thiếu ({len(missing_info)} cột):")
        print(missing_info.to_string())
    else:
        print("\n✅ Không có giá trị thiếu!")

    # Thống kê mô tả
    print(f"\n📈 Thống kê mô tả (đặc trưng số):")
    print(df.describe().round(2).to_string())

    # Phân bố biến mục tiêu
    target_col = None
    for col in ['target', 'SeriousDlqin2yrs', 'default', 'label']:
        if col in df.columns:
            target_col = col
            break

    if target_col:
        print(f"\n🎯 Phân bố biến mục tiêu '{target_col}':")
        target_counts = df[target_col].value_counts()
        target_pct = df[target_col].value_counts(normalize=True) * 100
        for val in target_counts.index:
            print(f"   Lớp {val}: {target_counts[val]} ({target_pct[val]:.1f}%)")

        imbalance_ratio = target_counts.max() / target_counts.min()
        print(f"   Tỷ lệ mất cân bằng: {imbalance_ratio:.1f}:1")

    # Các cột phân loại
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        print(f"\n📝 Đặc trưng phân loại ({len(cat_cols)}):")
        for col in cat_cols:
            n_unique = df[col].nunique()
            print(f"   - {col}: {n_unique} giá trị duy nhất")

    # Kiểm tra trùng lặp
    n_duplicates = df.duplicated().sum()
    print(f"\n🔄 Số mẫu trùng lặp: {n_duplicates}")

    info = {
        'shape': df.shape,
        'missing_count': missing.sum(),
        'target_col': target_col,
        'cat_cols': cat_cols,
        'n_duplicates': n_duplicates,
    }

    print("=" * 70)
    return info


def load_all_datasets(data_dir: str = None) -> dict:
    """
    Nạp tất cả bộ dữ liệu và hiển thị tổng quan.

    Returns:
        Dict chứa các DataFrame.
    """
    datasets = {}

    try:
        datasets['german_credit'] = load_german_credit(data_dir)
        dataset_overview(datasets['german_credit'], "German Credit Dataset")
    except FileNotFoundError:
        logger.warning("Bỏ qua German Credit Dataset (chưa tải)")

    try:
        datasets['give_me_credit'] = load_give_me_credit(data_dir)
        dataset_overview(datasets['give_me_credit'], "Give Me Some Credit Dataset")
    except FileNotFoundError:
        logger.warning("Bỏ qua Give Me Some Credit Dataset (chưa tải)")

    return datasets


if __name__ == '__main__':
    datasets = load_all_datasets()
    print(f"\nĐã nạp {len(datasets)} bộ dữ liệu.")
