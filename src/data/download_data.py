"""
download_data.py - Module tải dữ liệu cho dự án đánh giá rủi ro tín dụng.

Tải 2 bộ dữ liệu:
    1. German Credit Dataset (UCI Machine Learning Repository)
    2. Give Me Some Credit (Kaggle)

Tác giả: Nhóm 14
Môn: Học Máy
"""

import os
import urllib.request
import zipfile
import logging

import pandas as pd

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Đường dẫn thư mục gốc của dự án
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')


def ensure_dir(directory: str) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(directory, exist_ok=True)


def download_german_credit(save_dir: str = None) -> str:
    """
    Tải bộ dữ liệu German Credit từ UCI Repository.

    Bộ dữ liệu German Credit chứa 1000 mẫu với 20 đặc trưng,
    phân loại khách hàng thành rủi ro tốt (1) hoặc xấu (2).

    Args:
        save_dir: Thư mục lưu file. Mặc định là data/raw/

    Returns:
        Đường dẫn đến file dữ liệu đã tải.
    """
    if save_dir is None:
        save_dir = RAW_DATA_DIR
    ensure_dir(save_dir)

    # URL dữ liệu German Credit từ UCI
    url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    url_names = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"

    filepath_data = os.path.join(save_dir, 'german_credit.data')
    filepath_names = os.path.join(save_dir, 'german_credit.doc')

    if os.path.exists(filepath_data):
        logger.info(f"File đã tồn tại: {filepath_data}")
        return filepath_data

    try:
        logger.info("Đang tải German Credit Dataset từ UCI...")
        urllib.request.urlretrieve(url_data, filepath_data)
        urllib.request.urlretrieve(url_names, filepath_names)
        logger.info(f"Tải thành công! Lưu tại: {filepath_data}")

        # Đọc và lưu dưới dạng CSV với tên cột
        column_names = [
            'checking_account', 'duration', 'credit_history', 'purpose',
            'credit_amount', 'savings_account', 'employment', 'installment_rate',
            'personal_status', 'other_debtors', 'residence_since', 'property',
            'age', 'other_installment', 'housing', 'existing_credits',
            'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
        ]
        df = pd.read_csv(filepath_data, sep=' ', header=None, names=column_names)

        # Chuyển target: 1 = Good (0), 2 = Bad (1) -> nhị phân
        df['target'] = df['target'].map({1: 0, 2: 1})

        csv_path = os.path.join(save_dir, 'german_credit.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Đã lưu CSV tại: {csv_path}")
        logger.info(f"Kích thước: {df.shape[0]} mẫu, {df.shape[1]} đặc trưng")

        return csv_path

    except Exception as e:
        logger.error(f"Lỗi khi tải German Credit Dataset: {e}")
        raise


def download_give_me_credit(save_dir: str = None) -> str:
    """
    Tải bộ dữ liệu Give Me Some Credit.

    Bộ dữ liệu từ cuộc thi Kaggle, chứa ~150,000 mẫu với 10 đặc trưng tài chính,
    dự đoán xác suất khách hàng sẽ gặp khó khăn tài chính trong 2 năm tới.

    Lưu ý: Do giới hạn API Kaggle, file cần được tải thủ công từ:
    https://www.kaggle.com/c/GiveMeSomeCredit/data

    Nếu đã có file cs-training.csv trong data/raw/, hàm sẽ bỏ qua.

    Args:
        save_dir: Thư mục lưu file. Mặc định là data/raw/

    Returns:
        Đường dẫn đến file dữ liệu.
    """
    if save_dir is None:
        save_dir = RAW_DATA_DIR
    ensure_dir(save_dir)

    filepath = os.path.join(save_dir, 'cs-training.csv')

    if os.path.exists(filepath):
        logger.info(f"File đã tồn tại: {filepath}")
        return filepath

    # Thử tải bằng kaggle API nếu có
    try:
        logger.info("Đang thử tải Give Me Some Credit từ Kaggle API...")
        import subprocess
        subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', 'GiveMeSomeCredit',
             '-p', save_dir],
            check=True, capture_output=True, text=True
        )

        # Chỉ trích xuất đúng CSV cần thiết; không giải nén toàn bộ archive tải về.
        zip_path = os.path.join(save_dir, 'GiveMeSomeCredit.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as archive:
                member = next(
                    (name for name in archive.namelist()
                     if os.path.basename(name) == 'cs-training.csv'),
                    None,
                )
                if member is None:
                    raise FileNotFoundError(
                        "Archive Kaggle không chứa cs-training.csv. "
                        "Kiểm tra file tải về và giải nén thủ công."
                    )
                with archive.open(member) as source, open(filepath, 'wb') as target:
                    target.write(source.read())
            os.remove(zip_path)
            logger.info(f"Đã trích xuất dữ liệu cần thiết: {filepath}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Kaggle API không tạo ra {filepath}; hãy kiểm tra cấu trúc archive."
            )
        return filepath

    except (ImportError, FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.warning(f"Không thể tải tự động: {e}")
        logger.info(
            "\n" + "="*60 + "\n"
            "HƯỚNG DẪN TẢI THỦ CÔNG:\n"
            "1. Truy cập: https://www.kaggle.com/c/GiveMeSomeCredit/data\n"
            "2. Tải file 'cs-training.csv'\n"
            f"3. Lưu vào: {save_dir}/\n"
            "="*60
        )

        # Không tạo dữ liệu giả dưới tên bộ dữ liệu thật; yêu cầu tải thủ công.
        raise FileNotFoundError(
            f"Không tìm thấy {filepath}. Hãy tải cs-training.csv từ "
            "https://www.kaggle.com/competitions/GiveMeSomeCredit/data "
            "hoặc cấu hình Kaggle API rồi chạy lại."
        ) from e


# Xóa hàm tạo mẫu trước đó để không nhầm dữ liệu tổng hợp với dữ liệu thật.
# Dữ liệu mẫu nếu cần cho unit test nên được tạo trực tiếp trong tests/.

def _create_sample_give_me_credit(filepath: str) -> None:
    """Không hỗ trợ tạo dữ liệu giả trong thư mục dữ liệu thật."""
    raise RuntimeError(
        "Không tạo dữ liệu mô phỏng trong data/raw. Tạo fixture test riêng "
        "trong tests/ và ghi nhãn rõ là dữ liệu tổng hợp."
    )


def _create_sample_give_me_credit(filepath: str) -> None:
    """
    Tạo dữ liệu mẫu mô phỏng cấu trúc Give Me Some Credit.
    Dùng để test pipeline khi chưa tải được dữ liệu thật.
    """
    import numpy as np

    np.random.seed(42)
    n_samples = 5000  # Dữ liệu mẫu nhỏ

    data = {
        'SeriousDlqin2yrs': np.random.binomial(1, 0.07, n_samples),
        'RevolvingUtilizationOfUnsecuredLines': np.random.exponential(0.5, n_samples),
        'age': np.random.randint(21, 80, n_samples),
        'NumberOfTime30-59DaysPastDueNotWorse': np.random.poisson(0.4, n_samples),
        'DebtRatio': np.random.exponential(1.0, n_samples),
        'MonthlyIncome': np.where(
            np.random.random(n_samples) > 0.2,
            np.random.lognormal(8.5, 0.8, n_samples),
            np.nan  # ~20% missing values như dữ liệu thật
        ),
        'NumberOfOpenCreditLinesAndLoans': np.random.poisson(8, n_samples),
        'NumberOfTimes90DaysLate': np.random.poisson(0.1, n_samples),
        'NumberRealEstateLoansOrLines': np.random.poisson(1, n_samples),
        'NumberOfTime60-89DaysPastDueNotWorse': np.random.poisson(0.1, n_samples),
        'NumberOfDependents': np.where(
            np.random.random(n_samples) > 0.03,
            np.random.poisson(0.8, n_samples),
            np.nan  # ~3% missing
        ),
    }

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=True, index_label='Unnamed: 0')
    logger.info(f"Đã tạo dữ liệu mẫu: {filepath} ({n_samples} mẫu)")


def download_all(save_dir: str = None) -> dict:
    """
    Tải tất cả bộ dữ liệu.

    Returns:
        Dict chứa đường dẫn đến các file dữ liệu.
    """
    paths = {}

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TẢI DỮ LIỆU")
    logger.info("=" * 60)

    paths['german_credit'] = download_german_credit(save_dir)
    try:
        paths['give_me_credit'] = download_give_me_credit(save_dir)
    except FileNotFoundError as exc:
        # Kaggle yêu cầu đăng nhập/điều khoản; cho phép tiếp tục với German Credit.
        logger.warning("Bỏ qua Give Me Some Credit: %s", exc)

    logger.info("=" * 60)
    logger.info("TẢI CÁC BỘ DỮ LIỆU CÓ THỂ TRUY CẬP HOÀN TẤT!")
    logger.info("Đã sẵn sàng: %s", list(paths))
    logger.info("=" * 60)

    return paths


if __name__ == '__main__':
    download_all()
