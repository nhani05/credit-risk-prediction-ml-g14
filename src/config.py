"""
config.py - Module đọc và quản lý cấu hình dự án.

Đọc file configs/config.yaml và cung cấp cấu hình cho toàn bộ pipeline.

Tác giả: Nhóm 14
Môn: Học Máy
"""

import os
from typing import Any

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')


class Config:
    """Lớp quản lý cấu hình dự án."""

    def __init__(self, config_path: str = None):
        """
        Khởi tạo cấu hình.

        Args:
            config_path: Đường dẫn đến file config YAML.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        self.config_path = config_path
        self._config = self._load_config(config_path)

    def _load_config(self, path: str) -> dict:
        """Đọc file YAML."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị cấu hình theo key (hỗ trợ dot notation).

        Ví dụ: config.get('preprocessing.missing_strategy')

        Args:
            key: Khóa cấu hình (hỗ trợ 'parent.child.subchild').
            default: Giá trị mặc định nếu không tìm thấy.

        Returns:
            Giá trị cấu hình.
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def data(self) -> dict:
        """Cấu hình đường dẫn dữ liệu."""
        return self._config.get('data', {})

    @property
    def preprocessing(self) -> dict:
        """Cấu hình tiền xử lý."""
        return self._config.get('preprocessing', {})

    @property
    def feature_engineering(self) -> dict:
        """Cấu hình trích xuất đặc trưng."""
        return self._config.get('feature_engineering', {})

    @property
    def split(self) -> dict:
        """Cấu hình chia dữ liệu."""
        return self._config.get('split', {})

    @property
    def models(self) -> dict:
        """Cấu hình mô hình."""
        return self._config.get('models', {})

    @property
    def tuning(self) -> dict:
        """Cấu hình hyperparameter tuning."""
        return self._config.get('tuning', {})

    @property
    def output(self) -> dict:
        """Cấu hình đường dẫn đầu ra."""
        return self._config.get('output', {})

    def __repr__(self):
        return f"Config(path='{self.config_path}')"


def load_config(config_path: str = None) -> Config:
    """
    Hàm tiện ích để tải cấu hình.

    Args:
        config_path: Đường dẫn file config.

    Returns:
        Config instance.
    """
    return Config(config_path)
