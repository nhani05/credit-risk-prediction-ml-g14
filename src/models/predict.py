"""
predict.py - Module dự đoán sử dụng mô hình đã huấn luyện.

Cung cấp các hàm để:
    - Dự đoán nhãn (lớp)
    - Dự đoán xác suất
    - Dự đoán hàng loạt (batch)
    - Đánh giá trên dữ liệu mới

Tác giả: Nhóm 14
Môn: Học Máy
"""

import os
import logging
from typing import Union, Optional

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


class CreditRiskPredictor:
    """
    Lớp dự đoán rủi ro tín dụng.

    Đóng gói mô hình, scaler và các bước tiền xử lý
    để dự đoán trên dữ liệu mới.
    """

    def __init__(self, model=None, scaler=None, feature_names: list = None):
        """
        Khởi tạo predictor.

        Args:
            model: Mô hình đã huấn luyện.
            scaler: Scaler đã fit trên dữ liệu train.
            feature_names: Danh sách tên đặc trưng mô hình sử dụng.
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Dự đoán nhãn (0: rủi ro thấp, 1: rủi ro cao).

        Args:
            X: Dữ liệu đầu vào.

        Returns:
            Mảng nhãn dự đoán.
        """
        if self.model is None:
            raise ValueError("Chưa có mô hình! Vui lòng tải mô hình trước.")

        X_processed = self._preprocess(X)
        predictions = self.model.predict(X_processed)

        return predictions

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Dự đoán xác suất cho mỗi lớp.

        Args:
            X: Dữ liệu đầu vào.

        Returns:
            Mảng xác suất [P(lớp 0), P(lớp 1)].
        """
        if self.model is None:
            raise ValueError("Chưa có mô hình!")

        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(
                f"Mô hình {type(self.model).__name__} không hỗ trợ predict_proba"
            )

        X_processed = self._preprocess(X)
        probabilities = self.model.predict_proba(X_processed)

        return probabilities

    def predict_risk_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Dự đoán điểm rủi ro (xác suất thuộc lớp rủi ro cao).

        Args:
            X: Dữ liệu đầu vào.

        Returns:
            Mảng điểm rủi ro [0, 1].
        """
        proba = self.predict_proba(X)
        return proba[:, 1]  # Xác suất lớp 1 (rủi ro cao)

    def predict_with_confidence(
        self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Dự đoán kèm mức độ tin cậy.

        Args:
            X: Dữ liệu đầu vào.
            threshold: Ngưỡng phân loại (mặc định 0.5).

        Returns:
            DataFrame chứa nhãn dự đoán, xác suất và mức tin cậy.
        """
        proba = self.predict_proba(X)
        risk_score = proba[:, 1]

        predictions = (risk_score >= threshold).astype(int)
        confidence = np.where(predictions == 1, risk_score, 1 - risk_score)

        # Phân loại mức tin cậy
        confidence_level = pd.cut(
            confidence,
            bins=[0, 0.6, 0.8, 1.0],
            labels=['Thấp', 'Trung bình', 'Cao']
        )

        result = pd.DataFrame({
            'prediction': predictions,
            'risk_label': np.where(predictions == 1, 'Rủi ro cao', 'Rủi ro thấp'),
            'risk_score': risk_score.round(4),
            'confidence': confidence.round(4),
            'confidence_level': confidence_level,
        })

        return result

    def batch_predict(
        self,
        filepath: str,
        output_path: str = None,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Dự đoán hàng loạt từ file CSV.

        Args:
            filepath: Đường dẫn file CSV đầu vào.
            output_path: Đường dẫn file kết quả. Nếu None, không lưu.
            threshold: Ngưỡng phân loại.

        Returns:
            DataFrame chứa kết quả dự đoán.
        """
        logger.info(f"Đọc dữ liệu từ: {filepath}")
        X = pd.read_csv(filepath)

        results = self.predict_with_confidence(X, threshold)

        # Gộp dữ liệu gốc với kết quả
        output = pd.concat([X.reset_index(drop=True), results], axis=1)

        if output_path:
            output.to_csv(output_path, index=False)
            logger.info(f"Đã lưu kết quả dự đoán: {output_path}")

        # Tóm tắt
        logger.info(f"\n--- Tóm tắt dự đoán ---")
        logger.info(f"Tổng số mẫu: {len(results)}")
        logger.info(f"Rủi ro cao: {(results['prediction'] == 1).sum()} "
                     f"({(results['prediction'] == 1).mean()*100:.1f}%)")
        logger.info(f"Rủi ro thấp: {(results['prediction'] == 0).sum()} "
                     f"({(results['prediction'] == 0).mean()*100:.1f}%)")

        return output

    def _preprocess(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Tiền xử lý dữ liệu trước khi dự đoán.

        Args:
            X: Dữ liệu thô.

        Returns:
            Dữ liệu đã xử lý.
        """
        if isinstance(X, pd.DataFrame):
            # Chọn đúng đặc trưng nếu biết
            if self.feature_names is not None:
                missing = set(self.feature_names) - set(X.columns)
                if missing:
                    raise ValueError(f"Thiếu đặc trưng bắt buộc: {sorted(missing)}")
                X = X[self.feature_names]

        # Chuẩn hóa nếu có scaler
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if hasattr(X, "shape") and len(X.shape) != 2:
            raise ValueError("Dữ liệu đầu vào phải là ma trận 2 chiều.")
        expected = getattr(self.model, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(
                f"Mô hình cần {expected} đặc trưng nhưng nhận được {X.shape[1]}."
            )

        return X

    def save(self, directory: str = None, name: str = 'credit_risk_predictor'):
        """Lưu toàn bộ predictor."""
        if directory is None:
            directory = MODELS_DIR
        os.makedirs(directory, exist_ok=True)

        filepath = os.path.join(directory, f'{name}.joblib')
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
        }, filepath)
        logger.info(f"Đã lưu predictor: {filepath}")

    @classmethod
    def load(cls, directory: str = None, name: str = 'credit_risk_predictor'):
        """Tải predictor từ file."""
        if directory is None:
            directory = MODELS_DIR

        filepath = os.path.join(directory, f'{name}.joblib')
        data = joblib.load(filepath)

        predictor = cls(
            model=data['model'],
            scaler=data['scaler'],
            feature_names=data['feature_names'],
        )
        logger.info(f"Đã tải predictor: {filepath}")
        return predictor
