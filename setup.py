"""
Setup file cho dự án Đánh giá Rủi ro Tín dụng.
"""

from setuptools import setup, find_packages

setup(
    name='credit-risk-prediction',
    version='1.0.0',
    description='Mô hình đánh giá rủi ro tín dụng sử dụng Machine Learning',
    author='Nhóm 14',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'xgboost>=2.0.0',
        'lightgbm>=4.0.0',
        'imbalanced-learn>=0.11.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'shap>=0.42.0',
        'pyyaml>=6.0',
        'joblib>=1.3.0',
        'scipy>=1.11.0',
    ],
)
