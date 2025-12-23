# -*- coding: utf-8 -*-
"""
==========================================
Module: src
==========================================
Mô tả:
    Package chính chứa tất cả các module xử lý dữ liệu tai nạn giao thông.

Cấu trúc:
    - data_loader.py: Đọc dữ liệu từ file CSV
    - data_preprocessing.py: Tiền xử lý và làm sạch dữ liệu
    - feature_engineering.py: Tạo và chuyển đổi đặc trưng
    - visualization.py: Trực quan hóa dữ liệu
    - model.py: Huấn luyện và đánh giá mô hình XGBoost

Tác giả: PTDLL Team
==========================================
"""

from .data_loader import load_data
from .data_preprocessing import preprocess_data, clean_data
from .feature_engineering import engineer_features
from .visualization import create_visualizations
from .model import train_model, evaluate_model

__all__ = [
    'load_data',
    'preprocess_data',
    'clean_data',
    'engineer_features',
    'create_visualizations',
    'train_model',
    'evaluate_model'
]
