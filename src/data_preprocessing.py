# -*- coding: utf-8 -*-
"""
==========================================
Module: data_preprocessing
==========================================
Mô tả:
    Module này chịu trách nhiệm tiền xử lý và làm sạch dữ liệu:
    - Xử lý dữ liệu khuyết thiếu (missing values)
    - Loại bỏ các cột không cần thiết hoặc thiếu quá nhiều dữ liệu
    - Xóa các dòng trùng lặp (duplicates)
    - Gộp các lớp mục tiêu (target class mapping)

Các hàm chính:
    - check_missing_data(): Kiểm tra và báo cáo dữ liệu khuyết thiếu
    - remove_unnecessary_columns(): Xóa các cột không cần thiết
    - handle_missing_values(): Xử lý các giá trị khuyết thiếu
    - remove_duplicates(): Xóa các dòng trùng lặp
    - map_target_classes(): Gộp và map các lớp mục tiêu
    - preprocess_data(): Hàm tổng hợp thực hiện toàn bộ tiền xử lý
==========================================
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


def check_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra và tạo báo cáo về dữ liệu khuyết thiếu trong DataFrame.

    Tham số:
        df: DataFrame cần kiểm tra

    Trả về:
        DataFrame chứa thông tin về dữ liệu khuyết thiếu của từng cột:
        - column: Tên cột
        - missing_count: Số lượng giá trị khuyết thiếu
        - missing_percent: Phần trăm giá trị khuyết thiếu

    Ví dụ:
        >>> report = check_missing_data(df)
        >>> print(report[report['missing_count'] > 0])
    """
    # Đếm số lượng giá trị khuyết thiếu trong mỗi cột
    missing_count = df.isna().sum()

    # Tính phần trăm giá trị khuyết thiếu
    missing_percent = (missing_count / len(df)) * 100

    # Tạo DataFrame báo cáo
    report = pd.DataFrame({
        'column': missing_count.index,
        'missing_count': missing_count.values,
        'missing_percent': missing_percent.values
    })

    # Sắp xếp theo số lượng khuyết thiếu giảm dần
    report = report.sort_values('missing_count', ascending=False)
    report = report.reset_index(drop=True)

    # In thông tin tóm tắt
    total_missing = missing_count.sum()
    rows_with_missing = df.isna().any(axis=1).sum()
    print(f"[INFO] Tổng số ô khuyết thiếu: {total_missing:,}")
    print(f"[INFO] Số dòng có ít nhất 1 ô khuyết thiếu: {rows_with_missing:,} ({rows_with_missing/len(df)*100:.2f}%)")

    return report


def remove_unnecessary_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Xóa các cột không cần thiết hoặc chứa quá nhiều dữ liệu khuyết thiếu.

    Tham số:
        df: DataFrame cần xử lý
        columns_to_drop: Danh sách tên cột cần xóa.
            Nếu không cung cấp, sẽ xóa các cột mặc định đã được xác định
            là không hữu ích cho mô hình.

    Trả về:
        DataFrame đã xóa các cột không cần thiết

    Các cột bị xóa mặc định:
        1. Cột thiếu quá nhiều dữ liệu:
           - v2: Biến không xác định (thiếu >90%)
           - lartpc: Chiều rộng dải phân cách (thiếu nhiều)
           - larrout: Chiều rộng đường (thiếu nhiều)
           - occutc: Số người trên xe (thiếu rất nhiều)

        2. Cột không có giá trị dự đoán:
           - Num_Acc: Mã vụ tai nạn (chỉ là ID)
           - id_vehicule: ID phương tiện
           - num_veh: Số thứ tự xe trong vụ tai nạn
           - adr: Địa chỉ cụ thể (quá chi tiết, không khái quát được)
           - voie: Số hiệu đường
           - pr, pr1: Điểm mốc (point repère)
           - lat, long: Tọa độ GPS (quá cụ thể)
    """
    df = df.copy()  # Tạo bản sao để không ảnh hưởng DataFrame gốc

    if columns_to_drop is None:
        # Danh sách các cột mặc định cần xóa
        # Nhóm 1: Các cột thiếu quá nhiều dữ liệu (>50%)
        cols_high_missing = ['v2', 'lartpc', 'larrout', 'occutc']

        # Nhóm 2: Các cột định danh không có giá trị dự đoán
        cols_identifiers = ['Num_Acc', 'id_vehicule', 'num_veh']

        # Nhóm 3: Các cột địa chỉ/vị trí quá cụ thể
        cols_location = ['adr', 'voie', 'pr', 'pr1', 'lat', 'long']

        columns_to_drop = cols_high_missing + cols_identifiers + cols_location

    # Chỉ xóa các cột thực sự tồn tại trong DataFrame
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    missing_cols = [col for col in columns_to_drop if col not in df.columns]

    if missing_cols:
        print(f"[CẢNH BÁO] Các cột sau không tồn tại trong DataFrame: {missing_cols}")

    if existing_cols:
        df = df.drop(existing_cols, axis=1)
        print(f"[INFO] Đã xóa {len(existing_cols)} cột: {existing_cols}")

    print(f"[INFO] Còn lại {len(df.columns)} cột")

    return df


def handle_missing_values(
    df: pd.DataFrame,
    target_column: str = 'grav',
    max_missing_per_row: int = 10
) -> pd.DataFrame:
    """
    Xử lý các giá trị khuyết thiếu trong DataFrame.

    Chiến lược xử lý:
        1. Xóa các dòng thiếu giá trị ở cột mục tiêu (target)
        2. Xóa các dòng thiếu quá nhiều cột (> max_missing_per_row)
        3. Giữ lại các dòng khác (XGBoost có thể xử lý missing values)

    Tham số:
        df: DataFrame cần xử lý
        target_column: Tên cột mục tiêu (không được phép thiếu)
        max_missing_per_row: Số cột tối đa được phép thiếu trong 1 dòng

    Trả về:
        DataFrame đã xử lý missing values

    Lưu ý:
        XGBoost có khả năng xử lý tốt dữ liệu khuyết thiếu bằng cách
        học cách phân chia tối ưu cho các giá trị thiếu trong quá trình
        xây dựng cây. Vì vậy không cần điền tất cả các giá trị thiếu.
    """
    df = df.copy()
    initial_rows = len(df)

    # ============================================================
    # Bước 1: Xóa các dòng thiếu giá trị ở cột mục tiêu
    # ============================================================
    # Lý do: Không thể huấn luyện nếu không biết kết quả thực tế
    if target_column in df.columns:
        missing_target = df[target_column].isna().sum()
        if missing_target > 0:
            df = df.dropna(subset=[target_column])
            print(f"[INFO] Đã xóa {missing_target} dòng thiếu cột mục tiêu '{target_column}'")

    # ============================================================
    # Bước 2: Xóa các dòng thiếu quá nhiều dữ liệu
    # ============================================================
    # Lý do: Dòng thiếu quá nhiều không đáng tin cậy
    # thresh = số cột tối thiểu phải có dữ liệu
    # = tổng số cột - số cột tối đa được phép thiếu
    min_valid_cols = len(df.columns) - max_missing_per_row
    rows_before = len(df)
    df = df.dropna(thresh=min_valid_cols)
    rows_dropped = rows_before - len(df)

    if rows_dropped > 0:
        print(f"[INFO] Đã xóa {rows_dropped} dòng thiếu quá {max_missing_per_row} cột")

    # ============================================================
    # In báo cáo tổng kết
    # ============================================================
    total_dropped = initial_rows - len(df)
    print(f"[INFO] Tổng số dòng đã xóa: {total_dropped} ({total_dropped/initial_rows*100:.2f}%)")
    print(f"[INFO] Số dòng còn lại: {len(df):,}")

    # In thông tin về các cột vẫn còn missing (XGBoost sẽ xử lý)
    remaining_missing = df.isna().sum()
    cols_with_missing = remaining_missing[remaining_missing > 0]
    if len(cols_with_missing) > 0:
        print(f"[INFO] Vẫn còn {len(cols_with_missing)} cột có dữ liệu khuyết thiếu")
        print("[INFO] XGBoost sẽ tự động xử lý các giá trị này")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xóa các dòng trùng lặp trong DataFrame.

    Tham số:
        df: DataFrame cần xử lý

    Trả về:
        DataFrame đã xóa các dòng trùng lặp

    Lưu ý:
        - Giữ lại dòng đầu tiên trong các dòng trùng lặp
        - Hai dòng được coi là trùng lặp nếu TẤT CẢ các cột đều giống nhau
    """
    df = df.copy()

    # Đếm số dòng trùng lặp
    num_duplicates = df.duplicated().sum()

    if num_duplicates > 0:
        # Xóa các dòng trùng lặp, giữ lại dòng đầu tiên
        df = df.drop_duplicates(keep='first')
        print(f"[INFO] Đã xóa {num_duplicates} dòng trùng lặp")
    else:
        print("[INFO] Không có dòng trùng lặp")

    print(f"[INFO] Số dòng còn lại: {len(df):,}")

    return df


def map_target_classes(
    df: pd.DataFrame,
    target_column: str = 'grav',
    mapping: Optional[Dict[int, int]] = None
) -> pd.DataFrame:
    """
    Ánh xạ (map) các lớp mục tiêu sang các lớp mới.

    Mục đích:
        Giảm số lớp từ 4 xuống 3 bằng cách gộp các lớp tương tự:
        - Lớp 2 (Tử vong) và Lớp 3 (Bị thương nặng) -> Lớp 2 (Nghiêm trọng)

    Ánh xạ mặc định:
        1 (Không bị thương)  -> 0 (Không bị thương)
        4 (Bị thương nhẹ)    -> 1 (Bị thương nhẹ)
        2 (Tử vong)          -> 2 (Nghiêm trọng)
        3 (Bị thương nặng)   -> 2 (Nghiêm trọng)

    Tham số:
        df: DataFrame cần xử lý
        target_column: Tên cột mục tiêu
        mapping: Dictionary ánh xạ giá trị cũ -> giá trị mới
            Nếu không cung cấp, sẽ sử dụng ánh xạ mặc định

    Trả về:
        DataFrame với cột mục tiêu đã được ánh xạ

    Lý do gộp lớp:
        - Lớp 2 (Tử vong) và Lớp 3 (Nặng) đều là trường hợp nghiêm trọng
        - Việc gộp giúp cân bằng dữ liệu tốt hơn
        - Mô hình tập trung vào phân biệt: không thương tích / nhẹ / nghiêm trọng
    """
    df = df.copy()

    if mapping is None:
        # Ánh xạ mặc định: Gộp Tử vong (2) và Nặng (3) thành Nghiêm trọng (2)
        mapping = {
            1: 0,  # Không bị thương -> Class 0
            4: 1,  # Bị thương nhẹ -> Class 1
            2: 2,  # Tử vong -> Class 2 (Nghiêm trọng)
            3: 2   # Bị thương nặng -> Class 2 (Nghiêm trọng)
        }

    # In thông tin phân bố trước khi gộp
    print("[INFO] Phân bố lớp mục tiêu TRƯỚC khi gộp:")
    print(df[target_column].value_counts().sort_index())

    # Thực hiện ánh xạ
    df[target_column] = df[target_column].map(mapping)

    # Kiểm tra xem có giá trị nào không được map không
    unmapped = df[target_column].isna().sum()
    if unmapped > 0:
        print(f"[CẢNH BÁO] Có {unmapped} giá trị không được ánh xạ (trở thành NaN)")

    # In thông tin phân bố sau khi gộp
    print("\n[INFO] Phân bố lớp mục tiêu SAU khi gộp:")
    print(df[target_column].value_counts().sort_index())

    # In ý nghĩa các lớp mới
    print("\n[INFO] Ý nghĩa các lớp sau khi gộp:")
    print("  0 = Không bị thương")
    print("  1 = Bị thương nhẹ")
    print("  2 = Nghiêm trọng (bao gồm tử vong và bị thương nặng)")

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm tổng hợp thực hiện toàn bộ quy trình tiền xử lý dữ liệu.

    Quy trình:
        1. Kiểm tra dữ liệu khuyết thiếu
        2. Xóa các cột không cần thiết
        3. Xử lý giá trị khuyết thiếu
        4. Xóa dòng trùng lặp
        5. Ánh xạ lớp mục tiêu

    Tham số:
        df: DataFrame gốc sau khi đã gộp các bảng

    Trả về:
        DataFrame đã được tiền xử lý, sẵn sàng cho bước feature engineering
    """
    print("=" * 60)
    print("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    initial_shape = df.shape
    print(f"\n[INFO] Kích thước ban đầu: {initial_shape[0]:,} dòng x {initial_shape[1]} cột")

    # Bước 1: Kiểm tra dữ liệu khuyết thiếu
    print("\n" + "-" * 40)
    print("BƯỚC 1: Kiểm tra dữ liệu khuyết thiếu")
    print("-" * 40)
    _ = check_missing_data(df)

    # Bước 2: Xóa các cột không cần thiết
    print("\n" + "-" * 40)
    print("BƯỚC 2: Xóa các cột không cần thiết")
    print("-" * 40)
    df = remove_unnecessary_columns(df)

    # Bước 3: Xử lý giá trị khuyết thiếu
    print("\n" + "-" * 40)
    print("BƯỚC 3: Xử lý giá trị khuyết thiếu")
    print("-" * 40)
    df = handle_missing_values(df)

    # Bước 4: Xóa dòng trùng lặp
    print("\n" + "-" * 40)
    print("BƯỚC 4: Xóa dòng trùng lặp")
    print("-" * 40)
    df = remove_duplicates(df)

    # Bước 5: Ánh xạ lớp mục tiêu
    print("\n" + "-" * 40)
    print("BƯỚC 5: Ánh xạ lớp mục tiêu")
    print("-" * 40)
    df = map_target_classes(df)

    # Báo cáo tổng kết
    print("\n" + "=" * 60)
    print("HOÀN TẤT TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    final_shape = df.shape
    print(f"Kích thước ban đầu: {initial_shape[0]:,} dòng x {initial_shape[1]} cột")
    print(f"Kích thước cuối cùng: {final_shape[0]:,} dòng x {final_shape[1]} cột")
    rows_removed = initial_shape[0] - final_shape[0]
    cols_removed = initial_shape[1] - final_shape[1]
    print(f"Đã xóa: {rows_removed:,} dòng ({rows_removed/initial_shape[0]*100:.1f}%)")
    print(f"Đã xóa: {cols_removed} cột")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alias cho hàm preprocess_data() để tương thích ngược.

    Xem tài liệu của hàm preprocess_data() để biết chi tiết.
    """
    return preprocess_data(df)
