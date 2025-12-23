# -*- coding: utf-8 -*-
"""
==========================================
Module: feature_engineering
==========================================
Mô tả:
    Module này chịu trách nhiệm tạo và chuyển đổi các đặc trưng (features)
    để chuẩn bị dữ liệu cho việc huấn luyện mô hình.

Các bước chính:
    1. Tạo đặc trưng mới từ dữ liệu có sẵn (ví dụ: tuổi từ năm sinh)
    2. Chuyển đổi đặc trưng thời gian (giờ, phút -> số giờ)
    3. Mã hóa các biến phân loại (Label Encoding)

Các hàm chính:
    - create_age_feature(): Tạo cột tuổi từ năm sinh
    - transform_time_feature(): Chuyển đổi cột thời gian
    - encode_categorical_features(): Mã hóa biến phân loại
    - engineer_features(): Hàm tổng hợp thực hiện tất cả các bước
==========================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional, Tuple


def create_age_feature(
    df: pd.DataFrame,
    birth_year_column: str = 'an_nais',
    reference_year: int = 2019
) -> pd.DataFrame:
    """
    Tạo cột tuổi (age) từ cột năm sinh.

    Công thức:
        age = reference_year - an_nais

    Tham số:
        df: DataFrame cần xử lý
        birth_year_column: Tên cột chứa năm sinh (mặc định: 'an_nais')
        reference_year: Năm tham chiếu để tính tuổi (mặc định: 2019 - năm dữ liệu)

    Trả về:
        DataFrame với cột 'age' mới và cột năm sinh đã được xóa

    Lý do sử dụng 'age' thay vì 'an_nais':
        - 'age' (tuổi) trực quan hơn và dễ hiểu hơn
        - Mô hình học tốt hơn với tuổi vì quan hệ với rủi ro rõ ràng hơn
        - Tuổi cao -> phản ứng chậm hơn -> nguy hiểm hơn
        - Tuổi thấp (thanh niên) -> liều lĩnh hơn -> nguy hiểm hơn
    """
    df = df.copy()

    if birth_year_column not in df.columns:
        print(f"[CẢNH BÁO] Không tìm thấy cột '{birth_year_column}' trong DataFrame")
        return df

    # Tính tuổi = Năm tham chiếu - Năm sinh
    # Ví dụ: 2019 - 1990 = 29 tuổi
    df['age'] = reference_year - df[birth_year_column]

    # Xóa cột năm sinh vì đã có cột tuổi
    df = df.drop(birth_year_column, axis=1)

    print(f"[INFO] Đã tạo cột 'age' từ cột '{birth_year_column}'")
    print(f"[INFO] Tuổi trung bình: {df['age'].mean():.1f}")
    print(f"[INFO] Tuổi min-max: {df['age'].min():.0f} - {df['age'].max():.0f}")

    # Cảnh báo nếu có giá trị tuổi bất thường
    invalid_ages = (df['age'] < 0) | (df['age'] > 120)
    if invalid_ages.sum() > 0:
        print(f"[CẢNH BÁO] Có {invalid_ages.sum()} giá trị tuổi bất thường (<0 hoặc >120)")

    return df


def transform_time_feature(
    df: pd.DataFrame,
    time_column: str = 'hrmn'
) -> pd.DataFrame:
    """
    Chuyển đổi cột thời gian (hrmn) thành số giờ trong ngày.

    Cột 'hrmn' trong dữ liệu gốc có dạng: HH:MM hoặc HH.MM
    Ví dụ: 14:30 hoặc 14.30 = 14 giờ 30 phút = 14.5 giờ

    Tham số:
        df: DataFrame cần xử lý
        time_column: Tên cột chứa thời gian (mặc định: 'hrmn')

    Trả về:
        DataFrame với cột 'hour' mới thể hiện số giờ trong ngày

    Ý nghĩa của đặc trưng thời gian:
        - Tai nạn vào ban đêm (0-6h) thường nghiêm trọng hơn
        - Giờ cao điểm (7-9h, 17-19h) nhiều tai nạn nhưng ít nghiêm trọng
        - Thời gian giúp mô hình học các pattern này
    """
    df = df.copy()

    if time_column not in df.columns:
        print(f"[CẢNH BÁO] Không tìm thấy cột '{time_column}' trong DataFrame")
        return df

    # Chuyển cột thời gian sang số
    # pd.to_numeric với errors='coerce' sẽ chuyển giá trị lỗi thành NaN
    df['hour'] = pd.to_numeric(df[time_column], errors='coerce')

    # Nếu giá trị là dạng decimal (0.xx đến 1.0 biểu diễn 0h-24h)
    # thì nhân với 24 để ra số giờ thực
    # Kiểm tra xem giá trị có nằm trong khoảng 0-1 không
    if df['hour'].max() <= 1.0:
        df['hour'] = df['hour'] * 24
        print("[INFO] Đã chuyển đổi thời gian dạng decimal (0-1) sang giờ (0-24)")

    # Xóa cột thời gian gốc
    if time_column != 'hour':
        df = df.drop(time_column, axis=1, errors='ignore')

    print(f"[INFO] Đã tạo cột 'hour' từ cột '{time_column}'")
    print(f"[INFO] Giờ trung bình: {df['hour'].mean():.1f}")

    return df


def encode_categorical_features(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Mã hóa các biến phân loại (categorical) bằng Label Encoding.

    Label Encoding:
        Chuyển các giá trị text sang số nguyên
        Ví dụ: ['A', 'B', 'C'] -> [0, 1, 2]

    Tham số:
        df: DataFrame cần xử lý
        exclude_columns: Danh sách các cột không cần mã hóa

    Trả về:
        Tuple gồm:
            - DataFrame đã được mã hóa
            - Dictionary chứa các LabelEncoder đã fit (để decode nếu cần)

    Lưu ý về Label Encoding:
        - Ưu điểm: Đơn giản, không tăng số chiều
        - Nhược điểm: Tạo ra thứ tự giả (0 < 1 < 2) cho các biến không có thứ tự
        - XGBoost có thể xử lý tốt vấn đề này vì nó dựa trên cây quyết định
        - Nếu dùng các mô hình khác (như Linear Regression), nên dùng One-Hot Encoding
    """
    df = df.copy()

    if exclude_columns is None:
        exclude_columns = []

    # Tìm các cột có kiểu dữ liệu 'object' (text/string)
    object_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Loại bỏ các cột trong danh sách exclude
    columns_to_encode = [col for col in object_columns if col not in exclude_columns]

    if not columns_to_encode:
        print("[INFO] Không có cột nào cần mã hóa")
        return df, {}

    print(f"[INFO] Các cột sẽ được mã hóa: {columns_to_encode}")

    # Dictionary lưu các LabelEncoder
    encoders = {}

    for col in columns_to_encode:
        # Tạo LabelEncoder mới cho mỗi cột
        le = LabelEncoder()

        # Chuyển tất cả giá trị sang string để tránh lỗi khi có mixed types
        # (một số giá trị là số, một số là text)
        df[col] = df[col].astype(str)

        # Fit và transform: học ánh xạ + áp dụng chuyển đổi
        df[col] = le.fit_transform(df[col])

        # Lưu encoder để có thể decode sau này nếu cần
        encoders[col] = le

        print(f"  ✓ Đã encode cột '{col}' ({len(le.classes_)} giá trị unique)")

    print(f"[INFO] Đã mã hóa {len(columns_to_encode)} cột")

    return df, encoders


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm tổng hợp thực hiện toàn bộ quy trình feature engineering.

    Quy trình:
        1. Tạo cột tuổi (age) từ năm sinh (an_nais)
        2. Chuyển đổi cột thời gian (hrmn) thành giờ (hour)
        3. Mã hóa các biến phân loại bằng Label Encoding

    Tham số:
        df: DataFrame đã qua bước tiền xử lý

    Trả về:
        DataFrame với các đặc trưng đã được chuyển đổi,
        sẵn sàng cho việc huấn luyện mô hình
    """
    print("=" * 60)
    print("BẮT ĐẦU FEATURE ENGINEERING")
    print("=" * 60)

    initial_columns = df.columns.tolist()

    # Bước 1: Tạo cột tuổi
    print("\n" + "-" * 40)
    print("BƯỚC 1: Tạo đặc trưng tuổi (age)")
    print("-" * 40)
    df = create_age_feature(df)

    # Bước 2: Chuyển đổi thời gian
    print("\n" + "-" * 40)
    print("BƯỚC 2: Chuyển đổi đặc trưng thời gian (hour)")
    print("-" * 40)
    df = transform_time_feature(df)

    # Bước 3: Mã hóa biến phân loại
    print("\n" + "-" * 40)
    print("BƯỚC 3: Mã hóa biến phân loại (Label Encoding)")
    print("-" * 40)
    df, encoders = encode_categorical_features(df)

    # Báo cáo tổng kết
    print("\n" + "=" * 60)
    print("HOÀN TẤT FEATURE ENGINEERING")
    print("=" * 60)

    final_columns = df.columns.tolist()
    new_columns = [col for col in final_columns if col not in initial_columns]
    removed_columns = [col for col in initial_columns if col not in final_columns]

    print(f"Cột mới thêm: {new_columns}")
    print(f"Cột đã xóa: {removed_columns}")
    print(f"Tổng số cột hiện tại: {len(final_columns)}")

    # Kiểm tra kiểu dữ liệu
    print("\n[INFO] Kiểu dữ liệu của các cột:")
    print(df.dtypes.value_counts())

    return df


def get_feature_importance_names(
    df: pd.DataFrame,
    target_column: str = 'grav'
) -> List[str]:
    """
    Lấy danh sách tên các cột đặc trưng (không bao gồm cột mục tiêu).

    Tham số:
        df: DataFrame chứa dữ liệu
        target_column: Tên cột mục tiêu cần loại trừ

    Trả về:
        Danh sách tên các cột đặc trưng

    Ứng dụng:
        Sử dụng khi cần hiển thị feature importance của mô hình
    """
    return [col for col in df.columns if col != target_column]
