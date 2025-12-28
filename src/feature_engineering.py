import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional, Tuple


def create_age_feature(
    df: pd.DataFrame,
    birth_year_column: str = 'an_nais',
    reference_year: int = 2019
) -> pd.DataFrame:
    df = df.copy()

    if birth_year_column not in df.columns:
        print(f"[CẢNH BÁO] Không tìm thấy cột '{birth_year_column}' trong DataFrame")
        return df

    df['age'] = reference_year - df[birth_year_column]

    df = df.drop(birth_year_column, axis=1)

    print(f"[INFO] Đã tạo cột 'age' từ cột '{birth_year_column}'")
    print(f"[INFO] Tuổi trung bình: {df['age'].mean():.1f}")
    print(f"[INFO] Tuổi min-max: {df['age'].min():.0f} - {df['age'].max():.0f}")

    invalid_ages = (df['age'] < 0) | (df['age'] > 120)
    if invalid_ages.sum() > 0:
        print(f"[CẢNH BÁO] Có {invalid_ages.sum()} giá trị tuổi bất thường (<0 hoặc >120)")

    return df


def transform_time_feature(
    df: pd.DataFrame,
    time_column: str = 'hrmn'
) -> pd.DataFrame:
    df = df.copy()

    if time_column not in df.columns:
        print(f"[CẢNH BÁO] Không tìm thấy cột '{time_column}' trong DataFrame")
        return df

    df['hour'] = pd.to_numeric(df[time_column], errors='coerce')

    if df['hour'].max() <= 1.0:
        df['hour'] = df['hour'] * 24
        print("[INFO] Đã chuyển đổi thời gian dạng decimal (0-1) sang giờ (0-24)")

    if time_column != 'hour':
        df = df.drop(time_column, axis=1, errors='ignore')

    print(f"[INFO] Đã tạo cột 'hour' từ cột '{time_column}'")
    print(f"[INFO] Giờ trung bình: {df['hour'].mean():.1f}")

    return df


def encode_categorical_features(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()

    if exclude_columns is None:
        exclude_columns = []

    object_columns = df.select_dtypes(include=['object']).columns.tolist()

    columns_to_encode = [col for col in object_columns if col not in exclude_columns]

    if not columns_to_encode:
        print("[INFO] Không có cột nào cần mã hóa")
        return df, {}

    print(f"[INFO] Các cột sẽ được mã hóa: {columns_to_encode}")

    encoders = {}

    for col in columns_to_encode:
        le = LabelEncoder()

        df[col] = df[col].astype(str)

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

        print(f"  ✓ Đã encode cột '{col}' ({len(le.classes_)} giá trị unique)")

    print(f"[INFO] Đã mã hóa {len(columns_to_encode)} cột")

    return df, encoders


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("BẮT ĐẦU FEATURE ENGINEERING")
    print("=" * 60)

    initial_columns = df.columns.tolist()

    print("\n" + "-" * 40)
    print("BƯỚC 1: Tạo đặc trưng tuổi (age)")
    print("-" * 40)
    df = create_age_feature(df)

    print("\n" + "-" * 40)
    print("BƯỚC 2: Chuyển đổi đặc trưng thời gian (hour)")
    print("-" * 40)
    df = transform_time_feature(df)

    print("\n" + "-" * 40)
    print("BƯỚC 3: Mã hóa biến phân loại (Label Encoding)")
    print("-" * 40)
    df, encoders = encode_categorical_features(df)

    print("\n" + "=" * 60)
    print("HOÀN TẤT FEATURE ENGINEERING")
    print("=" * 60)

    final_columns = df.columns.tolist()
    new_columns = [col for col in final_columns if col not in initial_columns]
    removed_columns = [col for col in initial_columns if col not in final_columns]

    print(f"Cột mới thêm: {new_columns}")
    print(f"Cột đã xóa: {removed_columns}")
    print(f"Tổng số cột hiện tại: {len(final_columns)}")

    print("\n[INFO] Kiểu dữ liệu của các cột:")
    print(df.dtypes.value_counts())

    return df


def get_feature_importance_names(
    df: pd.DataFrame,
    target_column: str = 'grav'
) -> List[str]:
    return [col for col in df.columns if col != target_column]
