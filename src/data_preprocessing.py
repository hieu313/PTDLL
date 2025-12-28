import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


def check_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()

    missing_percent = (missing_count / len(df)) * 100

    report = pd.DataFrame({
        'column': missing_count.index,
        'missing_count': missing_count.values,
        'missing_percent': missing_percent.values
    })

    report = report.sort_values('missing_count', ascending=False)
    report = report.reset_index(drop=True)

    total_missing = missing_count.sum()
    rows_with_missing = df.isna().any(axis=1).sum()
    print(f"[INFO] Tổng số ô khuyết thiếu: {total_missing:,}")
    print(f"[INFO] Số dòng có ít nhất 1 ô khuyết thiếu: {rows_with_missing:,} ({rows_with_missing/len(df)*100:.2f}%)")

    return report


def remove_unnecessary_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None
) -> pd.DataFrame:
    df = df.copy()

    if columns_to_drop is None:
        cols_high_missing = ['v2', 'lartpc', 'larrout', 'occutc']

        cols_identifiers = ['Num_Acc', 'id_vehicule', 'num_veh']

        cols_location = ['adr', 'voie', 'pr', 'pr1', 'lat', 'long']

        columns_to_drop = cols_high_missing + cols_identifiers + cols_location

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
    df = df.copy()
    initial_rows = len(df)

    if target_column in df.columns:
        missing_target = df[target_column].isna().sum()
        if missing_target > 0:
            df = df.dropna(subset=[target_column])
            print(f"[INFO] Đã xóa {missing_target} dòng thiếu cột mục tiêu '{target_column}'")

    min_valid_cols = len(df.columns) - max_missing_per_row
    rows_before = len(df)
    df = df.dropna(thresh=min_valid_cols)
    rows_dropped = rows_before - len(df)

    if rows_dropped > 0:
        print(f"[INFO] Đã xóa {rows_dropped} dòng thiếu quá {max_missing_per_row} cột")

    total_dropped = initial_rows - len(df)
    print(f"[INFO] Tổng số dòng đã xóa: {total_dropped} ({total_dropped/initial_rows*100:.2f}%)")
    print(f"[INFO] Số dòng còn lại: {len(df):,}")

    remaining_missing = df.isna().sum()
    cols_with_missing = remaining_missing[remaining_missing > 0]
    if len(cols_with_missing) > 0:
        print(f"[INFO] Vẫn còn {len(cols_with_missing)} cột có dữ liệu khuyết thiếu")
        print("[INFO] XGBoost sẽ tự động xử lý các giá trị này")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_duplicates = df.duplicated().sum()

    if num_duplicates > 0:
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
    df = df.copy()

    if mapping is None:
        mapping = {
            1: 0,
            4: 1,
            2: 2,
            3: 2
        }

    print("[INFO] Phân bố lớp mục tiêu TRƯỚC khi gộp:")
    print(df[target_column].value_counts().sort_index())

    df[target_column] = df[target_column].map(mapping)

    unmapped = df[target_column].isna().sum()
    if unmapped > 0:
        print(f"[CẢNH BÁO] Có {unmapped} giá trị không được ánh xạ (trở thành NaN)")

    print("\n[INFO] Phân bố lớp mục tiêu SAU khi gộp:")
    print(df[target_column].value_counts().sort_index())

    print("\n[INFO] Ý nghĩa các lớp sau khi gộp:")
    print("  0 = Không bị thương")
    print("  1 = Bị thương nhẹ")
    print("  2 = Nghiêm trọng (bao gồm tử vong và bị thương nặng)")

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    initial_shape = df.shape
    print(f"\n[INFO] Kích thước ban đầu: {initial_shape[0]:,} dòng x {initial_shape[1]} cột")

    print("\n" + "-" * 40)
    print("BƯỚC 1: Kiểm tra dữ liệu khuyết thiếu")
    print("-" * 40)
    _ = check_missing_data(df)

    print("\n" + "-" * 40)
    print("BƯỚC 2: Xóa các cột không cần thiết")
    print("-" * 40)
    df = remove_unnecessary_columns(df)

    print("\n" + "-" * 40)
    print("BƯỚC 3: Xử lý giá trị khuyết thiếu")
    print("-" * 40)
    df = handle_missing_values(df)

    print("\n" + "-" * 40)
    print("BƯỚC 4: Xóa dòng trùng lặp")
    print("-" * 40)
    df = remove_duplicates(df)

    print("\n" + "-" * 40)
    print("BƯỚC 5: Ánh xạ lớp mục tiêu")
    print("-" * 40)
    df = map_target_classes(df)

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
    return preprocess_data(df)
