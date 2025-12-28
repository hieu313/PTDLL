

import pandas as pd
import warnings
from typing import Tuple, Optional
import os

warnings.filterwarnings('ignore')


def load_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', None)

        if data_dir is None:
            kaggle_path = '/kaggle/input/2019-database-of-road-traffic-injuries'
            if os.path.exists(kaggle_path):
                data_dir = kaggle_path
            else:
                data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    print(f"[INFO] Đang đọc dữ liệu từ: {data_dir}")

    try:
        df_caracs = pd.read_csv(
            os.path.join(data_dir, 'caracteristiques-2019.csv'),
            sep=',',
            encoding='utf-8'
        )
        print(f"  ✓ Đã đọc caracteristiques: {len(df_caracs)} dòng, {len(df_caracs.columns)} cột")

        df_lieux = pd.read_csv(
            os.path.join(data_dir, 'lieux-2019.csv'),
            sep=',',
            encoding='utf-8'
        )
        print(f"  ✓ Đã đọc lieux: {len(df_lieux)} dòng, {len(df_lieux.columns)} cột")

        df_usagers = pd.read_csv(
            os.path.join(data_dir, 'usagers-2019.csv'),
            sep=',',
            encoding='utf-8'
        )
        print(f"  ✓ Đã đọc usagers: {len(df_usagers)} dòng, {len(df_usagers.columns)} cột")

        df_vehicules = pd.read_csv(
            os.path.join(data_dir, 'vehicules-2019.csv'),
            sep=','
        )
        print(f"  ✓ Đã đọc vehicules: {len(df_vehicules)} dòng, {len(df_vehicules.columns)} cột")

        print("[INFO] Đã load tất cả file thành công!")

        return df_caracs, df_lieux, df_usagers, df_vehicules

    except FileNotFoundError as e:
        print(f"[LỖI] Không tìm thấy file dữ liệu: {e}")
        print(f"[GỢI Ý] Hãy kiểm tra đường dẫn: {data_dir}")
        print("[GỢI Ý] Hoặc đặt biến môi trường DATA_DIR trỏ đến thư mục chứa dữ liệu")
        raise

    except pd.errors.ParserError as e:
        print(f"[LỖI] File CSV bị lỗi định dạng: {e}")
        raise

    except Exception as e:
        print(f"[LỖI] Lỗi không xác định khi đọc dữ liệu: {e}")
        raise


def merge_dataframes(
    df_caracs: pd.DataFrame,
    df_lieux: pd.DataFrame,
    df_usagers: pd.DataFrame,
    df_vehicules: pd.DataFrame
) -> pd.DataFrame:
    print("[INFO] Bắt đầu gộp các DataFrame...")

    df_caracs = df_caracs.copy()
    df_lieux = df_lieux.copy()
    df_usagers = df_usagers.copy()
    df_vehicules = df_vehicules.copy()

    df_caracs['Num_Acc'] = df_caracs['Num_Acc'].astype(str)
    df_lieux['Num_Acc'] = df_lieux['Num_Acc'].astype(str)
    df_usagers['Num_Acc'] = df_usagers['Num_Acc'].astype(str)
    df_vehicules['Num_Acc'] = df_vehicules['Num_Acc'].astype(str)

    df_1 = pd.merge(
        df_caracs,
        df_lieux,
        on='Num_Acc',
        how='left'
    )
    print(f"  ✓ Sau gộp đặc điểm + địa điểm: {len(df_1)} dòng")

    df_2 = pd.merge(
        df_usagers,
        df_vehicules,
        on=['Num_Acc', 'num_veh', 'id_vehicule'],
        how='left'
    )
    print(f"  ✓ Sau gộp người dùng + phương tiện: {len(df_2)} dòng")

    df = pd.merge(
        df_1,
        df_2,
        on='Num_Acc',
        how='left'
    )
    print(f"  ✓ Sau gộp tất cả: {len(df)} dòng, {len(df.columns)} cột")
    print("[INFO] Gộp DataFrame hoàn tất!")

    return df
