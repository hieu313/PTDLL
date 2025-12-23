# -*- coding: utf-8 -*-
"""
==========================================
Module: data_loader
==========================================
Mô tả:
    Module này chịu trách nhiệm đọc dữ liệu từ các file CSV
    chứa thông tin về tai nạn giao thông năm 2019 tại Pháp.

Các file dữ liệu bao gồm:
    - caracteristiques-2019.csv: Đặc điểm của vụ tai nạn (thời gian, địa điểm, điều kiện...)
    - lieux-2019.csv: Thông tin về địa điểm xảy ra tai nạn
    - usagers-2019.csv: Thông tin về người tham gia giao thông
    - vehicules-2019.csv: Thông tin về phương tiện liên quan

Hàm chính:
    - load_data(): Đọc và trả về 4 DataFrame tương ứng với 4 file CSV
==========================================
"""

import pandas as pd
import warnings
from typing import Tuple, Optional
import os

# Tắt các cảnh báo không cần thiết để output gọn gàng hơn
warnings.filterwarnings('ignore')


def load_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Đọc dữ liệu từ 4 file CSV chứa thông tin tai nạn giao thông.

    Tham số:
        data_dir (str, optional): Đường dẫn đến thư mục chứa dữ liệu.
            - Nếu không cung cấp, sẽ sử dụng đường dẫn mặc định (Kaggle hoặc thư mục data/)

    Trả về:
        Tuple gồm 4 DataFrame:
            - df_caracs: Đặc điểm vụ tai nạn (caracteristiques)
            - df_lieux: Thông tin địa điểm (lieux = places/locations)
            - df_usagers: Thông tin người dùng đường (usagers = road users)
            - df_vehicules: Thông tin phương tiện (vehicules)

    Ngoại lệ:
        FileNotFoundError: Khi không tìm thấy file dữ liệu
        pd.errors.ParserError: Khi file CSV bị lỗi định dạng

    Ví dụ sử dụng:
        >>> df_caracs, df_lieux, df_usagers, df_vehicules = load_data()
        >>> print(f"Số dòng đặc điểm: {len(df_caracs)}")
    """

    # Xác định đường dẫn đến thư mục dữ liệu
    # Ưu tiên: tham số truyền vào > biến môi trường > đường dẫn Kaggle > thư mục data local
    if data_dir is None:
        # Kiểm tra biến môi trường DATA_DIR nếu có
        data_dir = os.environ.get('DATA_DIR', None)

        if data_dir is None:
            # Thử đường dẫn Kaggle (khi chạy trên Kaggle Notebooks)
            kaggle_path = '/kaggle/input/2019-database-of-road-traffic-injuries'
            if os.path.exists(kaggle_path):
                data_dir = kaggle_path
            else:
                # Sử dụng thư mục data/ trong project
                data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    print(f"[INFO] Đang đọc dữ liệu từ: {data_dir}")

    try:
        # ============================================================
        # Đọc file caracteristiques (đặc điểm vụ tai nạn)
        # ============================================================
        # File này chứa thông tin chung về mỗi vụ tai nạn:
        # - Num_Acc: Mã định danh duy nhất của vụ tai nạn
        # - jour/mois/an: Ngày/tháng/năm xảy ra
        # - hrmn: Giờ phút xảy ra
        # - lum: Điều kiện ánh sáng (ban ngày/đêm/hoàng hôn...)
        # - dep/com: Mã tỉnh/xã
        # - agg: Trong hay ngoài khu vực đô thị
        # - int: Loại giao lộ
        # - atm: Điều kiện thời tiết
        # - col: Loại va chạm
        df_caracs = pd.read_csv(
            os.path.join(data_dir, 'caracteristiques-2019.csv'),
            sep=',',           # Phân cách bằng dấu phẩy
            encoding='utf-8'   # Encoding UTF-8 để đọc được tiếng Pháp
        )
        print(f"  ✓ Đã đọc caracteristiques: {len(df_caracs)} dòng, {len(df_caracs.columns)} cột")

        # ============================================================
        # Đọc file lieux (địa điểm)
        # ============================================================
        # File này chứa thông tin về nơi xảy ra tai nạn:
        # - catr: Loại đường (quốc lộ, tỉnh lộ, đường xã...)
        # - voie: Số hiệu đường
        # - circ: Chế độ lưu thông (một chiều, hai chiều...)
        # - nbv: Số làn đường
        # - vosp: Có làn đường dành riêng không
        # - prof: Độ dốc của đường
        # - plan: Hình dạng đường (thẳng, cong...)
        # - surf: Tình trạng mặt đường (khô, ướt, đóng băng...)
        # - infra: Cơ sở hạ tầng (đường hầm, cầu...)
        # - situ: Vị trí tai nạn (trên đường, lề đường...)
        df_lieux = pd.read_csv(
            os.path.join(data_dir, 'lieux-2019.csv'),
            sep=',',
            encoding='utf-8'
        )
        print(f"  ✓ Đã đọc lieux: {len(df_lieux)} dòng, {len(df_lieux.columns)} cột")

        # ============================================================
        # Đọc file usagers (người dùng đường)
        # ============================================================
        # File này chứa thông tin về từng người liên quan đến tai nạn:
        # - place: Vị trí ngồi trên xe
        # - catu: Loại người dùng (lái xe, hành khách, người đi bộ)
        # - grav: Mức độ chấn thương (BIẾN MỤC TIÊU)
        #         1 = Không bị thương, 2 = Tử vong, 3 = Bị thương nặng, 4 = Bị thương nhẹ
        # - sexe: Giới tính
        # - an_nais: Năm sinh
        # - trajet: Mục đích chuyến đi
        # - secu1, secu2, secu3: Thiết bị an toàn đang sử dụng
        # - locp: Vị trí người đi bộ
        # - actp: Hành động của người đi bộ
        # - etatp: Có đi cùng người khác không
        df_usagers = pd.read_csv(
            os.path.join(data_dir, 'usagers-2019.csv'),
            sep=',',
            encoding='utf-8'
        )
        print(f"  ✓ Đã đọc usagers: {len(df_usagers)} dòng, {len(df_usagers.columns)} cột")

        # ============================================================
        # Đọc file vehicules (phương tiện)
        # ============================================================
        # File này chứa thông tin về từng phương tiện trong vụ tai nạn:
        # - num_veh: Số thứ tự phương tiện trong vụ tai nạn
        # - senc: Hướng đi (tăng/giảm)
        # - catv: Loại phương tiện (xe máy, ô tô, xe tải, xe buýt...)
        # - obs: Vật cản cố định va chạm phải
        # - obsm: Vật cản di động va chạm phải
        # - choc: Điểm va chạm ban đầu
        # - manv: Thao tác chính trước tai nạn
        # - motor: Loại động cơ
        # - occutc: Số người ngồi trong xe (thường thiếu dữ liệu)
        df_vehicules = pd.read_csv(
            os.path.join(data_dir, 'vehicules-2019.csv'),
            sep=','  # Không cần encoding vì file này ít ký tự đặc biệt hơn
        )
        print(f"  ✓ Đã đọc vehicules: {len(df_vehicules)} dòng, {len(df_vehicules.columns)} cột")

        print("[INFO] Đã load tất cả file thành công!")

        return df_caracs, df_lieux, df_usagers, df_vehicules

    except FileNotFoundError as e:
        # Lỗi khi không tìm thấy file
        print(f"[LỖI] Không tìm thấy file dữ liệu: {e}")
        print(f"[GỢI Ý] Hãy kiểm tra đường dẫn: {data_dir}")
        print("[GỢI Ý] Hoặc đặt biến môi trường DATA_DIR trỏ đến thư mục chứa dữ liệu")
        raise

    except pd.errors.ParserError as e:
        # Lỗi khi file CSV bị hỏng hoặc sai định dạng
        print(f"[LỖI] File CSV bị lỗi định dạng: {e}")
        raise

    except Exception as e:
        # Các lỗi khác
        print(f"[LỖI] Lỗi không xác định khi đọc dữ liệu: {e}")
        raise


def merge_dataframes(
    df_caracs: pd.DataFrame,
    df_lieux: pd.DataFrame,
    df_usagers: pd.DataFrame,
    df_vehicules: pd.DataFrame
) -> pd.DataFrame:
    """
    Gộp 4 DataFrame thành 1 DataFrame thống nhất.

    Quy trình gộp:
        1. Gộp df_caracs với df_lieux theo Num_Acc (đặc điểm + địa điểm)
        2. Gộp df_usagers với df_vehicules theo Num_Acc, num_veh, id_vehicule
        3. Gộp kết quả của bước 1 và 2 theo Num_Acc

    Tham số:
        df_caracs: DataFrame chứa đặc điểm vụ tai nạn
        df_lieux: DataFrame chứa thông tin địa điểm
        df_usagers: DataFrame chứa thông tin người dùng đường
        df_vehicules: DataFrame chứa thông tin phương tiện

    Trả về:
        DataFrame đã được gộp chứa tất cả thông tin

    Lưu ý:
        - Sử dụng left join để giữ lại tất cả bản ghi từ bảng bên trái
        - Cột Num_Acc được chuyển sang string để tránh lỗi khi gộp
    """

    print("[INFO] Bắt đầu gộp các DataFrame...")

    # ============================================================
    # Bước 1: Chuyển cột Num_Acc sang kiểu string
    # ============================================================
    # Lý do: Đảm bảo các giá trị có thể so sánh được khi gộp
    # Tránh lỗi khi một số giá trị là số, một số là chuỗi
    df_caracs = df_caracs.copy()
    df_lieux = df_lieux.copy()
    df_usagers = df_usagers.copy()
    df_vehicules = df_vehicules.copy()

    df_caracs['Num_Acc'] = df_caracs['Num_Acc'].astype(str)
    df_lieux['Num_Acc'] = df_lieux['Num_Acc'].astype(str)
    df_usagers['Num_Acc'] = df_usagers['Num_Acc'].astype(str)
    df_vehicules['Num_Acc'] = df_vehicules['Num_Acc'].astype(str)

    # ============================================================
    # Bước 2: Gộp đặc điểm với địa điểm
    # ============================================================
    # Mỗi vụ tai nạn (Num_Acc) có 1 dòng đặc điểm và 1 dòng địa điểm
    # => Quan hệ 1-1, gộp theo cột Num_Acc
    df_1 = pd.merge(
        df_caracs,
        df_lieux,
        on='Num_Acc',     # Cột khóa để gộp
        how='left'        # Left join: giữ tất cả từ df_caracs
    )
    print(f"  ✓ Sau gộp đặc điểm + địa điểm: {len(df_1)} dòng")

    # ============================================================
    # Bước 3: Gộp người dùng với phương tiện
    # ============================================================
    # Mỗi người (usager) ngồi trên 1 xe (vehicule)
    # => Gộp theo cả Num_Acc, num_veh (số xe trong vụ tai nạn), và id_vehicule
    df_2 = pd.merge(
        df_usagers,
        df_vehicules,
        on=['Num_Acc', 'num_veh', 'id_vehicule'],  # 3 cột khóa
        how='left'
    )
    print(f"  ✓ Sau gộp người dùng + phương tiện: {len(df_2)} dòng")

    # ============================================================
    # Bước 4: Gộp tất cả lại với nhau
    # ============================================================
    # Kết quả cuối cùng: mỗi dòng là 1 người trong 1 vụ tai nạn
    # với đầy đủ thông tin về tai nạn, địa điểm, xe và người đó
    df = pd.merge(
        df_1,
        df_2,
        on='Num_Acc',
        how='left'
    )
    print(f"  ✓ Sau gộp tất cả: {len(df)} dòng, {len(df.columns)} cột")
    print("[INFO] Gộp DataFrame hoàn tất!")

    return df
