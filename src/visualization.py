# -*- coding: utf-8 -*-
"""
==========================================
Module: visualization
==========================================
Mô tả:
    Module này chịu trách nhiệm tạo các biểu đồ trực quan hóa dữ liệu
    để hiểu rõ hơn về phân bố và mối quan hệ giữa các biến.

Các biểu đồ bao gồm:
    1. Phân bố mức độ nghiêm trọng (Gravity Distribution)
    2. Phân bố tai nạn theo giờ trong ngày
    3. Mối quan hệ tuổi và mức độ nghiêm trọng
    4. Top 10 loại phương tiện gặp nạn

Các hàm chính:
    - plot_gravity_distribution(): Biểu đồ phân bố mức độ nghiêm trọng
    - plot_hourly_distribution(): Biểu đồ phân bố theo giờ
    - plot_age_vs_gravity(): Biểu đồ boxplot tuổi theo mức độ
    - plot_vehicle_types(): Biểu đồ top loại phương tiện
    - create_visualizations(): Tạo tất cả biểu đồ trong 1 hình
==========================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple


# Thiết lập style mặc định cho matplotlib
# Style 'seaborn' tạo biểu đồ đẹp và chuyên nghiệp hơn
plt.style.use('seaborn-v0_8-whitegrid')

# Thiết lập font hỗ trợ tiếng Việt (nếu có)
plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_gravity_distribution(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    palette: str = 'viridis'
) -> plt.Axes:
    """
    Vẽ biểu đồ cột thể hiện phân bố mức độ nghiêm trọng của tai nạn.

    Tham số:
        df: DataFrame chứa cột 'grav'
        ax: Matplotlib Axes để vẽ (nếu không cung cấp, tạo mới)
        palette: Bảng màu Seaborn (mặc định: 'viridis')

    Trả về:
        Matplotlib Axes chứa biểu đồ

    Ý nghĩa:
        Biểu đồ này cho thấy tỷ lệ các loại chấn thương:
        - Giúp nhận biết dataset có cân bằng hay không
        - Nếu mất cân bằng, cần dùng sample weight khi huấn luyện
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Vẽ biểu đồ đếm (count plot)
    # x='grav': trục x là mức độ nghiêm trọng
    # data=df: dữ liệu từ DataFrame
    # palette: bảng màu
    sns.countplot(x='grav', data=df, ax=ax, palette=palette)

    # Thiết lập tiêu đề và nhãn
    ax.set_title('Phân bố mức độ nghiêm trọng (Gravity)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mức độ nghiêm trọng', fontsize=12)
    ax.set_ylabel('Số lượng', fontsize=12)

    # Thêm nhãn cho các cột
    # 0: Không bị thương, 1: Nhẹ, 2: Nghiêm trọng
    ax.set_xticklabels(['0: Không thương', '1: Nhẹ', '2: Nghiêm trọng'])

    # Thêm số liệu trên đầu mỗi cột
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{int(height):,}',  # Format số với dấu phẩy phân cách hàng nghìn
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    return ax


def plot_hourly_distribution(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    color: str = 'orange'
) -> plt.Axes:
    """
    Vẽ biểu đồ histogram thể hiện phân bố tai nạn theo giờ trong ngày.

    Tham số:
        df: DataFrame chứa cột 'hour'
        ax: Matplotlib Axes để vẽ
        color: Màu của biểu đồ

    Trả về:
        Matplotlib Axes chứa biểu đồ

    Ý nghĩa:
        - Xác định giờ cao điểm xảy ra tai nạn
        - Giúp cơ quan chức năng tập trung nguồn lực vào đúng thời điểm
        - Nhận biết pattern: ban ngày nhiều tai nạn nhưng ban đêm nghiêm trọng hơn
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Vẽ histogram với đường KDE (Kernel Density Estimation)
    # bins=24: chia thành 24 cột tương ứng 24 giờ
    # kde=True: thêm đường cong ước lượng mật độ
    sns.histplot(
        data=df,
        x='hour',
        bins=24,
        kde=True,
        ax=ax,
        color=color
    )

    # Thiết lập tiêu đề và nhãn
    ax.set_title('Phân bố tai nạn theo giờ trong ngày', fontsize=14, fontweight='bold')
    ax.set_xlabel('Giờ (0-24)', fontsize=12)
    ax.set_ylabel('Số lượng tai nạn', fontsize=12)

    # Đặt tick marks cho mỗi 3 giờ
    ax.set_xticks(range(0, 25, 3))

    return ax


def plot_age_vs_gravity(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Vẽ biểu đồ boxplot thể hiện mối quan hệ giữa tuổi và mức độ nghiêm trọng.

    Tham số:
        df: DataFrame chứa cột 'age' và 'grav'
        ax: Matplotlib Axes để vẽ

    Trả về:
        Matplotlib Axes chứa biểu đồ

    Ý nghĩa:
        Boxplot cho thấy:
        - Median (đường ngang trong hộp): tuổi trung vị
        - IQR (hộp): 50% dữ liệu ở giữa (từ Q1 đến Q3)
        - Whiskers (râu): dữ liệu không phải outlier
        - Điểm ngoài: outliers

        So sánh giữa các mức độ nghiêm trọng:
        - Người già có xu hướng bị thương nặng hơn không?
        - Thanh niên liều lĩnh hơn hay an toàn hơn?
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Lọc bỏ các giá trị tuổi bất thường để biểu đồ đẹp hơn
    df_filtered = df[(df['age'] >= 0) & (df['age'] <= 100)]

    # Vẽ boxplot
    # x='grav': trục x là mức độ nghiêm trọng
    # y='age': trục y là tuổi
    sns.boxplot(x='grav', y='age', data=df_filtered, ax=ax)

    # Thiết lập tiêu đề và nhãn
    ax.set_title('Độ tuổi theo mức độ nghiêm trọng', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mức độ nghiêm trọng', fontsize=12)
    ax.set_ylabel('Tuổi', fontsize=12)

    # Thêm nhãn cho các lớp
    ax.set_xticklabels(['0: Không thương', '1: Nhẹ', '2: Nghiêm trọng'])

    return ax


def plot_vehicle_types(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    top_n: int = 10
) -> plt.Axes:
    """
    Vẽ biểu đồ thanh ngang thể hiện top N loại phương tiện gặp nạn nhiều nhất.

    Tham số:
        df: DataFrame chứa cột 'catv' (category vehicle)
        ax: Matplotlib Axes để vẽ
        top_n: Số lượng loại phương tiện hiển thị (mặc định: 10)

    Trả về:
        Matplotlib Axes chứa biểu đồ

    Lưu ý về cột 'catv':
        Sau khi Label Encoding, 'catv' là số nguyên.
        Các giá trị gốc có thể là:
        - 01: Xe đạp
        - 02: Xe máy dưới 50cc
        - 03: Xe máy trên 50cc
        - 07: Xe ô tô con
        - 10: Xe tải
        - 13: Xe buýt
        - ...
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Lấy top N loại phương tiện phổ biến nhất
    top_vehicles = df['catv'].value_counts().index[:top_n]

    # Lọc dữ liệu chỉ chứa top N loại
    df_filtered = df[df['catv'].isin(top_vehicles)]

    # Vẽ biểu đồ thanh ngang
    # y='catv': loại phương tiện trên trục y (ngang)
    # order=top_vehicles: sắp xếp theo thứ tự đã xác định
    sns.countplot(
        y='catv',
        data=df_filtered,
        ax=ax,
        order=top_vehicles
    )

    # Thiết lập tiêu đề và nhãn
    ax.set_title(f'Top {top_n} loại phương tiện gặp nạn', fontsize=14, fontweight='bold')
    ax.set_xlabel('Số lượng', fontsize=12)
    ax.set_ylabel('Mã loại phương tiện (catv)', fontsize=12)

    return ax


def create_visualizations(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Tạo tất cả các biểu đồ trong một hình gồm 4 ô (2x2).

    Tham số:
        df: DataFrame đã qua tiền xử lý và feature engineering
        save_path: Đường dẫn lưu hình (nếu cung cấp, sẽ lưu file)
        figsize: Kích thước hình (width, height) tính bằng inch

    Trả về:
        Matplotlib Figure chứa tất cả các biểu đồ

    Cấu trúc hình:
        +-------------------+-------------------+
        |  Phân bố mức độ   | Phân bố theo giờ  |
        |   nghiêm trọng    |     trong ngày    |
        +-------------------+-------------------+
        |  Tuổi theo mức    | Top 10 loại       |
        |   độ nghiêm trọng |    phương tiện    |
        +-------------------+-------------------+
    """
    print("[INFO] Đang tạo các biểu đồ trực quan hóa...")

    # Tạo figure với 4 subplots (2 hàng x 2 cột)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Vẽ từng biểu đồ
    # axes[0, 0]: Góc trên bên trái
    plot_gravity_distribution(df, ax=axes[0, 0])

    # axes[0, 1]: Góc trên bên phải
    plot_hourly_distribution(df, ax=axes[0, 1])

    # axes[1, 0]: Góc dưới bên trái
    plot_age_vs_gravity(df, ax=axes[1, 0])

    # axes[1, 1]: Góc dưới bên phải
    plot_vehicle_types(df, ax=axes[1, 1])

    # Điều chỉnh khoảng cách giữa các subplot
    plt.tight_layout()

    # Lưu hình nếu có đường dẫn
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Đã lưu biểu đồ vào: {save_path}")

    # Hiển thị hình
    plt.show()

    print("[INFO] Hoàn tất tạo biểu đồ!")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) dưới dạng heatmap.

    Tham số:
        y_true: Nhãn thực tế
        y_pred: Nhãn dự đoán
        class_names: Tên các lớp để hiển thị
        save_path: Đường dẫn lưu hình

    Trả về:
        Matplotlib Figure chứa biểu đồ

    Ý nghĩa của Confusion Matrix:
        - Đường chéo chính: Số dự đoán đúng
        - Các ô khác: Số dự đoán sai

        Ví dụ với 3 lớp (0, 1, 2):
                    Dự đoán
                    0    1    2
        Thực  0   [50]   5    2     <- 50 đúng, 7 sai
        tế    1    3   [40]   8     <- 40 đúng, 11 sai
              2    1    4   [35]    <- 35 đúng, 5 sai
    """
    from sklearn.metrics import confusion_matrix

    if class_names is None:
        class_names = ['Không thương', 'Nhẹ', 'Nghiêm trọng']

    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Tạo figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Vẽ heatmap
    # annot=True: hiển thị số trong mỗi ô
    # fmt='d': format số nguyên
    # cmap='Blues': bảng màu xanh dương
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dự đoán', fontsize=12)
    ax.set_ylabel('Thực tế', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Đã lưu confusion matrix vào: {save_path}")

    return fig


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Vẽ biểu đồ mức độ quan trọng của các đặc trưng (Feature Importance).

    Tham số:
        model: Mô hình XGBoost đã huấn luyện
        feature_names: Danh sách tên các đặc trưng
        top_n: Số đặc trưng hiển thị (mặc định: 20)
        save_path: Đường dẫn lưu hình

    Trả về:
        Matplotlib Figure chứa biểu đồ

    Ý nghĩa:
        Feature Importance cho biết mỗi đặc trưng đóng góp bao nhiêu
        vào khả năng dự đoán của mô hình:
        - Giá trị cao: đặc trưng quan trọng
        - Giá trị thấp: đặc trưng ít quan trọng
    """
    # Lấy feature importance từ mô hình
    importance = model.feature_importances_

    # Tạo DataFrame để sắp xếp
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    # Sắp xếp theo độ quan trọng giảm dần và lấy top N
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Vẽ biểu đồ thanh ngang
    sns.barplot(
        x='importance',
        y='feature',
        data=importance_df,
        ax=ax,
        palette='viridis'
    )

    ax.set_title(f'Top {top_n} đặc trưng quan trọng nhất', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mức độ quan trọng', fontsize=12)
    ax.set_ylabel('Đặc trưng', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Đã lưu feature importance vào: {save_path}")

    return fig
