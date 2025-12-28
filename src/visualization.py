import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple


plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_gravity_distribution(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    palette: str = 'viridis'
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.countplot(x='grav', data=df, ax=ax, palette=palette)

    ax.set_title('Phân bố mức độ nghiêm trọng (Gravity)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mức độ nghiêm trọng', fontsize=12)
    ax.set_ylabel('Số lượng', fontsize=12)

    ax.set_xticklabels(['0: Không thương', '1: Nhẹ', '2: Nghiêm trọng'])

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{int(height):,}',
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.histplot(
        data=df,
        x='hour',
        bins=24,
        kde=True,
        ax=ax,
        color=color
    )

    ax.set_title('Phân bố tai nạn theo giờ trong ngày', fontsize=14, fontweight='bold')
    ax.set_xlabel('Giờ (0-24)', fontsize=12)
    ax.set_ylabel('Số lượng tai nạn', fontsize=12)

    ax.set_xticks(range(0, 25, 3))

    return ax


def plot_age_vs_gravity(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    df_filtered = df[(df['age'] >= 0) & (df['age'] <= 100)]

    sns.boxplot(x='grav', y='age', data=df_filtered, ax=ax)

    ax.set_title('Độ tuổi theo mức độ nghiêm trọng', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mức độ nghiêm trọng', fontsize=12)
    ax.set_ylabel('Tuổi', fontsize=12)

    ax.set_xticklabels(['0: Không thương', '1: Nhẹ', '2: Nghiêm trọng'])

    return ax


def plot_vehicle_types(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    top_n: int = 10
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    top_vehicles = df['catv'].value_counts().index[:top_n]

    df_filtered = df[df['catv'].isin(top_vehicles)]

    sns.countplot(
        y='catv',
        data=df_filtered,
        ax=ax,
        order=top_vehicles
    )

    ax.set_title(f'Top {top_n} loại phương tiện gặp nạn', fontsize=14, fontweight='bold')
    ax.set_xlabel('Số lượng', fontsize=12)
    ax.set_ylabel('Mã loại phương tiện (catv)', fontsize=12)

    return ax


def create_visualizations(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:

    print("[INFO] Đang tạo các biểu đồ trực quan hóa...")

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_gravity_distribution(df, ax=axes[0, 0])

    plot_hourly_distribution(df, ax=axes[0, 1])

    plot_age_vs_gravity(df, ax=axes[1, 0])

    plot_vehicle_types(df, ax=axes[1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Đã lưu biểu đồ vào: {save_path}")

    plt.show()

    print("[INFO] Hoàn tất tạo biểu đồ!")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    save_path: Optional[str] = None
) -> plt.Figure:

    from sklearn.metrics import confusion_matrix

    if class_names is None:
        class_names = ['Không thương', 'Nhẹ', 'Nghiêm trọng']

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

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

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

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
