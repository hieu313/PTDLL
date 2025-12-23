# -*- coding: utf-8 -*-
"""
==========================================
CHƯƠNG TRÌNH CHÍNH - PHÂN TÍCH TAI NẠN GIAO THÔNG
==========================================

Mô tả:
    Chương trình này thực hiện phân tích dữ liệu tai nạn giao thông năm 2019
    tại Pháp và huấn luyện mô hình XGBoost để dự đoán mức độ nghiêm trọng.

Quy trình:
    1. Đọc dữ liệu từ 4 file CSV
    2. Gộp các bảng thành 1 DataFrame thống nhất
    3. Tiền xử lý dữ liệu (xóa cột không cần, xử lý missing, xóa trùng lặp)
    4. Feature engineering (tạo đặc trưng tuổi, thời gian, mã hóa biến)
    5. Trực quan hóa dữ liệu
    6. Huấn luyện mô hình XGBoost
    7. Đánh giá mô hình
    8. Lưu mô hình

Cách sử dụng:
    python main_new.py

Yêu cầu:
    - Python 3.7+
    - pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib

Tác giả: PTDLL Team
==========================================
"""

# ============================================================
# IMPORT CÁC THƯ VIỆN
# ============================================================

import warnings
# Tắt các cảnh báo không cần thiết để output gọn gàng
warnings.filterwarnings('ignore')

# Import các module đã tạo
from src.data_loader import load_data, merge_dataframes
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features, get_feature_importance_names
from src.visualization import create_visualizations, plot_confusion_matrix, plot_feature_importance
from src.model import prepare_data, train_model, evaluate_model, save_model


def main():
    """
    Hàm chính điều phối toàn bộ quy trình phân tích và huấn luyện mô hình.

    Các bước thực hiện:
        1. Đọc dữ liệu
        2. Gộp dữ liệu
        3. Tiền xử lý
        4. Feature engineering
        5. Trực quan hóa
        6. Chuẩn bị train/test
        7. Huấn luyện mô hình
        8. Đánh giá
        9. Lưu mô hình
    """
    print("=" * 60)
    print("CHƯƠNG TRÌNH PHÂN TÍCH TAI NẠN GIAO THÔNG")
    print("Dữ liệu: Pháp năm 2019")
    print("=" * 60)

    # ============================================================
    # BƯỚC 1: ĐỌC DỮ LIỆU
    # ============================================================
    print("\n" + "=" * 60)
    print("BƯỚC 1: ĐỌC DỮ LIỆU")
    print("=" * 60)

    try:
        # Đọc 4 file CSV
        # data_dir=None sẽ tự động detect đường dẫn phù hợp
        df_caracs, df_lieux, df_usagers, df_vehicules = load_data()
    except FileNotFoundError:
        print("\n[LỖI] Không tìm thấy dữ liệu!")
        print("[GỢI Ý] Hãy đảm bảo các file CSV nằm trong thư mục 'data/'")
        print("[GỢI Ý] Hoặc đặt biến môi trường DATA_DIR trỏ đến thư mục chứa dữ liệu")
        return

    # ============================================================
    # BƯỚC 2: GỘP DỮ LIỆU
    # ============================================================
    print("\n" + "=" * 60)
    print("BƯỚC 2: GỘP DỮ LIỆU")
    print("=" * 60)

    # Gộp 4 DataFrame thành 1
    df = merge_dataframes(df_caracs, df_lieux, df_usagers, df_vehicules)

    print(f"\n[INFO] Kích thước dữ liệu sau khi gộp: {df.shape[0]:,} dòng x {df.shape[1]} cột")

    # ============================================================
    # BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU
    # ============================================================
    print("\n" + "=" * 60)
    print("BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    # Thực hiện tiền xử lý:
    # - Xóa cột không cần thiết
    # - Xử lý missing values
    # - Xóa dòng trùng lặp
    # - Gộp lớp mục tiêu
    df = preprocess_data(df)

    # ============================================================
    # BƯỚC 4: FEATURE ENGINEERING
    # ============================================================
    print("\n" + "=" * 60)
    print("BƯỚC 4: FEATURE ENGINEERING")
    print("=" * 60)

    # Tạo và chuyển đổi các đặc trưng:
    # - Tạo cột tuổi từ năm sinh
    # - Chuyển đổi cột thời gian
    # - Mã hóa biến phân loại
    df = engineer_features(df)

    # In thông tin tổng quan về dữ liệu
    print("\n[INFO] Thống kê mô tả dữ liệu:")
    print(df.describe())

    # ============================================================
    # BƯỚC 5: TRỰC QUAN HÓA DỮ LIỆU
    # ============================================================
    print("\n" + "=" * 60)
    print("BƯỚC 5: TRỰC QUAN HÓA DỮ LIỆU")
    print("=" * 60)

    # Tạo các biểu đồ và lưu vào file
    # Có thể comment dòng này nếu chạy trên server không có GUI
    try:
        create_visualizations(df, save_path='output/data_visualization.png')
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể tạo biểu đồ: {e}")
        print("[INFO] Tiếp tục huấn luyện mô hình...")

    # ============================================================
    # BƯỚC 6: CHUẨN BỊ DỮ LIỆU TRAIN/TEST
    # ============================================================
    print("\n" + "=" * 60)
    print("BƯỚC 6: CHUẨN BỊ DỮ LIỆU TRAIN/TEST")
    print("=" * 60)

    # Chia dữ liệu thành train (80%) và test (20%)
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Lấy danh sách tên các đặc trưng
    feature_names = get_feature_importance_names(df)
    print(f"[INFO] Số đặc trưng: {len(feature_names)}")

    # ============================================================
    # BƯỚC 7: HUẤN LUYỆN MÔ HÌNH
    # ============================================================
    print("\n" + "=" * 60)
    print("BƯỚC 7: HUẤN LUYỆN MÔ HÌNH XGBOOST")
    print("=" * 60)

    # Huấn luyện mô hình XGBoost với sample weight để cân bằng các lớp
    model = train_model(X_train, y_train, use_sample_weight=True)

    # ============================================================
    # BƯỚC 8: ĐÁNH GIÁ MÔ HÌNH
    # ============================================================
    print("\n" + "=" * 60)
    print("BƯỚC 8: ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)

    # Đánh giá trên tập test
    results = evaluate_model(model, X_test, y_test)

    # Vẽ confusion matrix
    try:
        plot_confusion_matrix(
            y_test, results['y_pred'],
            save_path='output/confusion_matrix.png'
        )
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể vẽ confusion matrix: {e}")

    # Vẽ feature importance
    try:
        plot_feature_importance(
            model, feature_names,
            save_path='output/feature_importance.png'
        )
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể vẽ feature importance: {e}")

    # ============================================================
    # BƯỚC 9: LƯU MÔ HÌNH
    # ============================================================
    print("\n" + "=" * 60)
    print("BƯỚC 9: LƯU MÔ HÌNH")
    print("=" * 60)

    # Lưu mô hình vào file
    save_model(model, 'models/model_xgboost.pkl')

    # ============================================================
    # KẾT THÚC
    # ============================================================
    print("\n" + "=" * 60)
    print("HOÀN TẤT!")
    print("=" * 60)
    print("\nTóm tắt kết quả:")
    print(f"  - Độ chính xác (Accuracy): {results['accuracy']:.4f}")
    print(f"  - F1-Score (Macro): {results['f1']:.4f}")
    print(f"  - Mô hình đã lưu tại: models/model_xgboost.pkl")
    print("\nCách sử dụng mô hình:")
    print("  from src.model import load_model, predict_single")
    print("  model = load_model('models/model_xgboost.pkl')")
    print("  prediction, probabilities = predict_single(model, features, feature_names)")


if __name__ == '__main__':
    # Chạy hàm main khi file được thực thi trực tiếp
    main()
