

import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_data, merge_dataframes
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features, get_feature_importance_names
from src.visualization import create_visualizations, plot_confusion_matrix, plot_feature_importance
from src.model import prepare_data, train_model, evaluate_model, save_model


def main():

    print("=" * 60)
    print("CHƯƠNG TRÌNH PHÂN TÍCH TAI NẠN GIAO THÔNG")
    print("Dữ liệu: Pháp năm 2019")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("BƯỚC 1: ĐỌC DỮ LIỆU")
    print("=" * 60)

    try:
        df_caracs, df_lieux, df_usagers, df_vehicules = load_data()
    except FileNotFoundError:
        print("\n[LỖI] Không tìm thấy dữ liệu!")
        print("[GỢI Ý] Hãy đảm bảo các file CSV nằm trong thư mục 'data/'")
        print("[GỢI Ý] Hoặc đặt biến môi trường DATA_DIR trỏ đến thư mục chứa dữ liệu")
        return

    print("\n" + "=" * 60)
    print("BƯỚC 2: GỘP DỮ LIỆU")
    print("=" * 60)

    df = merge_dataframes(df_caracs, df_lieux, df_usagers, df_vehicules)

    print(f"\n[INFO] Kích thước dữ liệu sau khi gộp: {df.shape[0]:,} dòng x {df.shape[1]} cột")

    print("\n" + "=" * 60)
    print("BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    df = preprocess_data(df)

    print("\n" + "=" * 60)
    print("BƯỚC 4: FEATURE ENGINEERING")
    print("=" * 60)

    df = engineer_features(df)

    print("\n[INFO] Thống kê mô tả dữ liệu:")
    print(df.describe())

    print("\n" + "=" * 60)
    print("BƯỚC 5: TRỰC QUAN HÓA DỮ LIỆU")
    print("=" * 60)

    try:
        create_visualizations(df, save_path='output/data_visualization.png')
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể tạo biểu đồ: {e}")
        print("[INFO] Tiếp tục huấn luyện mô hình...")

    print("\n" + "=" * 60)
    print("BƯỚC 6: CHUẨN BỊ DỮ LIỆU TRAIN/TEST")
    print("=" * 60)

    X_train, X_test, y_train, y_test = prepare_data(df)

    feature_names = get_feature_importance_names(df)
    print(f"[INFO] Số đặc trưng: {len(feature_names)}")

    print("\n" + "=" * 60)
    print("BƯỚC 7: HUẤN LUYỆN MÔ HÌNH XGBOOST")
    print("=" * 60)

    model = train_model(X_train, y_train, use_sample_weight=True)

    print("\n" + "=" * 60)
    print("BƯỚC 8: ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)

    results = evaluate_model(model, X_test, y_test)

    try:
        plot_confusion_matrix(
            y_test, results['y_pred'],
            save_path='output/confusion_matrix.png'
        )
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể vẽ confusion matrix: {e}")

    try:
        plot_feature_importance(
            model, feature_names,
            save_path='output/feature_importance.png'
        )
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể vẽ feature importance: {e}")

    print("\n" + "=" * 60)
    print("BƯỚC 9: LƯU MÔ HÌNH")
    print("=" * 60)

    save_model(model, 'models/model_xgboost.pkl')

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
    main()
