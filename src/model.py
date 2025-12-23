# -*- coding: utf-8 -*-
"""
==========================================
Module: model
==========================================
Mô tả:
    Module này chịu trách nhiệm huấn luyện và đánh giá mô hình XGBoost
    cho bài toán phân loại mức độ nghiêm trọng của tai nạn giao thông.

Các hàm chính:
    - prepare_data(): Chuẩn bị dữ liệu train/test
    - train_model(): Huấn luyện mô hình XGBoost
    - evaluate_model(): Đánh giá hiệu suất mô hình
    - save_model(): Lưu mô hình đã huấn luyện
    - load_model(): Tải mô hình từ file

Mô hình sử dụng:
    XGBoost (eXtreme Gradient Boosting)
    - Là thuật toán ensemble learning dựa trên gradient boosting
    - Rất hiệu quả với dữ liệu dạng bảng (tabular data)
    - Có thể xử lý tốt missing values
    - Hỗ trợ regularization để tránh overfitting
==========================================
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from typing import Tuple, Optional, Dict, Any
import os


def prepare_data(
    df: pd.DataFrame,
    target_column: str = 'grav',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chuẩn bị dữ liệu cho việc huấn luyện: tách X/y và train/test.

    Tham số:
        df: DataFrame đã qua tiền xử lý và feature engineering
        target_column: Tên cột mục tiêu (mặc định: 'grav')
        test_size: Tỷ lệ dữ liệu test (mặc định: 0.2 = 20%)
        random_state: Seed để reproducibility (mặc định: 42)

    Trả về:
        Tuple gồm 4 phần:
            - X_train: Dữ liệu đặc trưng cho training
            - X_test: Dữ liệu đặc trưng cho testing
            - y_train: Nhãn cho training
            - y_test: Nhãn cho testing

    Lưu ý:
        - stratify=y: Đảm bảo tỷ lệ các lớp được giữ nguyên trong train và test
        - Quan trọng vì dữ liệu mất cân bằng (imbalanced)
    """
    print("[INFO] Đang chuẩn bị dữ liệu train/test...")

    # Tách biến đầu vào (X) và biến mục tiêu (y)
    # X chứa tất cả các cột trừ cột mục tiêu
    # y chứa cột mục tiêu
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    print(f"[INFO] Số đặc trưng (features): {X.shape[1]}")
    print(f"[INFO] Phân bố lớp mục tiêu:")
    print(y.value_counts())

    # Chia dữ liệu thành tập train và test
    # stratify=y: Đảm bảo tỷ lệ các lớp trong train và test giống nhau
    # Ví dụ: nếu dữ liệu gốc có 60% lớp 0, 30% lớp 1, 10% lớp 2
    #        thì train và test cũng sẽ có tỷ lệ tương tự
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Stratified sampling để cân bằng các lớp
    )

    print(f"\n[INFO] Kích thước dữ liệu:")
    print(f"  - Train: {X_train.shape[0]:,} mẫu")
    print(f"  - Test:  {X_test.shape[0]:,} mẫu")

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_sample_weight: bool = True,
    model_params: Optional[Dict[str, Any]] = None
) -> xgb.XGBClassifier:
    """
    Huấn luyện mô hình XGBoost Classifier.

    Tham số:
        X_train: Dữ liệu đặc trưng training
        y_train: Nhãn training
        use_sample_weight: Có sử dụng sample weight để cân bằng lớp không
        model_params: Dictionary các tham số cho mô hình (override mặc định)

    Trả về:
        Mô hình XGBoost đã được huấn luyện

    Giải thích các tham số mô hình:
        - n_estimators: Số lượng cây (weak learners)
            Nhiều cây hơn -> chính xác hơn nhưng chậm hơn
        - learning_rate: Tốc độ học (shrinkage rate)
            Nhỏ hơn -> học chậm hơn nhưng ổn định hơn
        - max_depth: Độ sâu tối đa của mỗi cây
            Sâu hơn -> phức tạp hơn -> có thể overfit
        - objective: Hàm mục tiêu
            'multi:softprob': Phân loại nhiều lớp, trả về xác suất
        - eval_metric: Metric để đánh giá trong quá trình training
            'mlogloss': Multiclass log loss
    """
    print("[INFO] Đang huấn luyện mô hình XGBoost...")

    # Thiết lập tham số mặc định
    default_params = {
        'n_estimators': 100,          # 100 cây decision tree
        'learning_rate': 0.1,         # Tốc độ học vừa phải
        'max_depth': 5,               # Độ sâu tối đa 5 để tránh overfit
        'objective': 'multi:softprob',  # Phân loại nhiều lớp
        'num_class': len(np.unique(y_train)),  # Số lượng lớp
        'eval_metric': 'mlogloss',    # Multiclass log loss
        'use_label_encoder': False,   # Tắt warning về label encoder
        'random_state': 42,           # Seed để reproducibility
        'n_jobs': -1                  # Sử dụng tất cả CPU cores
    }

    # Ghi đè bằng các tham số được cung cấp (nếu có)
    if model_params:
        default_params.update(model_params)

    # Khởi tạo mô hình
    model = xgb.XGBClassifier(**default_params)

    # Tính sample weight nếu cần
    # Sample weight giúp mô hình chú ý hơn đến các lớp thiểu số
    # Lớp có ít mẫu sẽ được gán weight cao hơn
    if use_sample_weight:
        sample_weight = compute_sample_weight(
            class_weight='balanced',  # Tự động tính weight dựa trên tần suất
            y=y_train
        )
        print("[INFO] Đang sử dụng sample weight để cân bằng các lớp")
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    print("[INFO] Huấn luyện mô hình hoàn tất!")

    # In thông tin về mô hình
    print(f"\n[INFO] Thông tin mô hình:")
    print(f"  - Số cây: {model.n_estimators}")
    print(f"  - Độ sâu tối đa: {model.max_depth}")
    print(f"  - Tốc độ học: {model.learning_rate}")

    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Đánh giá hiệu suất của mô hình trên tập test.

    Tham số:
        model: Mô hình đã huấn luyện
        X_test: Dữ liệu đặc trưng test
        y_test: Nhãn test thực tế
        class_names: Tên các lớp để hiển thị

    Trả về:
        Dictionary chứa các metrics đánh giá:
            - accuracy: Độ chính xác tổng thể
            - precision: Precision trung bình (macro)
            - recall: Recall trung bình (macro)
            - f1: F1-score trung bình (macro)
            - confusion_matrix: Ma trận nhầm lẫn
            - classification_report: Báo cáo chi tiết

    Giải thích các metrics:
        - Accuracy: Tỷ lệ dự đoán đúng trên tổng số mẫu
        - Precision: Trong số các mẫu dự đoán là lớp X, bao nhiêu thực sự là X
        - Recall: Trong số các mẫu thực sự là lớp X, bao nhiêu được dự đoán đúng
        - F1-score: Trung bình điều hòa của Precision và Recall
    """
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)

    if class_names is None:
        class_names = ['0 - Không bị thương', '1 - Bị thương nhẹ', '2 - Nghiêm trọng']

    # Dự đoán trên tập test
    y_pred = model.predict(X_test)

    # Tính các metrics
    # ============================================================
    # 1. Accuracy: Tỷ lệ dự đoán đúng
    # ============================================================
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy (Độ chính xác tổng thể): {accuracy:.4f} ({accuracy*100:.2f}%)")

    # ============================================================
    # 2. Precision, Recall, F1 (macro average)
    # ============================================================
    # 'macro': Tính trung bình không trọng số của các lớp
    # Phù hợp khi các lớp đều quan trọng như nhau
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")

    # ============================================================
    # 3. Classification Report: Báo cáo chi tiết từng lớp
    # ============================================================
    print("\n" + "-" * 40)
    print("Báo cáo chi tiết (Classification Report):")
    print("-" * 40)
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    # ============================================================
    # 4. Confusion Matrix: Ma trận nhầm lẫn
    # ============================================================
    print("-" * 40)
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Giải thích ma trận
    print("\nGiải thích ma trận:")
    print("  - Hàng: Lớp thực tế")
    print("  - Cột: Lớp dự đoán")
    print("  - Đường chéo: Số dự đoán đúng")

    # Tính tỷ lệ nhầm lẫn cho lớp nghiêm trọng (quan trọng nhất)
    if len(cm) > 2:
        # Lớp 2 là "Nghiêm trọng"
        true_severe = cm[2, :]  # Hàng thứ 3 (index 2)
        total_severe = true_severe.sum()
        correct_severe = cm[2, 2]
        severe_recall = correct_severe / total_severe if total_severe > 0 else 0

        print(f"\n[QUAN TRỌNG] Recall cho lớp 'Nghiêm trọng': {severe_recall:.4f}")
        print(f"  -> Trong {total_severe} ca nghiêm trọng thực tế,")
        print(f"     mô hình phát hiện đúng {correct_severe} ca ({severe_recall*100:.1f}%)")

    # Trả về kết quả dưới dạng dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred
    }

    return results


def save_model(
    model: xgb.XGBClassifier,
    filepath: str = 'model_xgboost.pkl'
) -> None:
    """
    Lưu mô hình đã huấn luyện vào file.

    Tham số:
        model: Mô hình cần lưu
        filepath: Đường dẫn file đích (mặc định: 'model_xgboost.pkl')

    Lưu ý:
        - Sử dụng joblib để lưu vì hiệu quả với các mô hình sklearn/xgboost
        - File .pkl là định dạng pickle của Python
        - Có thể tải lại bằng hàm load_model()
    """
    # Tạo thư mục nếu chưa tồn tại
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Lưu mô hình
    joblib.dump(model, filepath)
    print(f"[INFO] Đã lưu mô hình vào: {filepath}")


def load_model(filepath: str = 'model_xgboost.pkl') -> xgb.XGBClassifier:
    """
    Tải mô hình từ file.

    Tham số:
        filepath: Đường dẫn đến file mô hình

    Trả về:
        Mô hình XGBoost đã tải

    Ngoại lệ:
        FileNotFoundError: Khi không tìm thấy file mô hình
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {filepath}")

    model = joblib.load(filepath)
    print(f"[INFO] Đã tải mô hình từ: {filepath}")

    return model


def predict_single(
    model: xgb.XGBClassifier,
    features: dict,
    feature_order: list
) -> Tuple[int, np.ndarray]:
    """
    Dự đoán cho một mẫu dữ liệu đơn lẻ.

    Tham số:
        model: Mô hình đã huấn luyện
        features: Dictionary chứa các đặc trưng của mẫu cần dự đoán
        feature_order: Danh sách tên các đặc trưng theo đúng thứ tự

    Trả về:
        Tuple gồm:
            - prediction: Lớp dự đoán (0, 1, hoặc 2)
            - probabilities: Xác suất cho mỗi lớp

    Ví dụ:
        >>> features = {'hour': 14, 'age': 35, 'catv': 7, ...}
        >>> pred, probs = predict_single(model, features, feature_names)
        >>> print(f"Dự đoán: {pred}, Xác suất: {probs}")
    """
    # Tạo DataFrame từ dictionary với đúng thứ tự cột
    X = pd.DataFrame([features])[feature_order]

    # Dự đoán lớp
    prediction = model.predict(X)[0]

    # Lấy xác suất cho mỗi lớp
    probabilities = model.predict_proba(X)[0]

    return int(prediction), probabilities


def get_severity_label(prediction: int) -> str:
    """
    Chuyển đổi mã lớp thành nhãn có ý nghĩa.

    Tham số:
        prediction: Mã lớp dự đoán (0, 1, hoặc 2)

    Trả về:
        Chuỗi mô tả mức độ nghiêm trọng
    """
    labels = {
        0: "Không bị thương",
        1: "Bị thương nhẹ",
        2: "Nghiêm trọng (bao gồm tử vong và bị thương nặng)"
    }
    return labels.get(prediction, "Không xác định")
