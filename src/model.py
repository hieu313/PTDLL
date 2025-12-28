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
    print("[INFO] Đang chuẩn bị dữ liệu train/test...")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    print(f"[INFO] Số đặc trưng (features): {X.shape[1]}")
    print(f"[INFO] Phân bố lớp mục tiêu:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
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

    print("[INFO] Đang huấn luyện mô hình XGBoost...")

    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1
    }

    if model_params:
        default_params.update(model_params)

    model = xgb.XGBClassifier(**default_params)

    if use_sample_weight:
        sample_weight = compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )
        print("[INFO] Đang sử dụng sample weight để cân bằng các lớp")
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    print("[INFO] Huấn luyện mô hình hoàn tất!")

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

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)

    if class_names is None:
        class_names = ['0 - Không bị thương', '1 - Bị thương nhẹ', '2 - Nghiêm trọng']

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy (Độ chính xác tổng thể): {accuracy:.4f} ({accuracy*100:.2f}%)")

    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")

    print("\n" + "-" * 40)
    print("Báo cáo chi tiết (Classification Report):")
    print("-" * 40)
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    print("-" * 40)
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nGiải thích ma trận:")
    print("  - Hàng: Lớp thực tế")
    print("  - Cột: Lớp dự đoán")
    print("  - Đường chéo: Số dự đoán đúng")

    if len(cm) > 2:
        true_severe = cm[2, :]
        total_severe = true_severe.sum()
        correct_severe = cm[2, 2]
        severe_recall = correct_severe / total_severe if total_severe > 0 else 0

        print(f"\n[QUAN TRỌNG] Recall cho lớp 'Nghiêm trọng': {severe_recall:.4f}")
        print(f"  -> Trong {total_severe} ca nghiêm trọng thực tế,")
        print(f"     mô hình phát hiện đúng {correct_severe} ca ({severe_recall*100:.1f}%)")

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

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    joblib.dump(model, filepath)
    print(f"[INFO] Đã lưu mô hình vào: {filepath}")


def load_model(filepath: str = 'model_xgboost.pkl') -> xgb.XGBClassifier:

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

    X = pd.DataFrame([features])[feature_order]

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    return int(prediction), probabilities


def get_severity_label(prediction: int) -> str:

    labels = {
        0: "Không bị thương",
        1: "Bị thương nhẹ",
        2: "Nghiêm trọng (bao gồm tử vong và bị thương nặng)"
    }
    return labels.get(prediction, "Không xác định")
