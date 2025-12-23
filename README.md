# 🚗 Phân tích và Dự đoán Mức độ Nghiêm trọng Tai nạn Giao thông

## Mô tả

Dự án này phân tích dữ liệu tai nạn giao thông năm 2019 tại Pháp và xây dựng mô hình machine learning (XGBoost) để dự đoán mức độ nghiêm trọng của tai nạn.

## Cấu trúc dự án

```plaintext
PTDLL/
├── main.py               # File gốc (deprecated)
├── main_new.py           # File chính mới (sử dụng modules)
├── app.py                # Giao diện Streamlit
├── requirements.txt      # Danh sách thư viện
├── src/                  # Package modules
│   ├── __init__.py
│   ├── data_loader.py        # Đọc dữ liệu CSV
│   ├── data_preprocessing.py # Tiền xử lý dữ liệu
│   ├── feature_engineering.py # Tạo đặc trưng
│   ├── visualization.py      # Trực quan hóa
│   └── model.py              # Huấn luyện & đánh giá
├── data/                 # Thư mục chứa dữ liệu CSV
├── models/               # Thư mục lưu mô hình
├── output/               # Thư mục lưu biểu đồ
└── docs/                 # Tài liệu
    └── error_analysis.md # Phân tích lỗi code gốc
```

## Cài đặt

```bash
# Clone project
cd PTDLL

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

## Dữ liệu

Tải dữ liệu từ Kaggle: [2019 Database of Road Traffic Injuries](https://www.kaggle.com/datasets/ahmedlahlou/2019-database-of-road-traffic-injuries)

Đặt các file CSV vào thư mục `data/`:

- `caracteristiques-2019.csv`
- `lieux-2019.csv`
- `usagers-2019.csv`
- `vehicules-2019.csv`

## Sử dụng

### 1. Huấn luyện mô hình

```bash
python main_new.py
```

### 2. Chạy giao diện web Streamlit

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại <http://localhost:8501>

## Mức độ nghiêm trọng

Mô hình dự đoán 3 mức độ:

- **0 - Không bị thương**: Không có thương tích
- **1 - Bị thương nhẹ**: Thương tích nhẹ
- **2 - Nghiêm trọng**: Tử vong hoặc bị thương nặng

## Kết quả

- Accuracy: ~65-70%
- F1-Score (Macro): ~55-60%

## Modules

| Module                   | Chức năng                                            |
| ------------------------ | ---------------------------------------------------- |
| `data_loader.py`         | Đọc dữ liệu từ 4 file CSV và gộp thành 1 DataFrame   |
| `data_preprocessing.py`  | Xử lý missing values, xóa cột/dòng, gộp lớp mục tiêu |
| `feature_engineering.py` | Tạo cột tuổi, chuyển đổi thời gian, Label Encoding   |
| `visualization.py`       | Tạo các biểu đồ trực quan hóa                        |
| `model.py`               | Huấn luyện XGBoost, đánh giá, lưu/tải mô hình        |

## Tác giả

PTDLL Team
