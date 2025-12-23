import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# đọc dữ liệu
try:
    df_caracs = pd.read_csv('/kaggle/input/2019-database-of-road-traffic-injuries/caracteristiques-2019.csv', sep=',', encoding='utf-8')
    df_lieux = pd.read_csv('/kaggle/input/2019-database-of-road-traffic-injuries/lieux-2019.csv', sep=',', encoding='utf-8')
    df_usagers = pd.read_csv('/kaggle/input/2019-database-of-road-traffic-injuries/usagers-2019.csv', sep=',', encoding='utf-8')
    df_vehicules = pd.read_csv('/kaggle/input/2019-database-of-road-traffic-injuries/vehicules-2019.csv', sep=',')
    print("Đã load file thành công!")
except Exception as e:
    print("Lỗi load file")

# In tên cột ra để xem
# print("Cột trong Caracteristiques:", df_caracs.columns.tolist())

# chuyển thành str để gộp
df_caracs['Num_Acc'] = df_caracs['Num_Acc'].astype(str)
df_lieux['Num_Acc'] = df_lieux['Num_Acc'].astype(str)
df_usagers['Num_Acc'] = df_usagers['Num_Acc'].astype(str)
df_vehicules['Num_Acc'] = df_vehicules['Num_Acc'].astype(str)

# gộp bảng
df_1 = pd.merge(df_caracs, df_lieux, on='Num_Acc', how='left')
df_2 = pd.merge(df_usagers, df_vehicules, on=['Num_Acc', 'num_veh', 'id_vehicule'], how='left')
df = pd.merge(df_1, df_2, on='Num_Acc', how='left')

# check kết quả gộp
print(f"Kích thước bảng cuối cùng: {df.shape}")
print("-" * 30)
print(df.head(5))
print(df.columns)
# kiểm tra khuyết thiếu
print(f"số lượng khuyết thiếu dữ liệu mỗi cột: \n{df.isna().sum()}")
print("Các cột trong dataset:", df.columns)
slt = df.isna().any(axis=1).sum()
print("Số lượng hàng thiếu dữ liệu:", slt)
# kiểm tra số lượng lớp mục tiêu
print("Số lượng các lớp mục tiêu:\n", df['grav'].value_counts())
# xoá các cột thiếu nhiều dữ liệu
df = df.drop(['v2', 'lartpc', 'larrout', 'occutc'], axis=1)
# xoá dữ liệu không cần thiết
df = df.drop(['Num_Acc', 'id_vehicule', 'num_veh', 'adr', 'voie', 'pr', 'pr1', 'lat', 'long'], axis=1)
print(df.columns)

# xoá các dòng thiếu dữ liệu ở cột mục tiêu
grav_missing = df['grav'].isna().sum()
print('số dòng thiếu dữ liệu cột mục tiêu:', grav_missing)
df = df.dropna(subset=['grav'])
print(f'đã xoá {grav_missing} dòng')
# kiểm tra số dòng thiếu quá nhiều
slt = (df.isna().sum(axis=1)>10).sum()
print("Số lượng hàng thiếu quá nhiều dữ liệu:", slt)
df_clean = df.dropna(thresh=len(df.columns)-10)
print(f"Đã xóa {len(df) - len(df_clean)}")
df = df_clean

# Điền khuyết:
# không cần thiết vì xgboost có thể xử lý tốt dữ liệu khuyết thiếu nếu cột đó không thiếu quá nhiều
print(f"số lượng khuyết thiếu dữ liệu mỗi cột: \n{df.isna().sum()}")
# xoá trùng lặp
num_duplicates = df.duplicated().sum()
print(f"Tổng số dòng bị trùng lặp: {num_duplicates}")
df = df.drop_duplicates()
num_duplicates = df.duplicated().sum()
print(f"số dòng bị trùng lặp còn lại: {num_duplicates}")

# gộp lớp 2 - 3
mapping_grav = {
    1: 0,  # Không bị thương
    4: 1,  # Bị thương nhẹ
    2: 2,  # Tử vong -> Nghiêm trọng
    3: 2   # Nặng -> Nghiêm trọng
}
df['grav'] = df['grav'].map(mapping_grav)

print("Số lượng các lớp sau khi gộp:")
print(df['grav'].value_counts())
# xử lý dữ liệu
from sklearn.preprocessing import LabelEncoder

# Tạo cột Tuổi nếu chưa có
if 'an_nais' in df.columns:
    df['age'] = 2019 - df['an_nais'] 
    df = df.drop('an_nais', axis=1)
# xử lý cột 'hrmn' về dạng số nguyên giờ
df['hour'] = pd.to_numeric(df['hrmn'], errors='coerce') * 24

le = LabelEncoder()

# Lấy danh sách các cột cần label encode là các cột có kiểu object
# Chuyển hết các cột đang là kiểu 'object'
cols_encode = df.select_dtypes(include=['object']).columns
for col in cols_encode:
    # Chuyển về chuỗi hết để tránh lỗi lẫn lộn số/chữ
    df[col] = df[col].astype(str)
    # Thực hiện Label Encoding
    df[col] = le.fit_transform(df[col])
    print(f"Đã encode cột: {col}")
# Kiểm tra lại
print(df.info())
# khái quát hoá dữ liệu 
print(df.describe())
# Khái quát hoá dữ liệu bằng biểu đồ
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Thiết lập khung hình 2x2
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Biểu đồ mức độ nghiêm trọng (Bar Chart)
sns.countplot(x='grav', data=df, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Phân bố mức độ nghiêm trọng (Gravity)')

# 2. Biểu đồ theo Giờ (Histogram/KDE)
sns.histplot(data=df, x='hour', bins=24, kde=True, ax=axes[0, 1], color='orange')
axes[0, 1].set_title('Phân bố tai nạn theo giờ trong ngày')

# 3. Phân bố Tuổi (Boxplot)
sns.boxplot(x='grav', y='age', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Độ tuổi theo mức độ nghiêm trọng')

# 4. Các yếu tố xe cộ/Môi trường (Ví dụ: Loại phương tiện - catv)
# Chỉ lấy top 10 loại xe phổ biến nhất để đỡ rối
top_catv = df['catv'].value_counts().index[:10]
sns.countplot(y='catv', data=df[df['catv'].isin(top_catv)], ax=axes[1, 1], order=top_catv)
axes[1, 1].set_title('Top 10 loại phương tiện gặp nạn')

plt.tight_layout()
plt.show()
# tiến hành chia dữ liệu và huấn luyện
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# chuẩn bị dữ liệu
# Tách biến đầu vào (X) và biến mục tiêu (y)
X = df.drop('grav', axis=1)
y = df['grav']

# chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

weight_balanced = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)


# huấn luyện mô hình
model = xgb.XGBClassifier(
    n_estimators=100,     # Số lượng cây 
    learning_rate=0.1,    # Tốc độ học
    max_depth=5,          # Độ sâu của cây
    objective='multi:softprob', # Bài toán phân loại nhiều lớp
    num_class=len(np.unique(y)), # Số lượng lớp
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1      # chỉ định số lượng CPU (-1 là dùng hết)
)

print("đang huấn luyện mô hình")
model.fit(X_train, y_train, sample_weight=weight_balanced)

# Lưu mô hình vào file có tên 'model_xgboost.pkl'
joblib.dump(model, 'model_xgboost.pkl')

print("Đã lưu mô hình thành công!")
print("hoàn tất")
# đánh giá mô hình
y_pred = model.predict(X_test)

# In kết quả
print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"Accuracy (Độ chính xác tổng thể): {accuracy_score(y_test, y_pred):.4f}")
print("\nBáo cáo chi tiết (Classification Report):")
# target_names giúp hiển thị lại tên gốc thay vì 0,1,2
print(classification_report(y_test, y_pred, target_names=['1 - Không bị thương', '2 - Bị thương nhẹ', '3 - Nghiêm trọng']))

# Xem ma trận nhầm lẫn để biết nó hay dự đoán sai lớp nào
print("\nMa trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))

