import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    df_caracs = pd.read_csv('data/caracteristiques-2019.csv', sep=',', encoding='utf-8')
    df_lieux = pd.read_csv('data/lieux-2019.csv', sep=',', encoding='utf-8')
    df_usagers = pd.read_csv('data/usagers-2019.csv', sep=',', encoding='utf-8')
    df_vehicules = pd.read_csv('data/vehicules-2019.csv', sep=',')
    print("Đã load file thành công!")
except Exception as e:
    print("Lỗi load file")

df_caracs['Num_Acc'] = df_caracs['Num_Acc'].astype(str)
df_lieux['Num_Acc'] = df_lieux['Num_Acc'].astype(str)
df_usagers['Num_Acc'] = df_usagers['Num_Acc'].astype(str)
df_vehicules['Num_Acc'] = df_vehicules['Num_Acc'].astype(str)

df_1 = pd.merge(df_caracs, df_lieux, on='Num_Acc', how='left')
df_2 = pd.merge(df_usagers, df_vehicules, on=['Num_Acc', 'num_veh', 'id_vehicule'], how='left')
df = pd.merge(df_1, df_2, on='Num_Acc', how='left')

print(f"Kích thước bảng cuối cùng: {df.shape}")
print("-" * 30)
print(df.head(5))
print(df.columns)
print(f"số lượng khuyết thiếu dữ liệu mỗi cột: \n{df.isna().sum()}")
print("Các cột trong dataset:", df.columns)
slt = df.isna().any(axis=1).sum()
print("Số lượng hàng thiếu dữ liệu:", slt)
print("Số lượng các lớp mục tiêu:\n", df['grav'].value_counts())
df = df.drop(['v2', 'lartpc', 'larrout', 'occutc'], axis=1)
df = df.drop(['Num_Acc', 'id_vehicule', 'num_veh', 'adr', 'voie', 'pr', 'pr1', 'lat', 'long'], axis=1)
print(df.columns)

grav_missing = df['grav'].isna().sum()
print('số dòng thiếu dữ liệu cột mục tiêu:', grav_missing)
df = df.dropna(subset=['grav'])
print(f'đã xoá {grav_missing} dòng')
slt = (df.isna().sum(axis=1)>10).sum()
print("Số lượng hàng thiếu quá nhiều dữ liệu:", slt)
df_clean = df.dropna(thresh=len(df.columns)-10)
print(f"Đã xóa {len(df) - len(df_clean)}")
df = df_clean

print(f"số lượng khuyết thiếu dữ liệu mỗi cột: \n{df.isna().sum()}")
num_duplicates = df.duplicated().sum()
print(f"Tổng số dòng bị trùng lặp: {num_duplicates}")
df = df.drop_duplicates()
num_duplicates = df.duplicated().sum()
print(f"số dòng bị trùng lặp còn lại: {num_duplicates}")

mapping_grav = {
    1: 0,
    4: 1,
    2: 2,
    3: 2
}
df['grav'] = df['grav'].map(mapping_grav)

print("Số lượng các lớp sau khi gộp:")
print(df['grav'].value_counts())
from sklearn.preprocessing import LabelEncoder

if 'an_nais' in df.columns:
    df['age'] = 2019 - df['an_nais'] 
    df = df.drop('an_nais', axis=1)
df['hour'] = pd.to_numeric(df['hrmn'], errors='coerce') * 24

le = LabelEncoder()

cols_encode = df.select_dtypes(include=['object']).columns
for col in cols_encode:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    print(f"Đã encode cột: {col}")
print(df.info())
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


fig, axes = plt.subplots(2, 2, figsize=(18, 12))

sns.countplot(x='grav', data=df, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Phân bố mức độ nghiêm trọng (Gravity)')

sns.histplot(data=df, x='hour', bins=24, kde=True, ax=axes[0, 1], color='orange')
axes[0, 1].set_title('Phân bố tai nạn theo giờ trong ngày')

sns.boxplot(x='grav', y='age', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Độ tuổi theo mức độ nghiêm trọng')

top_catv = df['catv'].value_counts().index[:10]
sns.countplot(y='catv', data=df[df['catv'].isin(top_catv)], ax=axes[1, 1], order=top_catv)
axes[1, 1].set_title('Top 10 loại phương tiện gặp nạn')

plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib

X = df.drop('grav', axis=1)
y = df['grav']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

weight_balanced = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)


model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

print("đang huấn luyện mô hình")
model.fit(X_train, y_train, sample_weight=weight_balanced)

joblib.dump(model, 'model_xgboost.pkl')

print("Đã lưu mô hình thành công!")
print("hoàn tất")
y_pred = model.predict(X_test)

print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"Accuracy (Độ chính xác tổng thể): {accuracy_score(y_test, y_pred):.4f}")
print("\nBáo cáo chi tiết (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['1 - Không bị thương', '2 - Bị thương nhẹ', '3 - Nghiêm trọng']))

print("\nMa trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
