import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

try:
    from src.model import load_model, get_severity_label
except ImportError:
    import joblib

    def load_model(filepath):
        return joblib.load(filepath)

    def get_severity_label(prediction):
        labels = {
            0: "Không bị thương",
            1: "Bị thương nhẹ",
            2: "Nghiêm trọng (bao gồm tử vong và bị thương nặng)"
        }
        return labels.get(prediction, "Không xác định")


st.set_page_config(
    page_title="Dự đoán tai nạn giao thông",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .result-safe {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }

    .result-warning {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }

    .result-danger {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }

    .metric-container {
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


LIGHT_CONDITIONS = {
    "Ban ngày": 1,
    "Hoàng hôn hoặc bình minh": 2,
    "Ban đêm có đèn đường sáng": 3,
    "Ban đêm đèn đường tắt": 4,
    "Ban đêm không có đèn đường": 5
}

WEATHER_CONDITIONS = {
    "Bình thường": 1,
    "Mưa nhẹ": 2,
    "Mưa to": 3,
    "Tuyết/mưa đá": 4,
    "Sương mù/khói": 5,
    "Gió mạnh/bão": 6,
    "Thời tiết đẹp": 7,
    "Mây": 8
}

VEHICLE_TYPES = {
    "Xe đạp": 1,
    "Xe máy < 50cc": 2,
    "Xe máy 50-125cc": 3,
    "Xe máy > 125cc": 4,
    "Ô tô con": 7,
    "Xe tải nhẹ": 10,
    "Xe tải nặng": 13,
    "Xe buýt": 14,
    "Xe khách": 15,
    "Người đi bộ": 99
}

ROAD_SURFACE = {
    "Khô ráo": 1,
    "Ướt": 2,
    "Có nước đọng": 3,
    "Lũ lụt": 4,
    "Tuyết": 5,
    "Bùn": 6,
    "Đóng băng": 7,
    "Có dầu mỡ": 8
}

COLLISION_TYPE = {
    "Hai xe đối đầu": 1,
    "Hai xe cùng chiều": 2,
    "Đâm từ bên": 3,
    "Dây chuyền": 4,
    "Va chạm nhiều phương tiện": 5,
    "Va chạm khác": 6,
    "Không va chạm": 7
}


@st.cache_resource
def load_xgboost_model():

    model_paths = [
        'models/model_xgboost.pkl',
        'model_xgboost.pkl',
        '../models/model_xgboost.pkl'
    ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                model = load_model(path)
                return model, path
            except Exception as e:
                st.error(f"Lỗi khi tải mô hình từ {path}: {e}")

    return None, None


def create_probability_chart(probabilities, class_names):

    df = pd.DataFrame({
        'Mức độ': class_names,
        'Xác suất': probabilities * 100
    })

    colors = ['#28a745', '#ffc107', '#dc3545']

    fig = go.Figure(go.Bar(
        y=df['Mức độ'],
        x=df['Xác suất'],
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1f}%' for p in df['Xác suất']],
        textposition='auto'
    ))

    fig.update_layout(
        title='Xác suất cho mỗi mức độ nghiêm trọng',
        xaxis_title='Xác suất (%)',
        yaxis_title='',
        xaxis_range=[0, 100],
        height=300,
        showlegend=False
    )

    return fig


def get_result_style(prediction):

    if prediction == 0:
        return "result-safe"
    elif prediction == 1:
        return "result-warning"
    else:
        return "result-danger"


def get_result_emoji(prediction):

    if prediction == 0:
        return "✅"
    elif prediction == 1:
        return "⚠️"
    else:
        return "🚨"


def get_safety_tips(prediction):

    tips = {
        0: [
            "Tiếp tục duy trì các biện pháp an toàn",
            "Luôn đeo dây an toàn/mũ bảo hiểm",
            "Tuân thủ tốc độ quy định",
            "Giữ khoảng cách an toàn với xe phía trước"
        ],
        1: [
            "Cẩn thận hơn trong điều kiện hiện tại",
            "Giảm tốc độ nếu thời tiết xấu",
            "Bật đèn chiếu sáng nếu trời tối",
            "Tránh sử dụng điện thoại khi lái xe",
            "Tập trung quan sát đường"
        ],
        2: [
            "⚠️ CẢNH BÁO: Điều kiện hiện tại có nguy cơ cao!",
            "Cân nhắc hoãn chuyến đi nếu có thể",
            "Nếu bắt buộc phải đi, giảm tốc độ tối đa",
            "Bật đèn khẩn cấp nếu tầm nhìn kém",
            "Thông báo lộ trình cho người thân",
            "Kiểm tra kỹ tình trạng xe trước khi đi"
        ]
    }
    return tips.get(prediction, [])


def main():

    st.markdown('<h1 class="main-header">🚗 Dự đoán mức độ nghiêm trọng tai nạn giao thông</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Nhập thông tin tình huống giao thông để dự đoán mức độ nghiêm trọng nếu xảy ra tai nạn</p>',
                unsafe_allow_html=True)

    model, model_path = load_xgboost_model()

    with st.sidebar:
        st.header("ℹ️ Thông tin")

        if model is not None:
            st.success(f"✅ Mô hình đã tải thành công!")
            st.info(f"📁 Đường dẫn: {model_path}")
        else:
            st.error("❌ Không tìm thấy mô hình!")
            st.warning("""
            Hãy huấn luyện mô hình trước bằng cách chạy:
            ```
            python main_new.py
            ```
            """)

        st.divider()

        st.header("📊 Các mức độ nghiêm trọng")
        st.markdown("""
        - **0 - Không bị thương**: Không có thương tích
        - **1 - Bị thương nhẹ**: Thương tích nhẹ, không nguy hiểm đến tính mạng
        - **2 - Nghiêm trọng**: Bao gồm tử vong và bị thương nặng
        """)

        st.divider()

        st.header("📖 Hướng dẫn")
        st.markdown("""
        1. Nhập thông tin tình huống giao thông
        2. Nhấn nút "Dự đoán"
        3. Xem kết quả và lời khuyên an toàn
        """)

    if model is None:
        st.error("⚠️ Không thể thực hiện dự đoán vì chưa có mô hình!")
        st.info("Vui lòng huấn luyện mô hình trước bằng cách chạy `python main_new.py`")
        return

    st.header("📝 Nhập thông tin tình huống")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Thông tin người lái")

        age = st.slider(
            "Tuổi người lái",
            min_value=16,
            max_value=100,
            value=35,
            help="Độ tuổi của người điều khiển phương tiện"
        )

        gender = st.selectbox(
            "Giới tính",
            options=["Nam", "Nữ"],
            help="Giới tính của người lái"
        )

        vehicle_type = st.selectbox(
            "Loại phương tiện",
            options=list(VEHICLE_TYPES.keys()),
            index=4,
            help="Loại phương tiện đang sử dụng"
        )

    with col2:
        st.subheader("🌤️ Điều kiện môi trường")

        hour = st.slider(
            "Giờ (0-24)",
            min_value=0,
            max_value=23,
            value=14,
            help="Thời điểm trong ngày (giờ)"
        )

        light_condition = st.selectbox(
            "Điều kiện ánh sáng",
            options=list(LIGHT_CONDITIONS.keys()),
            help="Điều kiện ánh sáng tại thời điểm đó"
        )

        weather = st.selectbox(
            "Thời tiết",
            options=list(WEATHER_CONDITIONS.keys()),
            help="Điều kiện thời tiết"
        )

    with col3:
        st.subheader("🛣️ Điều kiện đường")

        road_surface = st.selectbox(
            "Tình trạng mặt đường",
            options=list(ROAD_SURFACE.keys()),
            help="Tình trạng mặt đường tại thời điểm đó"
        )

        collision_type = st.selectbox(
            "Loại va chạm (dự kiến)",
            options=list(COLLISION_TYPE.keys()),
            help="Loại va chạm có thể xảy ra"
        )

        urban_area = st.checkbox(
            "Trong khu vực đô thị",
            value=True,
            help="Tai nạn xảy ra trong hay ngoài khu vực đô thị"
        )

    st.divider()

    predict_button = st.button(
        "🔮 Dự đoán mức độ nghiêm trọng",
        type="primary",
        use_container_width=True
    )

    if predict_button:
        with st.spinner("Đang phân tích..."):
            features = {
                'age': age,
                'hour': hour,
                'lum': LIGHT_CONDITIONS[light_condition],
                'atm': WEATHER_CONDITIONS[weather],
                'catv': VEHICLE_TYPES[vehicle_type],
                'surf': ROAD_SURFACE[road_surface],
                'col': COLLISION_TYPE[collision_type],
                'sexe': 1 if gender == "Nam" else 2,
                'agg': 1 if urban_area else 2
            }

            try:
                X = pd.DataFrame([features])

                if hasattr(model, 'feature_names_in_'):
                    expected_features = model.feature_names_in_
                else:
                    expected_features = [f'feature_{i}' for i in range(model.n_features_in_)]

                for feature in expected_features:
                    if feature not in X.columns:
                        X[feature] = 0

                X = X.reindex(columns=expected_features, fill_value=0)

                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]

                st.divider()
                st.header("📊 Kết quả dự đoán")

                result_style = get_result_style(prediction)
                result_emoji = get_result_emoji(prediction)
                severity_label = get_severity_label(prediction)

                st.markdown(f"""
                <div class="result-card {result_style}">
                    <h2 style="text-align: center; margin: 0;">
                        {result_emoji} {severity_label}
                    </h2>
                    <p style="text-align: center; margin-top: 0.5rem;">
                        Độ tin cậy: {probabilities[prediction]*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

                class_names = ['Không thương', 'Nhẹ', 'Nghiêm trọng']
                fig = create_probability_chart(probabilities, class_names)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("💡 Lời khuyên an toàn")
                tips = get_safety_tips(prediction)
                for tip in tips:
                    st.markdown(f"- {tip}")

            except Exception as e:
                st.error(f"❌ Lỗi khi dự đoán: {e}")
                st.info("Mô hình có thể yêu cầu các features khác. Hãy kiểm tra lại.")

    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🚗 Ứng dụng dự đoán mức độ nghiêm trọng tai nạn giao thông</p>
        <p>Dữ liệu: Cơ sở dữ liệu tai nạn giao thông Pháp 2019</p>
        <p>⚠️ Lưu ý: Đây chỉ là công cụ tham khảo, không thay thế cho việc tuân thủ luật giao thông</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
