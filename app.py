import streamlit as st
import numpy as np
import pickle

# --- ตั้งค่าหน้าเว็บ (Page Configuration) ---
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="📊",
    layout="centered"
)

# --- โหลดโมเดลและ Scaler ที่บันทึกไว้ ---
@st.cache_resource
def load_artifacts():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    kmeans_model, scaler = load_artifacts()
except Exception as e:
    st.error(f"ไม่พบไฟล์โมเดล กรุณารันโค้ดสร้างโมเดลก่อน: {e}")
    st.stop()

# --- ตกแต่งส่วนหัว (Header) ---
st.title("📊 K-Means Clustering Prediction")
st.markdown("ระบบทำนายกลุ่มข้อมูล (Cluster) ด้วยโมเดล Machine Learning แบบเรียบง่าย")
st.markdown("---")

# --- ส่วนรับข้อมูลจากผู้ใช้ (Sidebar & Main Form) ---
st.subheader("🛠️ กรอกข้อมูลเพื่อทำนายกลุ่ม")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        feature_1 = st.number_input(
            "ค่าตัวแปรที่ 1 (Feature 1)", 
            value=0.0, 
            format="%.4f",
            help="กรอกค่าตัวเลขเชิงตัวแปรตัวที่ 1"
        )
    with col2:
        feature_2 = st.number_input(
            "ค่าตัวแปรที่ 2 (Feature 2)", 
            value=0.0, 
            format="%.4f",
            help="กรอกค่าตัวเลขเชิงตัวแปรตัวที่ 2"
        )
        
    submit_button = st.form_submit_button(label="🚀 ทำนายผล (Predict)")

# --- ส่วนแสดงผลลัพธ์ ---
if submit_button:
    # 1. จัดรูปแบบข้อมูลนำเข้า
    input_data = np.array([[feature_1, feature_2]])
    
    # 2. Transform ข้อมูลด้วย Scaler ที่บันทึกไว้
    input_scaled = scaler.transform(input_data)
    
    # 3. ทำนาย Cluster
    cluster_pred = kmeans_model.predict(input_scaled)[0]
    
    # 4. แสดงผลลัพธ์แบบสวยงาม
    st.markdown("---")
    st.subheader("📌 ผลการวิเคราะห์")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="กลุ่มที่ทำนายได้ (Cluster)", value=f"Cluster {cluster_pred}")
    with col_res2:
        # คำนวณระยะห่างจากจุดศูนย์กลางคลัสเตอร์ (Centroid Distance)
        distances = kmeans_model.transform(input_scaled)
        confidence = distances[0][cluster_pred]
        st.metric(label="ระยะห่างจากจุดศูนย์กลาง", value=f"{confidence:.4f}")
        
    st.success(f"ข้อมูลชุดนี้ถูกจัดอยู่ใน **Cluster กลุ่มที่ {cluster_pred}** เรียบร้อยแล้วครับ!")