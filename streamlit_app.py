"""
K-Means Clustering Web Application (Clean & Minimalist Theme)
Author: Machine Learning Course
Description: Interactive web app for K-Means clustering predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Clean & Minimalist White Theme
st.markdown("""
<style>
    /* Main theme background */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Container styling */
    .main .block-container {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
    }
    
    /* Header styling */
    .main-header {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        color: #1e293b;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
        color: #64748b;
    }
    
    /* Metric cards */
    .metric-card {
        background: #f1f5f9;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .metric-card h3 {
        margin: 0;
        color: #475569;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Result card */
    .result-card {
        background: #ffffff;
        border: 2px solid #3b82f6;
        padding: 2rem;
        border-radius: 12px;
        color: #1e293b;
        text-align: center;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.08);
        margin: 1.5rem 0;
    }
    
    .result-card h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .result-card .cluster-number {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        color: #2563eb;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: background-color 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    
    /* Info box */
    .info-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Model & Scaler
@st.cache_resource
def init_model():
    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X_scaled)
    
    return model, scaler, feature_names

model, scaler, feature_names = init_model()

# Main header
st.markdown("""
<div class="main-header">
    <h1>✨ K-Means Clustering App</h1>
    <p>Clean & Minimalist Machine Learning System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 About")
    st.info("""
    Interactive K-Means clustering web application built with Streamlit.
    
    **Model Specs:**
    - Algorithm: K-Means ($K=3$)
    - Dataset: Iris Dataset
    - Features: 4 Parameters
    """)
    st.markdown("---")
    if st.button("🔄 Reset App"):
        st.rerun()

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📝 Manual Prediction", "📁 Batch Prediction", "📊 Model Information"])

# Tab 1: Manual Prediction
with tab1:
    st.markdown("### 🎯 Enter Feature Values")
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0, 0.1)
    
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 1.5, 0.1)
    
    if st.button("🔮 Predict Cluster", use_container_width=True):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]
        
        distances = np.linalg.norm(model.cluster_centers_ - input_scaled, axis=1)
        closest_distance = distances[cluster]
        
        st.markdown(f"""
        <div class="result-card">
            <h2>Prediction Result</h2>
            <div class="cluster-number">Cluster {cluster}</div>
            <p style="color: #64748b;">Distance to cluster center: {closest_distance:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown(f'<div class="metric-card"><h3>Input Features</h3><p>4 values</p></div>', unsafe_allow_html=True)
        with mcol2:
            st.markdown(f'<div class="metric-card"><h3>Assigned Cluster</h3><p>{cluster}</p></div>', unsafe_allow_html=True)
        with mcol3:
            st.markdown(f'<div class="metric-card"><h3>Confidence</h3><p>{1/(1+closest_distance):.1%}</p></div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Feature Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=input_scaled[0].tolist() + [input_scaled[0][0]],
            theta=feature_names + [feature_names[0]],
            fill='toself',
            name='Input Sample',
            line_color='#2563eb'
        ))
        fig.add_trace(go.Scatterpolar(
            r=model.cluster_centers_[cluster].tolist() + [model.cluster_centers_[cluster][0]],
            theta=feature_names + [feature_names[0]],
            fill='toself',
            name=f'Cluster {cluster} Center',
            line_color='#ef4444'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)), 
            showlegend=True, 
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Batch Prediction
with tab2:
    st.markdown("### 📁 Batch Prediction via CSV")
    st.info(f"Required column names: `{', '.join(feature_names)}`")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if set(feature_names).issubset(set(df.columns)):
                X_batch = df[feature_names].values
                df['Predicted_Cluster'] = model.predict(scaler.transform(X_batch))
                
                st.dataframe(df, use_container_width=True)
                st.download_button("📥 Download Results CSV", df.to_csv(index=False), 'predictions.csv', 'text/csv', use_container_width=True)
            else:
                st.error(f"Column mismatch! Required columns: {list(feature_names)}")
        except Exception as e:
            st.error(f"Error: {e}")

# Tab 3: Model Information
with tab3:
    st.markdown("### 📋 Model Parameters")
    st.write(f"- **Number of Clusters:** {model.n_clusters}")
    st.write(f"- **Inertia:** {model.inertia_:.4f}")
    
    st.markdown("### 📍 Cluster Centers")
    st.dataframe(pd.DataFrame(model.cluster_centers_, columns=feature_names), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94a3b8; padding: 1rem;'><p>🎓 Machine Learning Course | Clean Streamlit Design</p></div>", unsafe_allow_html=True)