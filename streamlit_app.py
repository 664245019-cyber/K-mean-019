"""
K-Means Clustering Web Application
Author: Machine Learning Course
Description: Interactive web app for K-Means clustering predictions (Self-contained version)
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
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful design
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        background-color: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        color: #667eea;
        font-size: 1rem;
        font-weight: 600;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
        font-weight: 700;
        color: #333;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 2rem 0;
    }
    .result-card h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .result-card .cluster-number {
        font-size: 4rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Model & Scaler on-the-fly (Guaranteed to work without external files)
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
    <h1>🔮 K-Means Clustering App</h1>
    <p>Interactive Machine Learning Prediction System (Live Model)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 About")
    st.info("""
    This app runs an embedded K-Means clustering model using the Iris dataset.
    
    **Model Details:**
    - Algorithm: K-Means ($K=3$)
    - Features: 4 (Sepal & Petal dimensions)
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
            <h2>🎉 Prediction Result</h2>
            <div class="cluster-number">Cluster {cluster}</div>
            <p>Distance to cluster center: {closest_distance:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown(f'<div class="metric-card"><h3>Input Features</h3><p style="font-size: 1rem;">4 values</p></div>', unsafe_allow_html=True)
        with mcol2:
            st.markdown(f'<div class="metric-card"><h3>Cluster Assigned</h3><p>{cluster}</p></div>', unsafe_allow_html=True)
        with mcol3:
            st.markdown(f'<div class="metric-card"><h3>Confidence</h3><p style="font-size: 1.5rem;">{1/(1+closest_distance):.2%}</p></div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Feature Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=input_scaled[0].tolist() + [input_scaled[0][0]],
            theta=feature_names + [feature_names[0]],
            fill='toself',
            name='Input Sample',
            line_color='rgb(102, 126, 234)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=model.cluster_centers_[cluster].tolist() + [model.cluster_centers_[cluster][0]],
            theta=feature_names + [feature_names[0]],
            fill='toself',
            name=f'Cluster {cluster} Center',
            line_color='rgb(255, 99, 132)'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=500)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Batch Prediction
with tab2:
    st.markdown("### 📁 Upload CSV File")
    st.info(f"Required column names: `{', '.join(feature_names)}`")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if set(feature_names).issubset(set(df.columns)):
                X_batch = df[feature_names].values
                df['Predicted_Cluster'] = model.predict(scaler.transform(X_batch))
                
                st.dataframe(df, use_container_width=True)
                st.download_button("📥 Download Results", df.to_csv(index=False), 'predictions.csv', 'text/csv', use_container_width=True)
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
st.markdown("<div style='text-align: center; color: #666; padding: 1rem;'><p>🎓 Machine Learning Course | Built with Streamlit</p></div>", unsafe_allow_html=True)