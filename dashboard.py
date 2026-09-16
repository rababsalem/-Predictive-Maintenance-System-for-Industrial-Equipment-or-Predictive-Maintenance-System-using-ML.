import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# Page Configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Header Section
st.title(" Predictive Maintenance Dashboard")
st.subheader("NASA Turbofan Jet Engine - Remaining Useful Life (RUL) Prediction")

# Database Connection
engine_db = create_engine('sqlite:///E:/Predictive Maintenance/predictive_maintenance.db')
df = pd.read_sql("SELECT * FROM predictions", engine_db)

# 1. Key Performance Indicators (KPIs)
col1, col2, col3, col4 = st.columns(4)

mae = df['error'].mean()
rmse = (df['error'] ** 2).mean() ** 0.5
total_cycles = len(df)
total_engines = df['engine_id'].nunique()

col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} cycles")
col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} cycles")
col3.metric("Total Test Samples", f"{total_cycles:,}")
col4.metric("Engines Monitored", f"{total_engines}")

st.markdown("---")

# 2. RUL Tracking over Cycles Chart
col_chart, col_filter = st.columns([3, 1])

with col_filter:
    st.markdown("###  Controls & Filters")
    selected_engine = st.selectbox(
        "Select Engine ID:", 
        sorted(df['engine_id'].unique())
    )
    
    st.markdown("""
    **Legend Details:**
    - **actual_RUL**: Real ground truth remaining cycles.
    - **predicted_RUL**: ML model forecast.
    """)

with col_chart:
    st.markdown(f"### RUL Degradation Line - Engine #{selected_engine}")
    engine_data = df[df['engine_id'] == selected_engine].sort_values('cycle')
    
    fig_line = px.line(
        engine_data, 
        x='cycle', 
        y=['actual_RUL', 'predicted_RUL'],
        labels={'value': 'Remaining Useful Life (Cycles)', 'cycle': 'Engine Cycle', 'variable': 'Metrics'},
        title=f"Actual vs Predicted RUL over Operational Cycles (Engine {selected_engine})"
    )
    fig_line.update_layout(legend_title_text='RUL Type')
    st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

# 3. Model Evaluation & High Error Analysis
col_scatter, col_table = st.columns([1, 1])

with col_scatter:
    st.markdown("### 📊 Actual vs Predicted RUL Scatter Plot")
    fig_scatter = px.scatter(
        df, 
        x='actual_RUL', 
        y='predicted_RUL', 
        color='error',
        color_continuous_scale='Reds',
        labels={'actual_RUL': 'Actual RUL', 'predicted_RUL': 'Predicted RUL', 'error': 'Absolute Error'},
        title="Model Prediction Distribution"
    )
    # Adding Ideal Prediction Line
    fig_scatter.add_shape(
        type='line', x0=0, y0=0, x1=df['actual_RUL'].max(), y1=df['actual_RUL'].max(),
        line=dict(color='Gray', dash='dash')
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_table:
    st.markdown("### ⚠️ Top 10 Highest Prediction Errors")
    high_error = df.sort_values('error', ascending=False).head(10)
    st.dataframe(high_error, use_container_width=True)