# =====================================================
# 📈 Sales Prediction Dashboard
# Author: Shravani Kavle
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# ⚡ Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

sns.set_style("whitegrid")

# -----------------------------
# 🧠 Load Trained Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("sales_model.pkl")

rf_model = load_model()
model_columns = rf_model.feature_names_in_

# -----------------------------
# 📂 Load Sample Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Advertising.csv")

# -----------------------------
# 📈 App Header
# -----------------------------
st.title("📈 Sales Prediction Dashboard")
st.markdown(
    """
    Predict **product sales** based on advertising spend across different channels.
    
    - Manual sales forecasting
    - Batch predictions using CSV files
    - Model insights and feature importance
    """
)

# -----------------------------
# 📑 Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔮 Single Prediction", "📁 Batch Prediction", "📊 Data Insights"]
)

# =====================================================
# 🔮 TAB 1: SINGLE PREDICTION
# =====================================================
with tab1:
    st.subheader("Manual Sales Prediction")

    col1, col2 = st.columns(2)

    with col1:
        tv = st.slider("TV Advertising Spend ($)", 0.0, 300.0, 150.0)
        radio = st.slider("Radio Advertising Spend ($)", 0.0, 100.0, 25.0)

    with col2:
        newspaper = st.slider("Newspaper Advertising Spend ($)", 0.0, 100.0, 20.0)

    input_df = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    input_df = input_df[model_columns]

    prediction = rf_model.predict(input_df)[0]

    st.metric("📊 Predicted Sales", f"{prediction:.2f}")

# =====================================================
# 📁 TAB 2: BATCH PREDICTION
# =====================================================
with tab2:
    st.subheader("Batch Sales Prediction")

    use_sample = st.checkbox("Use sample dataset")

    if use_sample:
        batch_df = load_data()
        st.success("Sample dataset loaded successfully")
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
        else:
            st.info("Upload a CSV file or select sample dataset")
            st.stop()

    st.markdown("**Preview of Input Data**")
    st.dataframe(batch_df.head())

    encoded = batch_df[model_columns]

    batch_df["Predicted_Sales"] = rf_model.predict(encoded)

    st.subheader("Predicted Sales Results")
    st.dataframe(batch_df)

    st.download_button(
        label="📥 Download Predictions",
        data=batch_df.to_csv(index=False).encode("utf-8"),
        file_name="sales_predictions.csv",
        mime="text/csv"
    )

# =====================================================
# 📊 TAB 3: DATA INSIGHTS
# =====================================================
with tab3:
    st.subheader("Feature Importance Analysis")

    imp_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
    data=imp_df,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False,
    ax=ax

    )

    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

    st.caption("Feature importance explaining how advertising channels influence sales")
