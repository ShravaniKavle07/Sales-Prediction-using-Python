import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load model
rf_model = joblib.load('sales_model.pkl')  # Make sure this file exists
lr_model = LinearRegression()

# Title
st.title("📈 Sales Prediction Dashboard")

# Sidebar inputs
st.sidebar.header("Manual Input")
tv = st.sidebar.slider("TV Spend", 0.0, 300.0, 150.0)
radio = st.sidebar.slider("Radio Spend", 0.0, 100.0, 25.0)
newspaper = st.sidebar.slider("Newspaper Spend", 0.0, 100.0, 20.0)

# Predict manually
input_df = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
rf_pred = rf_model.predict(input_df)[0]
st.subheader("🔮 Predicted Sales (Manual Input)")
st.metric(label="Random Forest Prediction", value=f"{rf_pred:.2f}")

# Sample CSV format
st.markdown("""
**📄 Sample CSV Format:**""")

# Upload CSV for batch prediction
st.subheader("📁 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV with columns: TV, Radio, Newspaper", type="csv")

required_cols = ['TV', 'Radio', 'Newspaper']
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    if all(col in batch_df.columns for col in required_cols):
        batch_pred = rf_model.predict(batch_df[required_cols])
        batch_df['Predicted_Sales'] = batch_pred
        st.write(batch_df)
        st.download_button("Download Predictions", batch_df.to_csv(index=False), "predictions.csv")

        # Feature importance
        st.subheader("📊 Feature Importance")
        plt.clf()
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({'Feature': required_cols, 'Importance': importances})
        sns.barplot(x='Importance', y='Feature', data=imp_df)
        st.pyplot(plt)

        # Historical trends
        if 'Date' in batch_df.columns:
            try:
                batch_df['Date'] = pd.to_datetime(batch_df['Date'])
                st.subheader("📈 Historical Trends")
                st.line_chart(batch_df.set_index('Date')['Predicted_Sales'])
            except Exception as e:
                st.warning(f"⚠️ Could not parse 'Date' column: {e}")

        # Model comparison
        if 'Predicted_Sales' in batch_df.columns:
            try:
                st.subheader("⚖️ Model Comparison")
                lr_model.fit(batch_df[required_cols], batch_df['Predicted_Sales'])
                lr_pred = lr_model.predict(batch_df[required_cols])
                comparison_df = pd.DataFrame({
                    'Random Forest': batch_df['Predicted_Sales'],
                    'Linear Regression': lr_pred
                })
                st.line_chart(comparison_df)
            except Exception as e:
                st.warning(f"⚠️ Model comparison failed: {e}")
    else:
        st.error("❌ Uploaded CSV must contain columns: TV, Radio, Newspaper")