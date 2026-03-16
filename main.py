import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# --- PART 1: VS CODE TERMINAL COMPARISON (Module 3-B) ---
def run_model_comparison(X_train, X_test, y_train, y_test):
    m_lasso = Lasso(alpha=0.1).fit(X_train, y_train)
    m_rf = RandomForestRegressor(n_estimators=50).fit(X_train, y_train)
    m_lgb = lgb.LGBMRegressor(n_estimators=100, verbosity=-1).fit(X_train, y_train)
    
    results = []
    for name, model in {"Lasso": m_lasso, "Random Forest": m_rf, "LightGBM": m_lgb}.items():
        p = model.predict(X_test)
        results.append({
            "Model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, p)),
            "R2": r2_score(y_test, p)
        })
    return pd.DataFrame(results), m_lgb

# --- PART 2: DASHBOARD UI ---
st.set_page_config(page_title="Major Project - AI Inventory Orchestrator", layout="wide")
st.title("🛡️ Time-Ordered, Leakage-Safe Forecast-to-Inventory Workflow")

st.sidebar.header("📁 Data Layer")
uploaded_file = st.sidebar.file_uploader("Upload data.csv", type="csv")
lead_time_val = st.sidebar.slider("Lead-time (Days)", 1, 14, 7)

if uploaded_file:
    # --- MODULE 1: DATA LAYER ---
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    
    # Selection Filters
    st.sidebar.subheader("🎯 SKU Drill-Down")
    store_id = st.sidebar.selectbox("Select Store ID", sorted(data['Store ID'].unique()))
    prod_id = st.sidebar.selectbox("Select Product ID", sorted(data[data['Store ID']==store_id]['Product ID'].unique()))
    
    df = data[(data['Store ID'] == store_id) & (data['Product ID'] == prod_id)].sort_values('Date')

    # --- MODULE 2: FEATURE LAYER ---
    for l in [7, 14, 21]:
        df[f'lag_{l}'] = df['Units Sold'].shift(l)
    df['rolling_mean_7'] = df['Units Sold'].shift(1).rolling(7).mean()
    
    le = LabelEncoder()
    for col in ['Region', 'Category', 'Weather Condition', 'Holiday/Promotion']:
        df[col] = le.fit_transform(df[col].astype(str))
    df = df.dropna()

    # --- MODULE 3: LEARNING & DECISION LAYER ---
    features = [c for c in df.columns if any(x in c for x in ['lag', 'rolling'])] + \
               ['Region', 'Category', 'Weather Condition', 'Holiday/Promotion', 'Price', 'Discount', 'Inventory Level']
    
    train_idx = int(len(df) * 0.8)
    train, test = df.iloc[:train_idx], df.iloc[train_idx:]
    
    # Get Comparison Data & Best Model
    comparison_df, final_model = run_model_comparison(train[features], test[features], train['Units Sold'], test['Units Sold'])
    
    # Predictions
    original_preds = final_model.predict(test[features])

    # --- DASHBOARD OUTPUTS ---
    st.success(f"✅ Active Analysis for Store {store_id} | Product {prod_id}")
    
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    ss = 1.65 * np.sqrt(lead_time_val) * original_preds.std()
    rop = (original_preds.mean() * lead_time_val) + ss
    
    m1.metric("Model RMSE", f"{np.sqrt(mean_squared_error(test['Units Sold'], original_preds)):.2f}")
    m2.metric("Model R² Accuracy", f"{r2_score(test['Units Sold'], original_preds):.4f}")
    m3.metric("Safety Stock", f"{int(ss)} Units")
    m4.metric("Reorder Point", f"{int(rop)} Units")

    st.markdown("---")
    
    # GRAPHS SECTION
    g1, g2 = st.columns(2)
    
    with g1:
        st.subheader("📊 Model Performance Comparison (Box B)")
        # Show how different algorithms performed
        fig_comp = px.bar(comparison_df, x='Model', y='RMSE', color='Model', text_auto='.2f',
                          title="Algorithm Comparison (Lower RMSE is Better)")
        st.plotly_chart(fig_comp, use_container_width=True)

    with g2:
        st.subheader("📉 Actual vs. Predicted Units (Box C)")
        # Show the actual tracking of the model
        plot_df = pd.DataFrame({
            'Date': test['Date'].values,
            'Actual Sales': test['Units Sold'].values,
            'AI Forecast': original_preds
        })
        fig_line = px.line(plot_df, x='Date', y=['Actual Sales', 'AI Forecast'], 
                          title="Time-Series Forecasting Accuracy")
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
    g3, g4 = st.columns(2)

    with g3:
        st.subheader("🧠 Demand Driver Importance (Explainability)")
        imp_df = pd.DataFrame({'Feature': features, 'Importance': final_model.feature_importances_}).sort_values('Importance', ascending=True).tail(10)
        st.plotly_chart(px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance'), use_container_width=True)

    with g4:
        st.subheader("⚖️ Forecast Error Distribution")
        # Showing residuals/errors helps understand model bias
        errors = test['Units Sold'] - original_preds
        fig_hist = px.histogram(errors, nbins=30, title="Prediction Residuals (Error Spread)",
                               labels={'value': 'Error Magnitude'}, color_discrete_sequence=['indianred'])
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.info("Please upload 'data.csv' to visualize the Architecture Workflow.")
