import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# --- INITIAL CONFIGURATION ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Inventory Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a "Neat" UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- MODULE 1: DATA LAYER ---
@st.cache_data
def process_data(df_raw, store_id, prod_id):
    # Filter and chronological sorting [cite: 88, 91]
    df = df_raw[(df_raw['Store ID'] == store_id) & (df_raw['Product ID'] == prod_id)].copy()
    
    if len(df) < 40: # Minimum data threshold [cite: 20]
        return None
    
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    # Integrity & Missing Values [cite: 94, 97]
    df = df[(df['Price'] > 0) & (df['Inventory Level'] >= 0)]
    df['Inventory Level'] = df['Inventory Level'].ffill().bfill()

    # Outlier Capping (Winsorization) [cite: 105-108]
    for col in ['Units Sold', 'Price']:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df

# --- MODULE 2: REPRESENTATION LAYER ---
def feature_engineering(df):
    # Lags & Rolling (Leakage-Safe) [cite: 205-207]
    df['lag_1'] = df['Units Sold'].shift(1)
    df['lag_7'] = df['Units Sold'].shift(7)
    df['rolling_mean_7'] = df['Units Sold'].shift(1).rolling(7).mean()
    df['rolling_std_7'] = df['Units Sold'].shift(1).rolling(7).std()

    # Cyclic Time Features [cite: 202-205]
    day = df['Date'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * day / 7)
    df['day_cos'] = np.cos(2 * np.pi * day / 7)
    df['month'] = df['Date'].dt.month

    # Decision-Oriented Proxies [cite: 210, 213]
    df['effective_price'] = df['Price'] * (1 - df.get('Discount', 0))
    df['STR'] = df['Units Sold'].shift(1) / (df['Inventory Level'].shift(1) + 1e-6)
    df['DoS'] = df['Inventory Level'].shift(1) / (df['rolling_mean_7'] + 1e-6)

    return df.dropna().reset_index(drop=True)

# --- MODULE 3: LEARNING & DECISION LAYER ---
def train_model(df, features):
    n = len(df)
    train_idx, val_idx = int(n * 0.7), int(n * 0.85)
    train, val, test = df.iloc[:train_idx], df.iloc[train_idx:val_idx], df.iloc[val_idx:]
    
    # Base Learners [cite: 249]
    lasso = Lasso(alpha=0.1).fit(train[features], train['Units Sold'])
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(train[features], train['Units Sold'])
    lgbm = LGBMRegressor(n_estimators=100, verbosity=-1).fit(train[features], train['Units Sold'])

    # Validation-driven weights [cite: 251]
    val_preds = {
        'Lasso': lasso.predict(val[features]), 
        'RF': rf.predict(val[features]), 
        'LGBM': lgbm.predict(val[features])
    }
    inv_rmse = {name: 1/(np.sqrt(mean_squared_error(val['Units Sold'], p)) + 1e-6) for name, p in val_preds.items()}
    weights = {name: w/sum(inv_rmse.values()) for name, w in inv_rmse.items()}

    # Ensemble Forecast [cite: 251]
    ens_p = (lasso.predict(test[features]) * weights['Lasso']) + \
            (rf.predict(test[features]) * weights['RF']) + \
            (lgbm.predict(test[features]) * weights['LGBM'])

    # Performance Metrics [cite: 254, 314-316]
    wape = 100 * np.sum(np.abs(test['Units Sold'] - ens_p)) / (np.sum(test['Units Sold']) + 1e-6)
    smape = 100/len(test) * np.sum(2 * np.abs(test['Units Sold'] - ens_p) / (np.abs(test['Units Sold']) + np.abs(ens_p) + 1e-6))
    
    res = test.copy()
    res['Forecast'], res['Error'] = ens_p, test['Units Sold'] - ens_p
    importance = pd.DataFrame({'Feature': features, 'Importance': lgbm.feature_importances_}).sort_values('Importance', ascending=False)
    
    return res, importance, {'WAPE': wape, 'sMAPE': smape}, weights

# --- MAIN UI ---
with st.sidebar:
    st.title("⚙️ Planner Controls")
    uploaded_file = st.file_uploader("Upload Retail CSV", type="csv")
    if uploaded_file:
        raw_data = pd.read_csv(uploaded_file)
        st.divider()
        store = st.selectbox("Store ID", sorted(raw_data['Store ID'].unique()))
        product = st.selectbox("Product ID", sorted(raw_data[raw_data['Store ID']==store]['Product ID'].unique()))
        lt = st.slider("Lead Time (Days)", 1, 14, 7) # [cite: 257]
        sl = st.select_slider("Service Level", options=[0.90, 0.95, 0.99], value=0.95)
        z = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}[sl]

if not uploaded_file:
    st.info("👋 Welcome! Please upload your retail transaction data in the sidebar to begin.")
else:
    df_clean = process_data(raw_data, store, product)
    if df_clean is not None:
        df_feats = feature_engineering(df_clean)
        feats = ['lag_1', 'lag_7', 'rolling_mean_7', 'day_sin', 'day_cos', 'month', 'effective_price', 'STR', 'DoS']
        res_df, feat_imp, metrics, weights = train_model(df_feats, feats)

        # Top Metric Row
        safety_stock = z * np.sqrt(lt) * res_df['Forecast'].std()
        reorder_point = (res_df['Forecast'].mean() * lt) + safety_stock
        current_inv = res_df['Inventory Level'].iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WAPE Accuracy", f"{metrics['WAPE']:.1f}%")
        c2.metric("sMAPE", f"{metrics['sMAPE']:.1f}%")
        c3.metric("Safety Stock", int(safety_stock))
        c4.metric("Reorder Point", int(reorder_point))

        # Alert Banner [cite: 369]
        if current_inv < safety_stock: 
            st.error(f"🚨 **Stockout Risk**: Inventory ({int(current_inv)}) < Safety Stock ({int(safety_stock)})")
        elif current_inv < reorder_point: 
            st.warning("⚠️ **Replenishment Alert**: Current stock is below reorder point.")
        else: 
            st.success("✅ **Optimal Stock Levels**")

        # Visual Dashboard [cite: 261, 267]
        tab1, tab2, tab3 = st.tabs(["📈 Demand Forecast", "🔍 Explainability", "📊 Error Diagnostics"])
        
        with tab1:
            st.plotly_chart(px.line(res_df, x='Date', y=['Units Sold', 'Forecast'], 
                                   title="Actual vs Hybrid Ensemble Forecast",
                                   color_discrete_map={'Units Sold': '#3366CC', 'Forecast': '#FF9900'}), use_container_width=True)
        
        with tab2:
            col_a, col_b = st.columns(2)
            col_a.plotly_chart(px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Feature Drivers"), use_container_width=True)
            col_b.write("### Ensemble Composition")
            col_b.json(weights)

        with tab3:
            # Prediction-error distributions [cite: 355-356]
            st.plotly_chart(ff.create_distplot([res_df['Error']], ['Ensemble Error'], bin_size=1), use_container_width=True)
            st.write("### Detailed Statistics")
            st.dataframe(res_df[['Date', 'Units Sold', 'Forecast', 'Error', 'Inventory Level']].tail(10))

    else:
        st.error("Insufficient data (min 40 days) for the selected SKU.")
