# ============================================
# SALES FORECASTING DASHBOARD
# Rossmann Store Sales — Prophet Model
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon=" ",
    layout="wide"
)

# ── TITLE ────────────────────────────────────
st.title("Rossmann Sales Forecasting Dashboard")
st.markdown("**Predict future sales using Meta's Prophet model**")
st.divider()

# ── LOAD DATA ────────────────────────────────
@st.cache_data  # data ek baar load hoga, baar baar nahi
def load_data():
    train = pd.read_csv('train_small.csv')
    store = pd.read_csv('store.csv')

    # Cleaning
    train['Date'] = pd.to_datetime(train['Date'])
    store['CompetitionDistance'].fillna(
        store['CompetitionDistance'].median(), inplace=True)

    cols = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
            'Promo2SinceWeek', 'Promo2SinceYear']
    for col in cols:
        store[col].fillna(0, inplace=True)
    store['PromoInterval'].fillna('None', inplace=True)

    # Merge
    df = pd.merge(train, store, on='Store', how='left')
    df = df[df['Open'] == 1]
    df = df[df['Sales'] > 0]
    return df

df = load_data()
st.success("Data loaded successfully!")

# ── SIDEBAR ──────────────────────────────────
st.sidebar.header("Settings")

# Store selector
store_list = sorted(df['Store'].unique())
store_id = st.sidebar.selectbox(
    "Select Store",
    options=store_list,
    index=store_list.index(1)  # ← Store 1 default
)

# Forecast weeks
forecast_weeks = st.sidebar.slider(
    "Forecast Weeks",
    min_value=2,
    max_value=12,
    value=6
)

st.sidebar.divider()
st.sidebar.markdown("**About**")
st.sidebar.markdown("Dataset: Rossmann Store Sales")
st.sidebar.markdown("Model: Facebook Prophet")

# ── METRICS ROW ──────────────────────────────
store_data = df[df['Store'] == store_id]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",     f"{len(store_data):,}")
col2.metric("Avg Daily Sales",   f"€{store_data['Sales'].mean():,.0f}")
col3.metric("Max Daily Sales",   f"€{store_data['Sales'].max():,.0f}")
col4.metric("Promo Days",        f"{store_data['Promo'].sum():,}")

st.divider()

# ── PROPHET MODEL ────────────────────────────
st.subheader(f"Sales Forecast — Store {store_id}")

# Data prepare
prophet_df = store_data[['Date', 'Sales']].rename(
    columns={'Date': 'ds', 'Sales': 'y'}
).sort_values('ds').reset_index(drop=True)

# Train/test split
test_days = forecast_weeks * 7
train_df  = prophet_df.iloc[:-test_days]
test_df   = prophet_df.iloc[-test_days:].reset_index(drop=True)

# Model
with st.spinner("Training Prophet model..."):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(train_df)
    future   = model.make_future_dataframe(periods=test_days)
    forecast = model.predict(future)

# ── FORECAST CHART ───────────────────────────
forecast_test = forecast.iloc[-test_days:].reset_index(drop=True)

fig = go.Figure()

# Confidence band
fig.add_trace(go.Scatter(
    x=forecast_test['ds'], y=forecast_test['yhat_upper'],
    fill=None, mode='lines',
    line=dict(color='rgba(231,76,60,0)'), showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast_test['ds'], y=forecast_test['yhat_lower'],
    fill='tonexty', mode='lines',
    line=dict(color='rgba(231,76,60,0)'),
    fillcolor='rgba(231,76,60,0.15)',
    name='Confidence Range'
))

# Actual
fig.add_trace(go.Scatter(
    x=test_df['ds'], y=test_df['y'],
    name='Actual Sales',
    line=dict(color='#2ecc71', width=2.5)
))

# Predicted
fig.add_trace(go.Scatter(
    x=forecast_test['ds'], y=forecast_test['yhat'],
    name='Predicted Sales',
    line=dict(color='#e74c3c', width=2, dash='dash')
))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Sales (€)',
    height=450,
    legend=dict(x=0, y=1)
)
st.plotly_chart(fig, use_container_width=True)

# ── ACCURACY METRICS ─────────────────────────
st.subheader("Model Accuracy")

rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_test['yhat']))
mae  = mean_absolute_error(test_df['y'], forecast_test['yhat'])
mape = np.mean(np.abs((test_df['y'] - forecast_test['yhat']) / test_df['y'])) * 100

m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"€{rmse:,.0f}", help="Root Mean Square Error")
m2.metric("MAE",  f"€{mae:,.0f}",  help="Mean Absolute Error")
m3.metric("MAPE", f"{mape:.2f}%",  help="Mean Absolute Percentage Error")

if mape < 15:
    st.success(" Excellent Model — MAPE under 15%!")
elif mape < 25:
    st.info("Good Model!")
else:
    st.warning("Model needs improvement")

st.divider()

# ── RAW DATA ─────────────────────────────────
with st.expander("View Raw Data"):
    st.dataframe(store_data.head(100))


# streamlit run app.py
