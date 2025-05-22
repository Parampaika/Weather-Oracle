import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_city_yearly(city_id):
    engine = create_engine('postgresql://postgres:1@localhost:5433/weather_db')
    df = pd.read_sql(
        f"SELECT city_id, date, tavg FROM weather WHERE city_id={city_id}",
        engine
    )
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df_year = df.groupby('year')['tavg'].mean().reset_index()
    return df_year


def create_sequences(df_year, lookback, horizon):
    vals = df_year['tavg'].values.astype(np.float32)
    X, y = [], []
    for i in range(len(vals) - lookback - horizon + 1):
        X.append(vals[i:i + lookback])
        y.append(vals[i + lookback:i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_xgb(X_train, y_train, random_state):
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=random_state
    )
    mor = MultiOutputRegressor(model)
    mor.fit(X_train, y_train)
    return mor


def evaluate(mor, X_test, y_test):
    y_pred = mor.predict(X_test)
    for step in range(y_test.shape[1]):
        mae = mean_absolute_error(y_test[:, step], y_pred[:, step])
        rmse = np.sqrt(mean_squared_error(y_test[:, step], y_pred[:, step]))
        r2 = r2_score(y_test[:, step], y_pred[:, step])
        logging.info(f"Year+{step + 1}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.4f}")
    return y_pred


def split_train_test_years(df_year, lookback, horizon, test_size):
    years = df_year['year'].values
    n_years = len(years)
    min_train = lookback + horizon
    if n_years <= min_train:
        st.error(f"Недостаточно данных: надо минимум {min_train} годов, есть {n_years}.")
        st.stop()
    max_test = n_years - min_train
    desired_test = max(int(n_years * test_size), horizon)
    n_test = min(desired_test, max_test)
    cutoff = n_years - n_test
    return df_year.iloc[:cutoff], df_year.iloc[cutoff:], cutoff


def test_model_for_year(mor, df_year, year, lookback, horizon):
    years = df_year['year'].values
    idx = np.where(years == year)[0]
    if idx.size == 0 or idx[0] < lookback:
        st.warning("Невозможно протестировать для выбранного года.")
        return None
    idx = idx[0]
    window = df_year['tavg'].values[idx - lookback:idx]
    preds = mor.predict(window.reshape(1, -1)).flatten()
    reals = []
    for y in range(year, year + horizon):
        if y in years:
            reals.append(df_year.loc[df_year['year'] == y, 'tavg'].values[0])
        else:
            reals.append(None)
    mask = [r is not None for r in reals]
    y_true = np.array([r for r, m in zip(reals, mask) if m])
    y_pred = np.array([p for p, m in zip(preds, mask) if m])
    metrics = {}
    if y_true.size > 0:
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['R2'] = r2_score(y_true, y_pred)
    return {'years': list(range(year, year + horizon)), 'pred': preds, 'real': reals, 'metrics': metrics}


def plot_forecast(df_year, forecast_res):
    hist_years = df_year['year'].values
    hist_vals = df_year['tavg'].values
    fr_years = forecast_res['years']
    preds = forecast_res['pred']
    reals = forecast_res['real']
    plt.figure()
    plt.plot(hist_years, hist_vals, label='История')
    plt.plot(fr_years, preds, '--o', label='Прогноз')
    real_points = [(y, r) for y, r in zip(fr_years, reals) if r is not None]
    if real_points:
        yrs, vals = zip(*real_points)
        plt.scatter(yrs, vals, c='black', label='Факт')
    plt.xlabel('Год')
    plt.ylabel('Среднегодовая tavg')
    plt.legend()
    st.pyplot(plt)
