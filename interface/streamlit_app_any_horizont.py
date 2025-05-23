import logging
import os
import sys

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from predict_module.yearly_boosting import create_sequences, train_xgb, \
    test_model_for_year, plot_forecast, split_train_test_years
from sqlalchemy import create_engine

sys.path.insert(0, r"C:\Users\danii\Documents\Study\Mag_2\БД\open-meteo")

# === Функции работы с БД ===

load_dotenv()

DB_URI = os.getenv("DB_URL")

@st.cache_data
def get_city_name(city_id):
    engine = create_engine(DB_URI)
    query = f"SELECT name, country FROM cities WHERE id = {city_id};"
    df = pd.read_sql(query, engine)
    if df.empty:
        return None
    return f"{df.loc[0, 'name']}, {df.loc[0, 'country']}"


@st.cache_data
def load_city_yearly(city_id):
    engine = create_engine(DB_URI)
    df = pd.read_sql(
        f"SELECT date, tavg FROM weather WHERE city_id={city_id};",
        engine
    )
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df_year = df.groupby('year')['tavg'].mean().reset_index()
    return df_year


# === Настройки логирования ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === UI Streamlit ===
st.title("Прогноз средней годовой температуры")

with st.sidebar:
    city_id = st.number_input("City ID", value=1, step=1)
    city_name = get_city_name(city_id)
    if city_name:
        st.write(f"**Город:** {city_name}")
    lookback = st.slider("Lookback (лет)", min_value=10, max_value=70, value=50)
    horizon = st.slider("Horizon (лет)", min_value=1, max_value=10, value=2)
    test_size = 0.2
    random_state = 42
    test_year = st.number_input("Год для теста", value=2022, step=1)
    run_button = st.button("Запустить прогноз")

if run_button:
    df_year = load_city_yearly(city_id)
    if df_year.empty:
        st.error("Данных для выбранного города нет.")
        st.stop()
    st.write(f"Данные за {len(df_year)} лет, с {df_year['year'].min()} по {df_year['year'].max()}")

    df_train, df_test, cutoff = split_train_test_years(df_year, lookback, horizon, test_size)
    X_train, y_train = create_sequences(df_train, lookback, horizon)
    st.write(f"Train: {len(df_train)} лет → {X_train.shape[0]} окон")

    model = train_xgb(X_train, y_train, random_state)
    res = test_model_for_year(model, df_year, test_year, lookback, horizon)
    if res:
        if res['metrics']:
            st.write("### Метрики:")
            st.json(res['metrics'])
        st.write("### График прогноза и факта:")
        plot_forecast(df_year, res)
