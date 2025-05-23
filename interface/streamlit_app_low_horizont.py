import os

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from sqlalchemy import create_engine
from datetime import timedelta

from predict_module.lstm_low_horisont import load_saved_model, forecast_and_true


load_dotenv()

LOOKBACK = 30
HORIZON = 7
MODEL_PATH = "../predict_module/saved_model/epoch7.pt"
DB_URI = os.getenv("DB_URL")
NUM_CITIES = 393

@st.cache_resource
def get_engine(uri):
    return create_engine(uri)

@st.cache_data(ttl=3600)
def load_data(_engine):
    weather_df = pd.read_sql(
        "SELECT city_id, date, tavg FROM weather ORDER BY date", _engine
    )
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    weather_df['category_code'] = weather_df['city_id'].astype('category').cat.codes
    mapping = (
        weather_df[['city_id', 'category_code']]
        .drop_duplicates()
        .set_index('city_id')['category_code']
        .to_dict()
    )
    return weather_df, mapping

@st.cache_data(ttl=3600)
def load_cities(_engine):
    cities = pd.read_sql(
        "SELECT id, name, country, latitude, longitude FROM cities ORDER BY name", _engine
    )
    country_to_cities = cities.groupby('country') \
        .apply(lambda grp: list(zip(grp['name'], grp['id']))) \
        .to_dict()
    coords = dict(zip(
        cities['id'], zip(cities['latitude'], cities['longitude'], cities['name'])
    ))
    return country_to_cities, coords

@st.cache_resource
def load_model_resource(path):
    return load_saved_model(path, num_cities=NUM_CITIES)

engine = get_engine(DB_URI)
weather_df, city_id_mapping = load_data(engine)
country_to_cities, city_coords = load_cities(engine)
model = load_model_resource(MODEL_PATH)
min_val, max_val = weather_df['tavg'].min(), weather_df['tavg'].max()

st.title("⚡ Weather Forecast Dashboard")

country = st.selectbox("Страна", options=sorted(country_to_cities.keys()))
cities = country_to_cities[country]
print(country_to_cities, cities)
city_entry = st.selectbox(
    "Город", options=[f"{n} (ID: {i})" for n, i in cities]
)
city_name, city_id = city_entry.rsplit(' (ID: ', 1)
city_id = int(city_id.rstrip(')'))

city_dates = weather_df.loc[weather_df['city_id'] == city_id, 'date']
if city_dates.empty:
    st.error(f"Нет данных для города {city_name} (ID: {city_id}).")
    st.stop()
min_date = city_dates.min().date() + timedelta(days=LOOKBACK)
max_date = city_dates.max().date() - timedelta(days=HORIZON - 1)
if min_date > max_date:
    st.error("Мало данных для этого города, увеличьте диапазон истории или уменьшите горизонт прогноза.")
    st.stop()

st.markdown(f"**Доступный диапазон дат**: с {min_date} по {max_date}")

date_input = st.date_input(
    "Дата прогноза", value=max_date,
    min_value=min_date, max_value=max_date
)

if st.button("Сделать прогноз"):
    date_str = date_input.strftime("%Y-%m-%d")
    # Получаем category_code для прогноза
    code = city_id_mapping.get(city_id)
    if code is None:
        st.error(f"Город с ID {city_id} отсутствует в данных.")
        st.stop()

    try:
        win_dates, win_vals, pred_dates, preds, trues, missing = \
            forecast_and_true(
                weather_df, model, min_val, max_val,
                code, date_str
            )
    except ValueError as e:
        st.error(f"Ошибка в forecast_and_true: {e}")
        st.stop()

    if missing > 0:
        st.success(f"Заполнено {missing} пропущенных значений в истории.")
    else:
        st.info("Пропущенных значений не было.")

    df_pred = pd.DataFrame({
        "Дата": pd.to_datetime(pred_dates),
        "Прогноз": preds,
        "Реальные": trues
    })
    df_pred["Дата"] = df_pred["Дата"].dt.date
    st.subheader("Прогноз vs Реальные значения")
    st.dataframe(
        df_pred.style.format({"Прогноз": "{:.2f}", "Реальные": "{:.2f}"}),
        use_container_width=True
    )

    df_hist = pd.DataFrame({
        "Дата": pd.to_datetime(win_dates),
        "Температура": win_vals,
        "Тип": "История"
    })
    df_f = (
        df_pred.rename(columns={"Дата": "Дата", "Прогноз": "Температура"})
        .assign(Тип="Прогноз")
    )
    df_t = (
        df_pred.rename(columns={"Дата": "Дата", "Реальные": "Температура"})
        .assign(Тип="Реальные")
    )
    df_plot = pd.concat([df_hist, df_f, df_t], ignore_index=True)

    fig = px.line(
        df_plot, x="Дата", y="Температура", color="Тип",
        title=f"История и прогноз для {city_name} на {date_str}"
    )
    fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"))
    st.plotly_chart(fig, use_container_width=True)

    lat, lon, name = city_coords.get(city_id, (None, None, None))
    if lat is not None:
        st.subheader(f"Расположение: {name}")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
    else:
        st.warning("Координаты города не найдены.")

# Запуск:
# streamlit run streamlit_app_low_horizont.py
