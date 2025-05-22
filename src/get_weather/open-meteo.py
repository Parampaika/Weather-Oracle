import math
import logging
import time

from fake_useragent import UserAgent

import pandas as pd
from datetime import date

import openmeteo_requests
import requests_cache
from retry_requests import retry


from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Date,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from src.get_weather.get_city_list import get_cities_coordinates

# --------------------------------------------------
# Настройка логирования
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# 1. Настройки и подключение к базе
# --------------------------------------------------
# DB_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/weather_db')
DB_URL = 'postgresql://postgres:1@localhost:5433/weather_db'
engine = create_engine(DB_URL, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# --------------------------------------------------
# 2. Описание моделей ORM
# --------------------------------------------------
class City(Base):
    __tablename__ = 'cities'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint('name', 'country', name='uq_city_country'),
    )

    weather = relationship('Weather', back_populates='city', cascade='all, delete-orphan')

class Weather(Base):
    __tablename__ = 'weather'

    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey('cities.id', ondelete='CASCADE'), nullable=False)
    date = Column(Date, nullable=False)

    # Основные переменные погоды
    weather_code = Column(Integer, nullable=True)  # Добавлено nullable=True
    temperature_2m_mean = Column(Float, nullable=True)  # Добавлено nullable=True
    temperature_2m_max = Column(Float, nullable=True)  # Добавлено nullable=True
    temperature_2m_min = Column(Float, nullable=True)  # Добавлено nullable=True
    apparent_temperature_mean = Column(Float, nullable=True)  # Добавлено nullable=True
    apparent_temperature_max = Column(Float, nullable=True)  # Добавлено nullable=True
    apparent_temperature_min = Column(Float, nullable=True)  # Добавлено nullable=True
    sunrise = Column(Integer, nullable=True)  # Добавлено nullable=True
    sunset = Column(Integer, nullable=True)  # Добавлено nullable=True
    daylight_duration = Column(Float, nullable=True)  # Добавлено nullable=True
    sunshine_duration = Column(Float, nullable=True)  # Добавлено nullable=True
    precipitation_sum = Column(Float, nullable=True)  # Добавлено nullable=True
    rain_sum = Column(Float, nullable=True)  # Добавлено nullable=True
    snowfall_sum = Column(Float, nullable=True)  # Добавлено nullable=True
    precipitation_hours = Column(Float, nullable=True)  # Добавлено nullable=True
    wind_speed_10m_max = Column(Float, nullable=True)  # Добавлено nullable=True
    wind_gusts_10m_max = Column(Float, nullable=True)  # Добавлено nullable=True
    wind_direction_10m_dominant = Column(Float, nullable=True)  # Добавлено nullable=True
    shortwave_radiation_sum = Column(Float, nullable=True)  # Добавлено nullable=True
    et0_fao_evapotranspiration = Column(Float, nullable=True)  # Добавлено nullable=True
    cloud_cover_mean = Column(Float, nullable=True)  # Добавлено nullable=True
    dew_point_2m_mean = Column(Float, nullable=True)  # Добавлено nullable=True
    pressure_msl_mean = Column(Float, nullable=True)  # Добавлено nullable=True
    relative_humidity_2m_mean = Column(Float, nullable=True)  # Добавлено nullable=True
    surface_pressure_mean = Column(Float, nullable=True)  # Добавлено nullable=True
    visibility_mean = Column(Float, nullable=True)  # Добавлено nullable=True
    wet_bulb_temperature_2m_mean = Column(Float, nullable=True)  # Добавлено nullable=True

    __table_args__ = (
        UniqueConstraint('city_id', 'date', name='uq_city_date'),
    )

    city = relationship('City', back_populates='weather')

# --------------------------------------------------
# 3. Инициализация базы данных
# --------------------------------------------------
def init_db():
    logger.info("Инициализация базы данных...")
    Base.metadata.create_all(engine)
    logger.info("База данных и таблицы готовы.")

# --------------------------------------------------
# 4. Настройка клиента Open-Meteo с кешем и retry
# --------------------------------------------------
ua = UserAgent()
headers = {'User-Agent': ua.random}

cache_session = requests_cache.CachedSession('.cache', expire_after=86400)
cache_session.headers.update(headers)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Перечень запрашиваемых переменных (дневные)
DAILY_VARS = [
    "weather_code", "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min",
    "sunrise", "sunset", "daylight_duration", "sunshine_duration",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration", "cloud_cover_mean",
    "dew_point_2m_mean", "pressure_msl_mean", "relative_humidity_2m_mean",
    "surface_pressure_mean", "visibility_mean", "wet_bulb_temperature_2m_mean"
]

# --------------------------------------------------
# 5. Функция запроса и обработки данных погоды
# --------------------------------------------------
def fetch_daily_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    time.sleep(60)
    logger.info(f"Запрос данных с {start} по {end} для координат: ({lat}, {lon})")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": DAILY_VARS,
        "timezone": "UTC",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    # Создаем диапазон дат с использованием pd.date_range
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
    }

    # Собираем данные в словарь
    for idx, var in enumerate(DAILY_VARS):
        daily_data[var] = daily.Variables(idx).ValuesAsNumpy()

    # Логируем количество записей для переменной 'weather_code'
    logger.info(f"Получено {len(daily_data['weather_code'])} записей погоды")

    # Создаем DataFrame
    df = pd.DataFrame(daily_data)

    return df


def clean_nan_values(record: dict) -> dict:
    """Заменяет NaN и inf значения на None для корректной вставки в БД."""
    return {
        k: None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
        for k, v in record.items()
    }

# --------------------------------------------------
# 6. Основной процесс сбора и сохранения данных
# --------------------------------------------------
def main(countries: list[str], cities_file: str, top_n: int):
    init_db()
    session = Session()

    start_date = date(1940, 1, 1)
    end_date = date(2024, 12, 31)

    for country in countries:
        logger.info(f"Обработка страны: {country}")
        cities_df = get_cities_coordinates(cities_file, country, top_n=top_n)
        logger.info(f"Найдено {len(cities_df)} городов для страны {country}")

        for _, row in cities_df.iterrows():
            logger.info(f"Обработка города: {row['city']} ({row['latitude']}, {row['longitude']})")

            # Проверяем, существует ли город в базе
            city = session.query(City).filter_by(
                name=row['city'], country=country).first()

            if not city:
                # Если города нет, создаём новый объект
                city = City(
                    name=row['city'], country=country,
                    latitude=row['latitude'], longitude=row['longitude'],
                )
                session.add(city)  # Добавляем новый город в сессию

            session.flush()  # Обеспечиваем, чтобы id города был доступен для использования в дальнейшем

            weather_df = fetch_daily_weather(
                lat=row['latitude'], lon=row['longitude'],
                start=start_date, end=end_date,
            )

            for w in weather_df.itertuples(index=False):
                data = clean_nan_values(w._asdict())
                record = Weather(**data, city_id=city.id)
                session.add(record)
                try:
                    session.flush()
                except IntegrityError:
                    session.rollback()
                    session.begin()
            session.commit()

            logger.info(f"Данные для {city.name}, {country} сохранены в базе.")

    session.close()
    logger.info("Завершение процесса.")


if __name__ == '__main__':
    countries_to_fetch = ['Russia', 'Germany', 'Brazil', 'Canada']
    cities_csv_path = '../../data/archive/worldcities.csv'
    main(countries_to_fetch, cities_csv_path, 50)