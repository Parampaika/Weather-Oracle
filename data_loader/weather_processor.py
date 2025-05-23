import concurrent.futures
import logging
import math
import warnings
from datetime import date, datetime

import pandas as pd
from meteostat import Point, Daily
from sqlalchemy.exc import IntegrityError

from database import City, Weather, get_session, init_db
from get_city_list import get_cities_coordinates

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_daily_weather(lat: float, lon: float, start: date, end: date, timeout_sec=30) -> pd.DataFrame:
    logger.info(f"Запрос данных с {start} по {end} для координат: ({lat}, {lon})")

    def fetch_data():
        point = Point(lat, lon)
        return Daily(point, start, end).fetch()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_data)
        try:
            data = future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Запрос превысил таймаут ({timeout_sec} секунд). Пропуск.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка при запросе данных: {e}")
            return pd.DataFrame()

    if data.empty:
        logger.warning("Нет данных для этой точки за указанный период.")
        return pd.DataFrame()

    if 'time' not in data.columns:
        data = data.reset_index()

    if 'time' not in data.columns:
        logger.warning("Колонка 'time' отсутствует даже после reset_index()")
        return pd.DataFrame()

    df = data[['time', 'tavg', 'tmin', 'tmax', 'prcp', 'snow',
               'wdir', 'wspd', 'wpgt', 'pres', 'tsun']].copy()
    df.rename(columns={'time': 'date'}, inplace=True)

    return df


def clean_nan_values(record: dict) -> dict:
    return {
        k: None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
        for k, v in record.items()
    }


def main(countries: list[str], cities_file: str, top_n: int):
    init_db()
    session = get_session()

    start_date = datetime(1950, 1, 1)
    end_date = datetime(2024, 12, 31)

    for country in countries:
        logger.info(f"Обработка страны: {country}")
        cities_df = get_cities_coordinates(cities_file, country, top_n=top_n)
        logger.info(f"Найдено {len(cities_df)} городов для страны {country}")

        for _, row in cities_df.iterrows():
            logger.info(f"Обработка города: {row['city']} ({row['latitude']}, {row['longitude']})")

            city = session.query(City).filter_by(
                name=row['city'], country=country).first()

            if not city:
                city = City(
                    name=row['city'], country=country,
                    latitude=row['latitude'], longitude=row['longitude'],
                )
                session.add(city)
                session.flush()
            else:
                count = session.query(Weather).filter_by(city_id=city.id).count()
                if count > 0:
                    logger.info(f"Город {city.name}, {country} уже обработан: {count} записей. Пропуск.")
                    continue

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
    countries_to_fetch = ['Russia', 'Germany', 'Brazil', 'Canada', 'United States', 'Japan']
    cities_csv_path = '../data/archive/worldcities.csv'
    main(countries_to_fetch, cities_csv_path, 100)