from sqlalchemy import Integer, String, Float, Column, UniqueConstraint, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

# Таблица 1: среднегодовая температура по городам
class AvgCityTempByYear(Base):
    __tablename__ = 'avg_city_temp_by_year'

    id = Column(Integer, primary_key=True)
    country = Column(String(50), nullable=False)
    city = Column(String(100), nullable=False)
    year = Column(Integer, nullable=False)
    avg_temp = Column(Float, nullable=True)
    avg_range = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint('country', 'city', 'year', name='uq_country_city_year'),
    )

# Таблица 2: среднегодовая температура по странам
class AvgCountryTempByYear(Base):
    __tablename__ = 'avg_country_temp_by_year'

    id = Column(Integer, primary_key=True)
    country = Column(String(50), nullable=False)
    year = Column(Integer, nullable=False)
    avg_temp = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint('country', 'year', name='uq_country_year'),
    )

# Подключение к БД
DB_URL = "postgresql://postgres:1@localhost:5433/weather_db"
engine = create_engine(DB_URL, echo=False)
Session = sessionmaker(bind=engine)

# Создание таблиц
Base.metadata.create_all(engine)

# --- Запрос и заполнение таблицы по городам ---
sql_city = text("""
SELECT
    c.country,
    c.name AS city,
    EXTRACT(YEAR FROM w.date) AS year,
    AVG(w.tavg) AS avg_temp,
    AVG(w.tmax - w.tmin) AS avg_range
FROM
    weather w
JOIN
    cities c ON w.city_id = c.id
WHERE
    w.tavg IS NOT NULL AND w.tmax IS NOT NULL AND w.tmin IS NOT NULL
GROUP BY
    c.country, c.name, EXTRACT(YEAR FROM w.date)
ORDER BY
    c.country, c.name, year;
""")


# --- Запрос и заполнение таблицы по странам ---
sql_country = text("""
SELECT
    c.country,
    EXTRACT(YEAR FROM w.date) AS year,
    AVG(w.tavg) AS avg_temp
FROM
    weather w
JOIN
    cities c ON w.city_id = c.id
WHERE
    w.tavg IS NOT NULL
GROUP BY
    c.country, EXTRACT(YEAR FROM w.date)
ORDER BY
    c.country, year;
""")

with Session() as session:
    # Очистка старых данных
    session.execute(text("DELETE FROM avg_city_temp_by_year"))
    session.execute(text("DELETE FROM avg_country_temp_by_year"))

    # Заполнение по городам
    city_results = session.execute(sql_city).fetchall()
    for country, city, year, avg_temp, avg_range in city_results:
        session.add(AvgCityTempByYear(
            country=country,
            city=city,
            year=int(year),
            avg_temp=avg_temp,
            avg_range=avg_range
        ))

    # Заполнение по странам
    country_results = session.execute(sql_country).fetchall()
    for country, year, avg_temp in country_results:
        session.add(AvgCountryTempByYear(
            country=country,
            year=int(year),
            avg_temp=avg_temp
        ))

    session.commit()
    print(f"✅ Добавлено: {len(city_results)} записей в avg_city_temp_by_year")
    print(f"✅ Добавлено: {len(country_results)} записей в avg_country_temp_by_year")

