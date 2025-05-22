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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()


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

    tavg = Column(Float, nullable=True)
    tmin = Column(Float, nullable=True)
    tmax = Column(Float, nullable=True)
    prcp = Column(Float, nullable=True)
    snow = Column(Float, nullable=True)
    wdir = Column(Float, nullable=True)
    wspd = Column(Float, nullable=True)
    wpgt = Column(Float, nullable=True)
    pres = Column(Float, nullable=True)
    tsun = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint('city_id', 'date', name='uq_city_date'),
    )

    city = relationship('City', back_populates='weather')


def init_db():
    logger.info("Инициализация базы данных...")
    Base.metadata.create_all(engine)
    logger.info("База данных и таблицы готовы.")


def get_session():
    return Session()