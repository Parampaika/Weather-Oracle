import pandas as pd


def get_cities_coordinates(file_path: str, country: str = 'Russia', top_n: int = 50) -> pd.DataFrame:
    """
    Возвращает DataFrame с топ-N самыми населенными городами указанной страны и их координатами.

    Параметры:
        file_path (str): Путь к файлу CSV (формат worldcities)
        country (str): Название страны для фильтрации (по умолчанию 'Russia')
        top_n (int): Сколько городов с наибольшим населением вернуть

    Возвращает:
        pd.DataFrame: Колонки ['country', 'city', 'latitude', 'longitude']
    """
    df = pd.read_csv(file_path)
    df = df[df['country'] == country]
    df = df[df['population'].notnull()]  # исключаем города без данных по населению
    df = df.sort_values(by='population', ascending=False).head(top_n)

    result = df[['country', 'city_ascii', 'lat', 'lng']].rename(columns={
        'city_ascii': 'city',
        'lat': 'latitude',
        'lng': 'longitude'
    })
    return result.reset_index(drop=True)

