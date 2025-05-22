import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates

from sqlalchemy import create_engine
from tkintermapview import TkinterMapView

from predict_module.lstm_low_horisont import (
    forecast_and_true,
    load_saved_model
)

LOOKBACK = 30
HORIZON = 7
MODEL_PATH = "../predict_module/saved_model/epoch7.pt"
NUM_CITIES = 393  # TODO

engine = create_engine('postgresql://postgres:1@localhost:5433/weather_db')
df = pd.read_sql("SELECT city_id, date, tavg FROM weather", engine)
df['date'] = pd.to_datetime(df['date'])

df['category_code'] = df['city_id'].astype('category').cat.codes
original_to_category = df[['city_id', 'category_code']].drop_duplicates()
original_to_category.set_index('city_id', inplace=True)
city_id_mapping = original_to_category['category_code'].to_dict()

min_val, max_val = df['tavg'].min(), df['tavg'].max()
model = load_saved_model(MODEL_PATH, num_cities=NUM_CITIES)

cities_df = pd.read_sql("SELECT id, name, country, latitude, longitude FROM cities", engine)
city_coords = dict(zip(cities_df['id'], zip(cities_df['latitude'], cities_df['longitude'], cities_df['name'])))

country_to_cities = {}
for country, group in cities_df.groupby('country'):
    country_to_cities[country] = list(zip(group['name'], group['id']))

city_id_to_category = df[['city_id', 'category_code']].drop_duplicates()
city_id_to_category.set_index('city_id', inplace=True)
city_id_mapping = city_id_to_category['category_code'].to_dict()


class ForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Прогноз погоды")
        self.root.geometry("1200x800")

        self.control_frame = ttk.Frame(root, width=250)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(self.control_frame, text="Дата (YYYY-MM-DD):").pack(pady=5)
        self.date_entry = ttk.Entry(self.control_frame, width=15)
        self.date_entry.insert(0, "1972-12-24")
        self.date_entry.pack(pady=5)

        ttk.Label(self.control_frame, text="Страна:").pack(pady=5)
        self.country_combobox = ttk.Combobox(
            self.control_frame,
            values=sorted(country_to_cities.keys()),
            state='readonly'
        )
        self.country_combobox.current(0)
        self.country_combobox.pack(pady=5)
        self.country_combobox.bind("<<ComboboxSelected>>", self.update_city_list)

        ttk.Label(self.control_frame, text="Город:").pack(pady=5)
        self.city_combobox = ttk.Combobox(self.control_frame, state='readonly')
        self.city_combobox.pack(pady=5)

        self.forecast_btn = ttk.Button(self.control_frame, text="Прогноз", command=self.make_forecast)
        self.forecast_btn.pack(pady=10)

        self.status_label = ttk.Label(self.control_frame, text="", foreground="blue")
        self.status_label.pack(pady=5)

        self.table_frame = ttk.Frame(self.main_frame)
        self.table_frame.pack(fill=tk.X, padx=10, pady=5)

        self.bottom_plot_frame = ttk.Frame(self.main_frame)
        self.bottom_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.map_frame = ttk.Frame(self.control_frame)
        self.map_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.map_widget = TkinterMapView(self.map_frame, width=250, height=200, corner_radius=0)
        self.map_widget.pack(fill=tk.BOTH, expand=True)
        self.map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")

        self.map_widget.set_position(0, 0)  # Центр мира
        self.map_widget.set_zoom(2)  # Минимальный масштаб

        self.update_city_list()

    def update_city_list(self, event=None):
        selected_country = self.country_combobox.get()
        cities = country_to_cities[selected_country]
        city_names = [f"{name} (ID: {cid})" for name, cid in cities]
        self.city_combobox['values'] = city_names
        self.city_combobox.current(0)

        self.update_selected_city_and_map()

    def make_forecast(self):
        date_str = self.date_entry.get()
        selected_city = self.city_combobox.get()

        try:
            city_id = int(selected_city.split("ID: ")[1].strip(")"))
        except Exception as e:
            self.status_label.config(text="Ошибка: Не удалось извлечь ID города", foreground="red")
            return

        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            self.status_label.config(text="Ошибка: Неверный формат даты", foreground="red")
            return

        if city_id not in city_id_mapping:
            self.status_label.config(text=f"Ошибка: Город с ID {city_id} отсутствует в данных", foreground="red")
            return

        category_code = city_id_mapping[city_id]

        try:
            win_dates, win_vals, pred_dates, preds, trues, missing_count = forecast_and_true(
                df, model, min_val, max_val, category_code, date_str
            )
        except Exception as e:
            self.status_label.config(text=f"Ошибка: {str(e)}", foreground="red")
            return

        if missing_count > 0:
            self.status_label.config(
                text=f"Заполнено {missing_count} пропущенных значений в истории.",
                foreground="green"
            )
        else:
            self.status_label.config(text="Пропущенных значений не было.", foreground="green")

        for widget in self.table_frame.winfo_children():
            widget.destroy()
        for widget in self.bottom_plot_frame.winfo_children():
            widget.destroy()

        self.show_forecast_table(pred_dates, preds, trues)

        self.show_bottom_plot(win_dates, win_vals, pred_dates, preds, trues, pd.to_datetime(date_str))

    def show_bottom_plot(self, win_dates, win_vals, pred_dates, preds, trues, target_date):
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

        # Преобразуем в datetime64
        win_dates = pd.to_datetime(win_dates).to_numpy()
        pred_dates = pd.to_datetime(pred_dates).to_numpy()
        target_date = pd.to_datetime(target_date).to_numpy().astype('datetime64[D]')

        # Используем plot вместо plot_date
        ax.plot(win_dates, win_vals, label='История', color='blue', linestyle='-', marker='')
        ax.plot(pred_dates, preds, label='Прогноз', color='green', linestyle='--', marker='o')
        ax.plot(pred_dates, trues, label='Реальные', color='orange', linestyle='', marker='x')

        # Вертикальная линия на целевой дате
        ax.axvline(target_date, color='gray', linestyle='--', linewidth=1)

        # Форматирование дат на оси X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        fig.autofmt_xdate()

        ax.set_title("История + Прогноз + Реальные значения")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.bottom_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def show_forecast_table(self, pred_dates, preds, trues):
        frame = ttk.Frame(self.table_frame)
        frame.pack(pady=5)

        ttk.Label(frame, text="Прогноз vs Реальные значения", font=("Arial", 12, "bold")).pack()

        tree = ttk.Treeview(frame, columns=['Дата', 'Прогноз', 'Реальное'], show='headings', height=7)
        tree.heading('Дата', text='Дата')
        tree.heading('Прогноз', text='Прогноз')
        tree.heading('Реальное', text='Реальное')
        tree.column('Дата', width=100)
        tree.column('Прогноз', width=100)
        tree.column('Реальное', width=100)

        for date_str, pred, true in zip(pred_dates, preds, trues):
            tree.insert('', 'end', values=(date_str, f"{pred:.2f}", f"{true:.2f}"))
        tree.pack()

    def show_plots(self, win_dates, win_vals):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

        # График истории
        ax.plot(win_dates, win_vals, label='Температура', color='blue')
        ax.set_title(f"Температура за последние {LOOKBACK} дней")
        ax.grid(True)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def show_table(self, df, title):
        frame = ttk.Frame(self.canvas_frame)
        frame.pack(pady=10)

        ttk.Label(frame, text=title, font=("Arial", 12, "bold")).pack()

        tree = ttk.Treeview(frame, columns=list(df.columns), show='headings', height=8)
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        for _, row in df.iterrows():
            tree.insert('', 'end', values=tuple(row))
        tree.pack()

    def update_map(self, city_id):
        if city_id not in city_coords:
            # Установка начальной позиции и масштаба, если город не найден
            self.map_widget.set_position(0, 0)  # Центр мира
            self.map_widget.set_zoom(2)  # Очень маленький масштаб
            self.map_widget.delete_all_marker()
            return

        lat, lon, name = city_coords[city_id]

        # Очистка старых маркеров
        self.map_widget.delete_all_marker()

        # Установка точки и зума
        self.map_widget.set_position(lat, lon)
        self.map_widget.set_zoom(3)  # Меньший масштаб (больше охвата)

        # Красная метка с именем города
        self.map_widget.set_marker(
            lat, lon,
            text=name,
            marker_color_circle="red",
            marker_color_outline="red"
        )

    def update_selected_city_and_map(self):
        selected_city = self.city_combobox.get()
        try:
            city_id = int(selected_city.split("ID: ")[1].strip(")"))
            self.update_map(city_id)
        except Exception as e:
            # Проверяем, существует ли map_widget перед его использованием
            if hasattr(self, 'map_widget'):
                self.map_widget.delete_all_marker()
                self.map_widget.set_position(0, 0)
                self.map_widget.set_zoom(2)


if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastApp(root)
    root.mainloop()
