import os
import time
import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

LOOKBACK = 30
HORIZON = 7
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
LOG_INTERVAL = 10000
MODEL_DIR = 'saved_model'


def load_and_split_data(train_ratio=0.7, test_ratio=0.2):
    logging.info("Загрузка данных из БД...")
    engine = create_engine('postgresql://postgres:1@localhost:5433/weather_db')
    df = pd.read_sql("SELECT city_id, date, tavg FROM weather", engine)

    df = df.dropna(subset=['tavg'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['city_id', 'date']).reset_index(drop=True)
    df['city_id'] = df['city_id'].astype('category').cat.codes
    num_cities = df['city_id'].nunique()

    train_list, test_list, val_list = [], [], []
    for city, city_df in df.groupby('city_id'):
        n = len(city_df)
        i_train = int(n * train_ratio)
        i_test = int(n * (train_ratio + test_ratio))
        train_list.append(city_df.iloc[:i_train])
        test_list.append(city_df.iloc[i_train:i_test])
        val_list.append(city_df.iloc[i_test:])

    train_df = pd.concat(train_list).sort_values(['city_id', 'date']).reset_index(drop=True)
    test_df = pd.concat(test_list).sort_values(['city_id', 'date']).reset_index(drop=True)
    val_df = pd.concat(val_list).sort_values(['city_id', 'date']).reset_index(drop=True)

    logging.info(
        f"Данные загружены: cities={num_cities}, train={len(train_df)}, test={len(test_df)}, val={len(val_df)}")
    return df, train_df, test_df, val_df, num_cities


def create_sequences_with_city(data, lookback=LOOKBACK, horizon=HORIZON, min_val=None, max_val=None):
    logging.info("Подготовка последовательностей...")
    values = data['tavg'].values.astype(np.float32)
    if min_val is None or max_val is None:
        min_val, max_val = values.min(), values.max()
    norm = (values - min_val) / (max_val - min_val)
    city_ids = data['city_id'].values

    X, y, cids = [], [], []
    max_i = len(norm) - lookback - horizon + 1
    for i in range(max_i):
        if not np.all(city_ids[i:i + lookback] == city_ids[i]): continue
        X.append(norm[i:i + lookback])
        y.append(norm[i + lookback:i + lookback + horizon])
        cids.append(city_ids[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    cids = np.array(cids, dtype=np.int64)
    logging.info(f"Seqs: {len(X)}, min_val={min_val:.3f}, max_val={max_val:.3f}")
    return X, y, cids, min_val, max_val


class WeatherDataset(Dataset):
    def __init__(self, X, y, city_ids):
        self.X = torch.tensor(X).unsqueeze(-1)  # (N, lookback, 1)
        self.y = torch.tensor(y)  # (N, horizon)
        self.city_ids = torch.tensor(city_ids, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.city_ids[idx], self.y[idx]


class LSTMModelWithCity(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 num_cities=100, embedding_dim=10, horizon=HORIZON):
        super().__init__()
        self.city_embedding = nn.Embedding(num_cities, embedding_dim)
        self.lstm = nn.LSTM(input_size + embedding_dim, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x, city_ids):
        emb = self.city_embedding(city_ids)
        emb = emb.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.lstm(torch.cat([x, emb], dim=2))
        return self.fc(out[:, -1, :])


def train_model(model, train_loader, test_loader,
                min_val, max_val, epochs=EPOCHS, lr=LR,
                log_interval=LOG_INTERVAL):
    os.makedirs(MODEL_DIR, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for idx, (Xb, cityb, yb) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            pred = model(Xb, cityb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if idx % log_interval == 0:
                logging.info(f"E{epoch} B{idx} avg loss: {total_loss / idx:.4f}")

        model.eval()
        preds_list, trues_list = [], []
        with torch.no_grad():
            for Xb, cityb, yb in test_loader:
                out = model(Xb, cityb).cpu().numpy()
                preds_list.append(out)
                trues_list.append(yb.cpu().numpy())
        preds = np.vstack(preds_list) * (max_val - min_val) + min_val
        trues = np.vstack(trues_list) * (max_val - min_val) + min_val

        mse_all = mean_squared_error(trues.flatten(), preds.flatten())
        rmse_all = np.sqrt(mse_all)
        mae_all = mean_absolute_error(trues.flatten(), preds.flatten())
        r2_all = r2_score(trues.flatten(), preds.flatten())
        logging.info(f"-> Epoch {epoch} ALL Step: MAE={mae_all:.3f}, RMSE={rmse_all:.3f}, R2={r2_all:.4f}")

        for step in range(HORIZON):
            mae_s = mean_absolute_error(trues[:, step], preds[:, step])
            rmse_s = np.sqrt(mean_squared_error(trues[:, step], preds[:, step]))
            r2_s = r2_score(trues[:, step], preds[:, step])
            logging.info(f"   Step {step + 1}: MAE={mae_s:.3f}, RMSE={rmse_s:.3f}, R2={r2_s:.4f}")

        torch.save(model.state_dict(), f"{MODEL_DIR}/epoch{epoch}.pt")

    logging.info("=== Обучение завершено ===")
    return model


def load_saved_model(checkpoint_path, num_cities, embedding_dim=10,
                     input_size=1, hidden_size=64, num_layers=2):
    logging.info(f"Loading saved_model from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)
    model = LSTMModelWithCity(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              num_cities=num_cities,
                              embedding_dim=embedding_dim,
                              horizon=HORIZON)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def forecast_and_true(df, model, min_val, max_val, city_id, date_str):
    df_city = df[df['city_id'] == city_id].sort_values('date').reset_index(drop=True)
    target_date = pd.to_datetime(date_str)
    idx_list = df_city.index[df_city['date'] == target_date].tolist()
    if not idx_list:
        raise ValueError(f"Date {date_str} not found for city {city_id}")
    idx = idx_list[0]
    if idx < LOOKBACK:
        raise ValueError(f"Not enough history (need {LOOKBACK} days) for forecast")

    window_dates = df_city.iloc[idx - LOOKBACK:idx]['date'].values
    window = df_city.iloc[idx - LOOKBACK:idx]['tavg'].values.astype(np.float32)

    missing_mask = np.isnan(window)
    missing_count = np.sum(missing_mask)

    if missing_count > 10:
        raise ValueError(f"Too many missing values ({missing_count}) in the history window. Forecast is not possible.")

    if missing_count > 0:
        mean_val = np.nanmean(window)
        window[missing_mask] = mean_val

    norm_window = (window - min_val) / (max_val - min_val)
    x_input = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    city_tensor = torch.tensor([city_id], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        pred_norm = model(x_input, city_tensor).squeeze().cpu().numpy()
    preds = pred_norm * (max_val - min_val) + min_val

    pred_dates = df_city.iloc[idx:idx + HORIZON]['date'].values
    if len(pred_dates) < HORIZON:
        raise ValueError(f"Not enough future data (need {HORIZON} days) after {date_str}")
    true_vals = df_city.iloc[idx: idx + HORIZON]['tavg'].values.astype(np.float32)

    return window_dates, window, pred_dates, preds, true_vals, missing_count


def print_forecast_table(df, model, min_val, max_val, city_id, date_str):
    win_dates, win_vals, pred_dates, preds, trues = \
        forecast_and_true(df, model, min_val, max_val, city_id, date_str)

    hist_df = pd.DataFrame({'tavg': win_vals}, index=pd.to_datetime(win_dates))
    hist_df.index.name = 'date'

    comp_df = pd.DataFrame({
        'forecast': preds,
        'actual': trues
    }, index=pd.to_datetime(pred_dates))
    comp_df.index.name = 'date'

    print("\n=== Historical data (last {} days) ===".format(LOOKBACK))
    print(hist_df.tail(LOOKBACK).to_string())
    print("\n=== Forecast vs Actual for next {} days ===".format(HORIZON))
    print(comp_df.to_string())


def main():
    df, train_df, test_df, val_df, num_cities = load_and_split_data()

    X_tr, y_tr, c_tr, mn, mx = create_sequences_with_city(train_df)
    X_te, y_te, c_te, _, _ = create_sequences_with_city(test_df, min_val=mn, max_val=mx)
    X_val, y_val, c_val, _, _ = create_sequences_with_city(val_df, min_val=mn, max_val=mx)

    train_loader = DataLoader(WeatherDataset(X_tr, y_tr, c_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(WeatherDataset(X_te, y_te, c_te), batch_size=BATCH_SIZE)
    val_loader = DataLoader(WeatherDataset(X_val, y_val, c_val), batch_size=BATCH_SIZE)

    # saved_model = LSTMModelWithCity(num_cities=num_cities)
    # saved_model = train_model(saved_model, train_loader, test_loader, mn, mx)
    model = load_saved_model('saved_model/epoch7.pt', num_cities)
    print_forecast_table(df, model, mn, mx, city_id=61, date_str='1972-12-24')
    print_forecast_table(df, model, mn, mx, city_id=61, date_str='2004-01-26')
    print_forecast_table(df, model, mn, mx, city_id=61, date_str='2002-02-19')
    print_forecast_table(df, model, mn, mx, city_id=61, date_str='1973-04-06')

    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for Xb, cbd, yb in val_loader:
            out = model(Xb, cbd).cpu().numpy()
            preds_list.append(out)
            trues_list.append(yb.cpu().numpy())
    preds = np.vstack(preds_list) * (mx - mn) + mn
    trues = np.vstack(trues_list) * (mx - mn) + mn

    mse_all = mean_squared_error(trues.flatten(), preds.flatten())
    rmse_all = np.sqrt(mse_all)
    mae_all = mean_absolute_error(trues.flatten(), preds.flatten())
    r2_all = r2_score(trues.flatten(), preds.flatten())
    logging.info(f"Validation ALL Step: MAE={mae_all:.3f}, RMSE={rmse_all:.3f}, R2={r2_all:.4f}")
    for step in range(HORIZON):
        mae_s = mean_absolute_error(trues[:, step], preds[:, step])
        rmse_s = np.sqrt(mean_squared_error(trues[:, step], preds[:, step]))
        r2_s = r2_score(trues[:, step], preds[:, step])
        logging.info(f"   Step {step + 1}: MAE={mae_s:.3f}, RMSE={rmse_s:.3f}, R2={r2_s:.4f}")


if __name__ == '__main__':
    main()
