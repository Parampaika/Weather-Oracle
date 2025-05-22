import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import r2_score

# === 1. Загрузка и разбиение данных ===
def load_and_split_data(filepath, train_end='2015-12-31'):
    df = pd.read_csv(filepath, index_col='Unnamed: 0')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    train_df = df[df['date'] <= train_end]
    test_df = df[df['date'] > train_end]
    return train_df, test_df

# === 2. Подготовка входов для LSTM ===
def create_sequences(data, lookback=30, min_val=None, max_val=None):
    values = data['tavg'].values.astype(np.float32)
    if min_val is None or max_val is None:
        min_val = values.min()
        max_val = values.max()
    values_norm = (values - min_val) / (max_val - min_val)
    X, y = [], []
    for i in range(len(values_norm) - lookback):
        X.append(values_norm[i:i+lookback])
        y.append(values_norm[i+lookback])
    return np.array(X), np.array(y), min_val, max_val

# === 3. Dataset для PyTorch ===
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(-1)
        self.y = torch.tensor(y).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === 4. Модель LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# === 5. Обучение модели ===
def train_model(model, dataloader, epochs=20, lr=0.001):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    return model

# === 6. Сохранение и загрузка модели ===
def save_model(path, model, min_val, max_val):
    torch.save({
        'model_state_dict': model.state_dict(),
        'min_val': min_val,
        'max_val': max_val
    }, path)

def load_model(path, input_size=1, hidden_size=64, num_layers=2):
    checkpoint = torch.load(path, weights_only=False)
    model = LSTMModel(input_size, hidden_size, num_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['min_val'], checkpoint['max_val']

# === 7. Функция тестирования по дате ===
def test_model_for_date(df, date_str, lookback=30, model_path='lstm_weather_model.pt'):
    model, min_val, max_val = load_model(model_path)
    df = df.sort_values('date').reset_index(drop=True)
    idx_list = df.index[df['date'] == pd.to_datetime(date_str)].tolist()
    if not idx_list:
        raise ValueError(f"Date {date_str} not found in dataframe.")
    idx = idx_list[0]
    if idx < lookback:
        raise ValueError("Недостаточно данных для формирования окна.")

    window_df = df.iloc[idx - lookback: idx]
    true_temp = df.iloc[idx]['tavg']

    seq = window_df['tavg'].values.astype(np.float32)
    seq_norm = (seq - min_val) / (max_val - min_val)
    x = torch.tensor(seq_norm).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        pred_norm = model(x).numpy()[0][0]
    pred = pred_norm * (max_val - min_val) + min_val

    print(window_df[['date', 'tavg']].to_string(index=False))
    print(f"\nDate: {date_str} | True: {true_temp:.2f} | Pred: {pred:.2f}")

# === 8. Функция оценки на тестовом наборе ===
def evaluate_on_test(model_path, test_df, lookback=30, batch_size=64):
    model, min_val, max_val = load_model(model_path)
    test_df = test_df.dropna(subset=['tavg']).sort_values('date').reset_index(drop=True)
    X_test, y_test_norm, _, _ = create_sequences(test_df, lookback=lookback, min_val=min_val, max_val=max_val)

    test_dataset = WeatherDataset(X_test, y_test_norm)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    preds, trues = [], []
    with torch.no_grad():
        for batch_X, batch_y_norm in test_loader:
            out_norm = model(batch_X).numpy().flatten()
            out = out_norm * (max_val - min_val) + min_val
            true_vals = batch_y_norm.numpy().flatten() * (max_val - min_val) + min_val
            preds.extend(out)
            trues.extend(true_vals)

    preds = np.array(preds)
    trues = np.array(trues)

    # фильтрация NaN и Inf
    mask = np.isfinite(preds) & np.isfinite(trues)
    preds = preds[mask]
    trues = trues[mask]

    mae = np.mean(np.abs(trues - preds))
    mse = np.mean((trues - preds)**2)
    rmse = np.sqrt(mse)

    nonzero = trues != 0
    if nonzero.any():
        mape = np.mean(np.abs((trues[nonzero] - preds[nonzero]) / trues[nonzero])) * 100
    else:
        mape = np.nan

    smape = np.mean(2 * np.abs(preds - trues) / (np.abs(trues) + np.abs(preds))) * 100
    r2 = r2_score(trues, preds)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"SMAPE: {smape:.2f}%")
    print(f"R2: {r2:.4f}")

# === 9. main ===
def main():
    filepath = 'df1.csv'
    lookback = 30
    batch_size = 64
    epochs = 10
    model_path = 'lstm_weather_model.pt'
    test_date = '1950-01-31 00:00:00'

    train_df, test_df = load_and_split_data(filepath)
    train_df = train_df.dropna(subset=['tavg'])

    X, y, min_val, max_val = create_sequences(train_df, lookback=lookback)
    train_dataset = WeatherDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # saved_model = LSTMModel()
    # saved_model = train_model(saved_model, train_loader, epochs=epochs)
    #
    # save_model(model_path, saved_model, min_val, max_val)
    # print(f"[INFO] Model saved to {model_path}")

    test_model_for_date(df=train_df, date_str=test_date, lookback=lookback, model_path=model_path)

    print("\n[INFO] Evaluation on test dataset:")
    evaluate_on_test(model_path, test_df, lookback=lookback, batch_size=batch_size)

if __name__ == '__main__':
    main()
