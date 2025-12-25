# =============================================================================
# SAF LSTM + ATTENTION – TimeSeriesSplit + LEAK-FREE FINAL TEST
# =============================================================================

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# =============================================================================
# AYARLAR
# =============================================================================
SEED = 42
WINDOW_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
N_SPLITS = 5

DATA_PATH = r"C:\Users\meric\Desktop\kodlama\python\yapay_sinir_proje\AAPL.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# =============================================================================
# VERİ OKUMA
# =============================================================================
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df.dropna(inplace=True)

# =============================================================================
# FINAL TEST AYIR (EN BAŞTA)
# =============================================================================
test_size = int(len(df) * 0.15)
df_trainval = df.iloc[:-test_size]
df_test     = df.iloc[-test_size:]

# =============================================================================
# DATASET
# =============================================================================
class LSTMAttentionDataset(Dataset):
    def __init__(self, data, window):
        self.returns = data["log_return"].values.astype(np.float32)
        self.close = data["close"].values.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.returns) - self.window

    def __getitem__(self, idx):
        x = self.returns[idx:idx + self.window]
        y = self.returns[idx + self.window]
        ref = self.close[idx + self.window - 1]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y), ref

# =============================================================================
# ATTENTION
# =============================================================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.score(lstm_out), dim=1)
        return torch.sum(weights * lstm_out, dim=1)

# =============================================================================
# MODEL
# =============================================================================
class LSTM_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=4,
            batch_first=True,
            dropout=0.2
        )
        self.attention = Attention(128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        context = self.attention(out)
        out = self.output(context)
        return out.view(-1)

# =============================================================================
# TimeSeriesSplit – CV (SADECE DEĞERLENDİRME)
# =============================================================================
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

cv_rmse, cv_mae, cv_mape, cv_r2, cv_mse = [], [], [], [], []

print("\n===== TimeSeriesSplit CV BAŞLADI =====\n")

for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval), 1):

    print(f"\n===== FOLD {fold} =====")

    train_ds = LSTMAttentionDataset(df_trainval.iloc[train_idx], WINDOW_SIZE)
    val_ds   = LSTMAttentionDataset(df_trainval.iloc[val_idx], WINDOW_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTM_Attention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ---------------- TRAIN ----------------
    for epoch in range(EPOCHS):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # ---------------- VALIDATION METRICS ----------------
    model.eval()
    y_true, y_pred, refs = [], [], []

    with torch.no_grad():
        for x, y, ref in val_loader:
            preds = model(x.to(device)).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)
            refs.extend(ref.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    refs = np.array(refs)

    true_prices = refs * np.exp(y_true)
    pred_prices = refs * np.exp(y_pred)

    mse  = mean_squared_error(true_prices, pred_prices)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(true_prices, pred_prices)
    mape = np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100
    r2   = r2_score(true_prices, pred_prices)

    cv_mse.append(mse)
    cv_rmse.append(rmse)
    cv_mae.append(mae)
    cv_mape.append(mape)
    cv_r2.append(r2)

    print(
        f"MSE: {mse:.6f} | "
        f"RMSE: {rmse:.6f} | "
        f"MAE: {mae:.6f} | "
        f"MAPE: {mape:.6f} | "
        f"R2: {r2:.6f}"
    )

# =============================================================================
# GENEL CV SONUÇLARI
# =============================================================================
print("\n===== GENEL CV SONUÇLARI (LSTM + ATTENTION) =====")
print(f"MSE  : {np.mean(cv_mse):.6f}")
print(f"RMSE : {np.mean(cv_rmse):.6f} ± {np.std(cv_rmse):.6f}")
print(f"MAE  : {np.mean(cv_mae):.6f}")
print(f"MAPE : {np.mean(cv_mape):.6f}")
print(f"R2   : {np.mean(cv_r2):.6f}")

# =============================================================================
# FINAL MODEL (SIFIRDAN – LEAK-FREE)
# =============================================================================
print("\n===== FINAL MODEL EĞİTİMİ =====")

final_train_ds = LSTMAttentionDataset(df_trainval, WINDOW_SIZE)
final_train_loader = DataLoader(final_train_ds, batch_size=BATCH_SIZE, shuffle=False)

final_model = LSTM_Attention().to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    final_model.train()
    for x, y, _ in final_train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(final_model(x), y)
        loss.backward()
        optimizer.step()

# =============================================================================
# FINAL TEST (HİÇ GÖRÜLMEMİŞ VERİ)
# =============================================================================
test_loader = DataLoader(
    LSTMAttentionDataset(df_test, WINDOW_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

final_model.eval()
y_true, y_pred, refs = [], [], []

with torch.no_grad():
    for x, y, ref in test_loader:
        preds = final_model(x.to(device)).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)
        refs.extend(ref.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
refs = np.array(refs)

true_prices = refs * np.exp(y_true)
pred_prices = refs * np.exp(y_pred)

mse  = mean_squared_error(true_prices, pred_prices)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(true_prices, pred_prices)
mape = np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100
r2   = r2_score(true_prices, pred_prices)

print("\n===== FINAL TEST SONUÇLARI (LEAK-FREE) =====")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"MAPE : {mape:.6f}")
print(f"R2   : {r2:.6f}")
