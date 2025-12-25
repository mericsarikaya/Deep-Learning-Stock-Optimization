# =============================================================================
# SAF CNN + LSTM 
# TimeSeriesSplit + EPOCH + GENEL CV + FINAL TEST
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
BATCH_SIZE = 16
EPOCHS = 50
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
# FINAL TEST AYIR (DATA LEAKAGE YOK)
# =============================================================================
test_size = int(len(df) * 0.15)
df_trainval = df.iloc[:-test_size]
df_test     = df.iloc[-test_size:]

# =============================================================================
# DATASET
# =============================================================================
class CNNLSTMDataset(Dataset):
    def __init__(self, data, window):
        self.returns = data["log_return"].values.astype(np.float32)
        self.close   = data["close"].values.astype(np.float32)
        self.window  = window

    def __len__(self):
        return len(self.returns) - self.window

    def __getitem__(self, idx):
        x = self.returns[idx:idx + self.window]
        y = self.returns[idx + self.window]
        ref = self.close[idx + self.window - 1]
        return torch.tensor(x).unsqueeze(0), torch.tensor(y), ref

# =============================================================================
# MODEL: CNN + LSTM
# =============================================================================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.relu  = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# =============================================================================
# TimeSeriesSplit – CV
# =============================================================================
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

cv_rmse, cv_mae, cv_mape, cv_r2 = [], [], [], []

print("\n===== CNN + LSTM TimeSeriesSplit BAŞLADI =====\n")

for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval), 1):

    print(f"\n================ FOLD {fold} ================\n")

    train_ds = CNNLSTMDataset(df_trainval.iloc[train_idx], WINDOW_SIZE)
    val_ds   = CNNLSTMDataset(df_trainval.iloc[val_idx], WINDOW_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = CNN_LSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ================== TRAIN + VAL (EPOCH BAZLI) ==================
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                val_loss += criterion(model(x.to(device)), y.to(device)).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # ================== VALIDATION METRİK ==================
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
    refs   = np.array(refs)

    true_prices = refs * np.exp(y_true)
    pred_prices = refs * np.exp(y_pred)

    rmse = np.sqrt(mean_squared_error(true_prices, pred_prices))
    mae  = mean_absolute_error(true_prices, pred_prices)
    mape = np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100
    r2   = r2_score(true_prices, pred_prices)

    cv_rmse.append(rmse)
    cv_mae.append(mae)
    cv_mape.append(mape)
    cv_r2.append(r2)

    print(f"\nVAL RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: %{mape:.2f} | R2: {r2:.4f}")

# =============================================================================
# GENEL CV SONUÇLARI
# =============================================================================
print("\n===== GENEL CROSS-VALIDATION SONUÇLARI (CNN + LSTM) =====")
print(f"RMSE : {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
print(f"MAE  : {np.mean(cv_mae):.4f}")
print(f"MAPE : %{np.mean(cv_mape):.2f}")
print(f"R2   : {np.mean(cv_r2):.4f}")

# =============================================================================
# FINAL TEST (HİÇ GÖRÜLMEMİŞ VERİ)
# =============================================================================
test_loader = DataLoader(
    CNNLSTMDataset(df_test, WINDOW_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

model.eval()
y_true, y_pred, refs = [], [], []

with torch.no_grad():
    for x, y, ref in test_loader:
        preds = model(x.to(device)).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)
        refs.extend(ref.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
refs   = np.array(refs)

true_prices = refs * np.exp(y_true)
pred_prices = refs * np.exp(y_pred)

print("\n===== FINAL TEST SONUÇLARI (CNN + LSTM) =====")
print(f"RMSE : {np.sqrt(mean_squared_error(true_prices, pred_prices)):.4f}")
print(f"MAE  : {mean_absolute_error(true_prices, pred_prices):.4f}")
print(f"MAPE : %{np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100:.2f}")
print(f"R2   : {r2_score(true_prices, pred_prices):.4f}")
