# =============================================================================
# GERÇEK LSTM + ANN HİBRİT MODEL (LEAK-FREE)
# TimeSeriesSplit + DOĞRU FINAL TEST
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
# FINAL TEST AYIR (LEAK KORUMASI)
# =============================================================================
test_size = int(len(df) * 0.15)
df_trainval = df.iloc[:-test_size]
df_test     = df.iloc[-test_size:]

# =============================================================================
# DATASET
# =============================================================================
class LSTMHybridDataset(Dataset):
    def __init__(self, data, window):
        self.returns = data["log_return"].values.astype(np.float32)
        self.close   = data["close"].values.astype(np.float32)
        self.window  = window

    def __len__(self):
        return len(self.returns) - self.window

    def __getitem__(self, idx):
        seq = self.returns[idx:idx + self.window]
        target = self.returns[idx + self.window]
        ref = self.close[idx + self.window - 1]
        return torch.tensor(seq).unsqueeze(-1), torch.tensor(target), ref

# =============================================================================
# MODEL
# =============================================================================
class LSTM_ANN(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- LSTM (2 layer) ----
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,      # <<< 3 → 2
            batch_first=True,
            dropout=0.2
        )

        # ---- ANN (2 hidden layer) ----
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1)

# =============================================================================
# TimeSeriesSplit – CROSS VALIDATION
# =============================================================================
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

cv_mse, cv_rmse, cv_mae, cv_mape, cv_r2 = [], [], [], [], []

print("\n===== TimeSeriesSplit EĞİTİM BAŞLADI =====")

for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval), 1):

    print(f"\n--- FOLD {fold} ---")

    train_ds = LSTMHybridDataset(df_trainval.iloc[train_idx], WINDOW_SIZE)
    val_ds   = LSTMHybridDataset(df_trainval.iloc[val_idx], WINDOW_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTM_ANN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ---------------- TRAIN ----------------
    for _ in range(EPOCHS):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # ---------------- VALIDATION ----------------
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

    mse  = mean_squared_error(true_prices, pred_prices)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(true_prices, pred_prices)
    mape = np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100
    r2   = r2_score(true_prices, pred_prices)

    print(
        f"MSE: {mse:.6f} | RMSE: {rmse:.6f} | "
        f"MAE: {mae:.6f} | MAPE: %{mape:.6f} | R2: {r2:.6f}"
    )

# =============================================================================
# FINAL TEST – GERÇEK PERFORMANS
# =============================================================================
test_loader = DataLoader(
    LSTMHybridDataset(df_test, WINDOW_SIZE),
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

true_prices = np.array(refs) * np.exp(np.array(y_true))
pred_prices = np.array(refs) * np.exp(np.array(y_pred))

print("\n===== FINAL TEST SONUÇLARI (LEAK FREE) =====")
print(f"MSE  : {mean_squared_error(true_prices, pred_prices):.6f}")
print(f"RMSE : {np.sqrt(mean_squared_error(true_prices, pred_prices)):.6f}")
print(f"MAE  : {mean_absolute_error(true_prices, pred_prices):.6f}")
print(f"MAPE : %{np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100:.6f}")
print(f"R2   : {r2_score(true_prices, pred_prices):.6f}")
