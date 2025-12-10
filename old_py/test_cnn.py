import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


CSV_PATH = "sensor_data_20251203_232723.csv"
WINDOW_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_supervised(df: pd.DataFrame, window_size: int = 10):
    """
    X: [N, window_size, 4]  (Presence, Movement, MovingRange, BreathingRate)
    y: [N, 2]  (次の HeartRate, 次の BreathingRate)
    """
    features = df[["Presence", "Movement", "MovingRange", "BreathingRate"]].values
    heart = df["HeartRate"].values
    breath = df["BreathingRate"].values

    X_list = []
    y_list = []

    for i in range(len(df) - window_size - 1):
        window_feat = features[i: i + window_size]   # [window, 4]
        X_list.append(window_feat)
        y_list.append([heart[i + window_size], breath[i + window_size]])

    X = np.array(X_list)  # [N, window, 4]
    y = np.array(y_list)  # [N, 2]
    return X, y


def calc_match_rate(true, pred, tol_ratio=0.10):
    """
    真値の±tol_ratio (例: 0.10=±10%) 以内を「一致」とみなした割合(%)。
    """
    true = np.array(true)
    pred = np.array(pred)
    abs_err = np.abs(true - pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = abs_err / np.where(true == 0, 1, np.abs(true))
    match = (rel_err <= tol_ratio)
    return match.mean() * 100.0


class SensorDataset(Dataset):
    def __init__(self, X, y):
        # X: [N, seq, features] -> CNN用に [N, C=4, seq] に後で変換
        self.X = torch.tensor(X, dtype=torch.float32)  # [N, seq, 4]
        self.y = torch.tensor(y, dtype=torch.float32)  # [N, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]          # [seq, 4]
        x = x.permute(1, 0)      # [4, seq]  (channels=4, length=seq)
        return x, self.y[idx]


class CNN1DModel(nn.Module):
    """
    入力: [batch, 4, seq_len]
    出力: [batch, 2] (Heart, Breath)
    """
    def __init__(self, in_channels=4, num_filters=32, kernel_size=5, output_size=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 時間方向を平均して [batch, num_filters, 1]
        self.fc = nn.Linear(num_filters, output_size)

    def forward(self, x):
        # x: [batch, 4, seq_len]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.global_pool(x)       # [batch, num_filters, 1]
        x = x.squeeze(-1)             # [batch, num_filters]
        out = self.fc(x)              # [batch, 2]
        return out


def main():
    csv_path = os.path.join(os.getcwd(), CSV_PATH)
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # 教師ありデータに変換
    X, y = make_supervised(df, window_size=WINDOW_SIZE)

    # 特徴量スケーリング（MinMax）
    n_samples, seq_len, n_feat = X.shape
    scaler_x = MinMaxScaler()
    X_2d = X.reshape(-1, n_feat)          # [N*seq, 4]
    X_scaled_2d = scaler_x.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(n_samples, seq_len, n_feat)

    # 前80%をtrain、後20%をtest
    n_total = len(X_scaled)
    n_train = int(n_total * 0.8)
    X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    train_ds = SensorDataset(X_train, y_train)
    test_ds = SensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN1DModel(in_channels=4, num_filters=32, kernel_size=5, output_size=2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 学習ループ
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_ds)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, train MSE: {epoch_loss:.4f}")

    # train / test に対する予測
    model.eval()
    with torch.no_grad():
        # train
        train_preds = []
        for xb, _ in train_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            train_preds.append(pred)
        train_preds = np.concatenate(train_preds, axis=0)

        # test
        test_preds = []
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            test_preds.append(pred)
        test_preds = np.concatenate(test_preds, axis=0)

    heart_true_train = y_train[:, 0]
    breath_true_train = y_train[:, 1]
    heart_pred_train = train_preds[:, 0]
    breath_pred_train = train_preds[:, 1]

    heart_true_test = y_test[:, 0]
    breath_true_test = y_test[:, 1]
    heart_pred_test = test_preds[:, 0]
    breath_pred_test = test_preds[:, 1]

    # RMSE
    rmse_train_heart = mean_squared_error(heart_true_train, heart_pred_train) ** 0.5
    rmse_train_breath = mean_squared_error(breath_true_train, breath_pred_train) ** 0.5
    rmse_test_heart = mean_squared_error(heart_true_test, heart_pred_test) ** 0.5
    rmse_test_breath = mean_squared_error(breath_true_test, breath_pred_test) ** 0.5

    print("=== RMSE (HeartRate) [CNN] ===")
    print(f"Train RMSE: {rmse_train_heart:.3f}")
    print(f"Test  RMSE: {rmse_test_heart:.3f}")
    print("=== RMSE (BreathingRate) [CNN] ===")
    print(f"Train RMSE: {rmse_train_breath:.3f}")
    print(f"Test  RMSE: {rmse_test_breath:.3f}")

    # 一致率（±10%）
    heart_match = calc_match_rate(heart_true_test, heart_pred_test, tol_ratio=0.10)
    breath_match = calc_match_rate(breath_true_test, breath_pred_test, tol_ratio=0.10)

    print("=== Match Rate (±10%以内, CNN, Test Only) ===")
    print(f"HeartRate  match: {heart_match:.1f}%")
    print(f"Breathing match: {breath_match:.1f}%")

    # グラフ用インデックス（test部分）
    idx_offset = WINDOW_SIZE + 1 + n_train
    time_idx = np.arange(idx_offset, idx_offset + len(heart_true_test))

    # HeartRate プロット
    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, heart_true_test, label="True HeartRate (test)")
    plt.plot(time_idx, heart_pred_test, label="Pred HeartRate (CNN, test)", alpha=0.7)
    plt.xlabel("Time index (relative)")
    plt.ylabel("HeartRate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # BreathingRate プロット
    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, breath_true_test, label="True BreathingRate (test)")
    plt.plot(time_idx, breath_pred_test, label="Pred BreathingRate (CNN, test)", alpha=0.7)
    plt.xlabel("Time index (relative)")
    plt.ylabel("BreathingRate")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()