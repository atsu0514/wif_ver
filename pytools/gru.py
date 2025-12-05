import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


CSV_PATH = "sensor_data_20251203_000616.csv"  # パスは必要に応じて調整


# ===== データセット定義 =====
class SensorDataset(Dataset):
    """
    X : [window_size, 4]  (Presence, Movement, MovingRange, BreathingRate)
    y : [2] (次の HeartRate, 次の BreathingRate)
    """
    def __init__(self, df, window_size=10):
        self.window_size = window_size

        # 入力特徴
        features = df[["Presence", "Movement", "MovingRange", "BreathingRate"]].values
        # 目的変数: Heart と Breath を2次元でまとめる
        targets = df[["HeartRate", "BreathingRate"]].values  # shape: [N, 2]

        # スケーリング
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.features_scaled = self.scaler_x.fit_transform(features)
        self.targets_scaled = self.scaler_y.fit_transform(targets)

        self.length = len(df) - window_size - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.features_scaled[idx: idx + self.window_size]          # [window, 4]
        # 1ステップ先の Heart & Breath → 2次元
        y = self.targets_scaled[idx + self.window_size]                # [2]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ===== メイン処理 =====
def main():
    # --- CSV 読み込み ---
    df = pd.read_csv(CSV_PATH, parse_dates=["Timestamp"])
    df = df.reset_index(drop=True)

    window_size = 10
    dataset = SensorDataset(df, window_size=window_size)

    # train / val 分割
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # --- モデル定義 ---
    input_size = 4          # 特徴量数
    hidden_size = 32
    num_layers = 1
    output_size = 2         # HeartRate と BreathingRate の2出力

    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- 学習ループ ---
    num_epochs = 100
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)          # [batch, 2]
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * x_batch.size(0)

        epoch_train_loss = running_train_loss / n_train
        train_losses.append(epoch_train_loss)

        # --- 検証 ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * x_batch.size(0)
        epoch_val_loss = running_val_loss / n_val if n_val > 0 else 0.0
        val_losses.append(epoch_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {epoch_train_loss:.4f}  Val Loss: {epoch_val_loss:.4f}")

    # ===== 学習曲線 =====
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Heart+Breath)")
    plt.title("Training / Validation Loss (2-output LSTM)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===== 時系列上での予測 vs 実測 =====
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            x_input = x.unsqueeze(0)  # [1, window, 4]
            pred_scaled = model(x_input)          # [1,2]
            all_preds.append(pred_scaled.numpy().reshape(-1))  # [2]
            all_trues.append(y.numpy().reshape(-1))            # [2]

    all_preds = np.array(all_preds)   # shape: [N, 2]
    all_trues = np.array(all_trues)   # shape: [N, 2]

    # 逆スケーリング（Heart, Breath をまとめて元スケールへ）
    preds_inv = dataset.scaler_y.inverse_transform(all_preds)   # [N,2]
    trues_inv = dataset.scaler_y.inverse_transform(all_trues)   # [N,2]

    heart_true = trues_inv[:, 0]
    breath_true = trues_inv[:, 1]
    heart_pred = preds_inv[:, 0]
    breath_pred = preds_inv[:, 1]

    # HeartRate プロット
    plt.figure(figsize=(10, 4))
    plt.plot(heart_true, label="True HeartRate")
    plt.plot(heart_pred, label="Pred HeartRate", alpha=0.7)
    plt.xlabel("Time index (relative)")
    plt.ylabel("HeartRate")
    plt.title("HeartRate: True vs Predicted (1-step ahead, LSTM 2-output)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # BreathingRate プロット
    plt.figure(figsize=(10, 4))
    plt.plot(breath_true, label="True BreathingRate")
    plt.plot(breath_pred, label="Pred BreathingRate", alpha=0.7)
    plt.xlabel("Time index (relative)")
    plt.ylabel("BreathingRate")
    plt.title("BreathingRate: True vs Predicted (1-step ahead, LSTM 2-output)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()