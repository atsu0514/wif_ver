import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# ====== Autoencoder モデル定義 ======
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_enc = nn.Linear(hidden_size, latent_size)

        self.fc_dec = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        enc_out, _ = self.encoder(x)                # [batch, seq_len, hidden]
        h_last = enc_out[:, -1, :]                  # [batch, hidden]
        z = self.fc_enc(h_last)                     # [batch, latent]

        # デコーダ入力: 潜在ベクトルを各タイムステップ用に繰り返す
        dec_in = self.fc_dec(z)                     # [batch, hidden]
        dec_in = dec_in.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch, seq_len, hidden]
        dec_out, _ = self.decoder(dec_in)           # [batch, seq_len, hidden]
        recon = self.fc_out(dec_out)                # [batch, seq_len, input_size]
        return recon


# ====== Dataset ======
class SensorWindowDataset(Dataset):
    """
    オートエンコーダ用:
    X: [window_size, feature_dim] をそのまま入力＆出力とする
    """
    def __init__(self, df, scaler_x, window_size=100):
        self.window_size = window_size
        feature_cols = ["Presence", "Movement", "MovingRange", "HeartRate", "BreathingRate"]
        features = df[feature_cols].values

        self.features_scaled = scaler_x.transform(features)
        self.length = len(df) - window_size + 1
        if self.length < 0:
            self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.features_scaled[idx: idx + self.window_size]  # [window, feat]
        x_t = torch.tensor(x, dtype=torch.float32)
        return x_t, x_t  # 入力=出力


# ====== メイン ======
CSV_LIST = [
    "sensor_data_20251120_180935.csv",
    "sensor_data_20251120_213630.csv",
    "sensor_data_20251126_133050.csv",
    "sensor_data_20251126_134706.csv",
    "sensor_data_20251126_142058.csv",
    "sensor_data_20251126_143734.csv",
    "sensor_data_20251126_151424.csv",
    "sensor_data_20251126_160715.csv",
    "sensor_data_20251202_230232.csv",
    "sensor_data_20251203_000616.csv",
    "sensor_data_20251203_232723.csv",
]

TEST_CSV = "sensor_data_20251203_232723.csv"


def main():
    window_size = 100
    feature_dim = 5

    # 1. 正常とみなす TRAIN CSV でスケーラーをfit
    train_df_list = []
    for name in CSV_LIST:
        if name == TEST_CSV:
            continue
        path = os.path.join(os.getcwd(), name)
        if not os.path.exists(path):
            print(f"警告: {name} が見つかりません。(スケーラー作成時) スキップします。")
            continue
        df = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp")
        train_df_list.append(df)

    if not train_df_list:
        print("スケーラー作成用のTRAINデータがありません。")
        return

    full_train_df = pd.concat(train_df_list, ignore_index=True)
    scaler_x = MinMaxScaler()
    feature_cols = ["Presence", "Movement", "MovingRange", "HeartRate", "BreathingRate"]
    scaler_x.fit(full_train_df[feature_cols].values)

    # 2. Dataset 作成
    train_datasets = []
    test_datasets = []

    for name in CSV_LIST:
        path = os.path.join(os.getcwd(), name)
        if not os.path.exists(path):
            print(f"警告: {name} が見つかりません。スキップします。")
            continue
        df = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

        ds = SensorWindowDataset(df, scaler_x, window_size=window_size)
        if len(ds) == 0:
            print(f"警告: {name} の有効サンプルが0です。スキップします。")
            continue

        if name == TEST_CSV:
            print(f"{name}: サンプル数 {len(ds)} → TEST 用として使用")
            test_datasets.append(ds)
        else:
            print(f"{name}: サンプル数 {len(ds)} → TRAIN 用として使用")
            train_datasets.append(ds)

    if not train_datasets or not test_datasets:
        print("TRAIN/TEST データセット不足です。")
        return

    train_full = ConcatDataset(train_datasets)
    test_full = ConcatDataset(test_datasets)

    print(f"TRAIN 合計サンプル数: {len(train_full)}")
    print(f"TEST  合計サンプル数: {len(test_full)}")

    train_loader = DataLoader(train_full, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_full, batch_size=16, shuffle=False)

    # 3. Autoencoder モデル定義
    model = LSTMAutoencoder(
        input_size=feature_dim,
        hidden_size=32,
        latent_size=16,
        num_layers=1,
    )

    criterion = nn.MSELoss(reduction="none")  # 各タイムステップ・各特徴の誤差
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. 学習
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for x_batch, _ in train_loader:
            optimizer.zero_grad()
            recon = model(x_batch)
            # recon, x_batch: [batch, seq, feat]
            loss_elem = criterion(recon, x_batch)     # [b, t, f]
            loss = loss_elem.mean()                   # 全体平均
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            count += x_batch.size(0)

        epoch_loss = running_loss / count
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Recon Loss: {epoch_loss:.6f}")

    # 5. 学習データの再構成誤差分布 → threshold
    model.eval()
    train_scores = []
    with torch.no_grad():
        for x_batch, _ in train_loader:
            recon = model(x_batch)
            loss_elem = criterion(recon, x_batch)       # [b, t, f]
            # 各サンプルごとに (t, f) 平均 → [b]
            loss_sample = loss_elem.mean(dim=(1, 2))
            train_scores.extend(loss_sample.numpy().tolist())

    train_scores = np.array(train_scores)
    mean_score = train_scores.mean()
    std_score = train_scores.std()
    threshold = float(mean_score + 3 * std_score)
    print(f"AE Threshold (mean+3std): {threshold:.6f}")

    # 6. テストデータの再構成誤差 → 異常スコア可視化
    test_scores = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            recon = model(x_batch)
            loss_elem = criterion(recon, x_batch)
            loss_sample = loss_elem.mean(dim=(1, 2))
            test_scores.extend(loss_sample.numpy().tolist())

    plt.figure(figsize=(10, 6))
    plt.plot(test_scores, label="AE Reconstruction Error", color="red")
    plt.axhline(y=threshold, color="green", linestyle="--",
                label=f"Threshold (mean+3std={threshold:.6f})")
    plt.title("Unsupervised Anomaly Detection with LSTM Autoencoder")
    plt.xlabel("Test Window Index")
    plt.ylabel("Reconstruction Error (MSE in scaled space)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()