import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # output_size=2 (Heart, Breath)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out   # shape: [batch, 2]


# 学習に使うCSVファイル一覧（必要に応じて増やす）
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

# テスト用にするCSV
TEST_CSV = "sensor_data_20251203_232723.csv"


class SensorDataset(Dataset):
    """
    1つのCSV(=1系列)から教師ありデータを作る。
    X : [window_size, 4]  (Presence, Movement, MovingRange, BreathingRate)
    y : [2] (次の HeartRate, 次の BreathingRate)
    """
    def __init__(self, df, scaler_x, scaler_y, window_size=10):
        self.window_size = window_size

        feature_cols = ["Presence", "Movement", "MovingRange", "HeartRate", "BreathingRate"]
        target_cols = ["HeartRate", "BreathingRate"]

        features = df[feature_cols].values
        targets = df[target_cols].values  # shape: [N, 2]

        # ここでは transform だけ行う (fitは main で一度だけ)
        self.features_scaled = scaler_x.transform(features)
        self.targets_scaled = scaler_y.transform(targets)

        self.length = len(df) - window_size - 1
        if self.length < 0:
            self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.features_scaled[idx: idx + self.window_size]          # [window, 4]
        y = self.targets_scaled[idx + self.window_size]                # [2]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


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


# ===== メイン処理 =====
def main():
    window_size = 15

    # 1. まず全ての学習用データを1つのDataFrameにまとめて、スケーラーを作る
    train_df_list = []
    for name in CSV_LIST:
        if name == TEST_CSV:
            continue  # テストは混ぜない
        path = os.path.join(os.getcwd(), name)
        if not os.path.exists(path):
            print(f"警告: {name} が見つかりません。(スケーラー作成時) スキップします。")
            continue
        df = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp")
        train_df_list.append(df)

    if not train_df_list:
        print("スケーラー作成用のTRAINデータがありません。CSV_LIST / TEST_CSV を確認してください。")
        return

    full_train_df = pd.concat(train_df_list, ignore_index=True)

    # 全学習データでfitする (これが「正常の基準」になる)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    feature_cols = ["Presence", "Movement", "MovingRange", "HeartRate", "BreathingRate"]
    target_cols = ["HeartRate", "BreathingRate"]

    scaler_x.fit(full_train_df[feature_cols].values)
    scaler_y.fit(full_train_df[target_cols].values)

    # 2. 各CSVから SensorDataset を作成し、train/test に振り分け
    train_datasets = []
    test_datasets = []

    for name in CSV_LIST:
        path = os.path.join(os.getcwd(), name)
        if not os.path.exists(path):
            print(f"警告: {name} が見つかりません。スキップします。")
            continue
        df = pd.read_csv(path, parse_dates=["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

        ds = SensorDataset(df, scaler_x, scaler_y, window_size=window_size)
        if len(ds) == 0:
            print(f"警告: {name} の有効サンプルが0です。スキップします。")
            continue

        if name == TEST_CSV:
            print(f"{name}: サンプル数 {len(ds)} → TEST 用として使用")
            test_datasets.append(ds)
        else:
            print(f"{name}: サンプル数 {len(ds)} → TRAIN 用として使用")
            train_datasets.append(ds)

    if not train_datasets:
        print("TRAIN 用データセットがありません。CSV_LIST / TEST_CSV を確認してください。")
        return
    if not test_datasets:
        print("TEST 用データセットがありません。TEST_CSV を確認してください。")
        return

    # ConcatDataset で結合
    train_full = ConcatDataset(train_datasets)
    test_full = ConcatDataset(test_datasets)

    print(f"TRAIN 合計サンプル数: {len(train_full)}")
    print(f"TEST  合計サンプル数: {len(test_full)}")

    train_loader = DataLoader(train_full, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_full, batch_size=16, shuffle=False)

    # --- モデル定義 ---
    input_size = 5          # 特徴量数
    hidden_size = 32
    num_layers = 1
    output_size = 2         # HeartRate と BreathingRate の2出力

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- 学習ループ（TRAIN のみで学習）---
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * x_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_full)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}")

    # === 学習データの誤差分布を確認して閾値を決める ===
    model.eval()
    criterion_test = nn.MSELoss(reduction='none')  # 個別サンプルごとの誤差
    train_losses = []
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            out = model(x_batch)                       # [batch, 2]
            loss_batch = criterion_test(out, y_batch)  # [batch, 2]
            loss_batch = loss_batch.mean(dim=1)        # -> [batch]
            train_losses.extend(loss_batch.numpy())

    train_losses = np.array(train_losses)
    train_mean = train_losses.mean()
    train_std = train_losses.std()

    # 「平均 + 3シグマ」を閾値にする
    threshold = float(train_mean + 3 * train_std)
    print(f"Calculated Threshold (Mean + 3std): {threshold:.4f}")

    # ===== TEST データ上での予測 vs 実測（スケール空間での一致率）=====
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred_scaled = model(x_batch)
            all_preds.append(pred_scaled.numpy())
            all_trues.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds, axis=0)   # [N_test, 2]
    all_trues = np.concatenate(all_trues, axis=0)   # [N_test, 2]

    heart_true = all_trues[:, 0]
    breath_true = all_trues[:, 1]
    heart_pred = all_preds[:, 0]
    breath_pred = all_preds[:, 1]

    # 一致率（±10%以内）※スケール空間での相対誤差
    heart_match = calc_match_rate(heart_true, heart_pred, tol_ratio=0.10)
    breath_match = calc_match_rate(breath_true, breath_pred, tol_ratio=0.10)

    print("=== Match Rate (±10%以内, LSTM, Multi-file, 指定TEST, GlobalScaler) ===")
    print(f"HeartRate  match: {heart_match:.1f}%")
    print(f"Breathing match: {breath_match:.1f}%")

    # グラフ用インデックス（テストサンプルの相対インデックス）
    time_idx = np.arange(len(heart_true))

    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, heart_true, label="True HeartRate (scaled, test)")
    plt.plot(time_idx, heart_pred, label="Pred HeartRate (LSTM, test)", alpha=0.7)
    plt.xlabel("Sample index (test file only)")
    plt.ylabel("Scaled HeartRate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, breath_true, label="True BreathingRate (scaled, test)")
    plt.plot(time_idx, breath_pred, label="Pred BreathingRate (LSTM, test)", alpha=0.7)
    plt.xlabel("Sample index (test file only)")
    plt.ylabel("Scaled BreathingRate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===== 異常検知としての評価（テストデータで予測誤差を「異常スコア」として可視化）=====
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)                           # [batch, 2]
            loss_batch = criterion_test(outputs, y_batch)      # [batch, 2]
            loss_batch = loss_batch.mean(dim=1)                # -> [batch] 各サンプルの平均MSE
            anomaly_scores.extend(loss_batch.numpy().tolist())

    plt.figure(figsize=(10, 6))
    plt.plot(anomaly_scores, label="Anomaly Score (Prediction Error)", color='red')
    plt.axhline(y=threshold, color='green', linestyle='--',
                label=f"Threshold (mean+3std={threshold:.4f})")

    plt.title("Anomaly Detection using LSTM (Prediction Error on TEST_CSV)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Anomaly Score (MSE in scaled space)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()