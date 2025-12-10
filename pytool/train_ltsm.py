import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import joblib  # スケーラー保存用
import datetime  # 追加

# --- 設定 ---
WINDOW_SIZE = 15
HIDDEN_SIZE = 32
NUM_LAYERS = 1
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
# テスト用にするCSV（学習には含めないが、閾値決定の確認用などに使う）
TEST_CSV = "sensor_data_20251203_232723.csv"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # プロジェクトルート
MODELS_DIR = os.path.join(BASE_DIR, "models")

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ARTIFACT_DIR = os.path.join(MODELS_DIR, RUN_ID)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACT_DIR, "lstm_model.pth")
SCALER_X_PATH = os.path.join(ARTIFACT_DIR, "scaler_x.pkl")
SCALER_Y_PATH = os.path.join(ARTIFACT_DIR, "scaler_y.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.txt")

LATEST_PATH = os.path.join(MODELS_DIR, "latest.txt")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class SensorDataset(Dataset):
    def __init__(self, df, scaler_x, scaler_y, window_size=10):
        self.window_size = window_size
        feature_cols = ["Presence", "Movement", "MovingRange", "HeartRate", "BreathingRate"]
        target_cols = ["HeartRate", "BreathingRate"]
        features = df[feature_cols].values
        targets = df[target_cols].values
        
        self.features_scaled = scaler_x.transform(features)
        self.targets_scaled = scaler_y.transform(targets)
        self.length = len(df) - window_size - 1
        if self.length < 0: self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.features_scaled[idx: idx + self.window_size]
        y = self.targets_scaled[idx + self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main():
    # 1. データ読み込みとスケーラー作成
    train_df_list = []
    for name in CSV_LIST:
        if name == TEST_CSV: continue
        path = os.path.join(os.getcwd(), name)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp")
            train_df_list.append(df)
    
    full_train_df = pd.concat(train_df_list, ignore_index=True)
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    feature_cols = ["Presence", "Movement", "MovingRange", "HeartRate", "BreathingRate"]
    target_cols = ["HeartRate", "BreathingRate"]
    
    scaler_x.fit(full_train_df[feature_cols].values)
    scaler_y.fit(full_train_df[target_cols].values)
    
    # スケーラー保存
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    print("Scalers saved.")

    # 2. Dataset作成
    train_datasets = []
    for name in CSV_LIST:
        if name == TEST_CSV: continue
        path = os.path.join(os.getcwd(), name)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
            ds = SensorDataset(df, scaler_x, scaler_y, window_size=WINDOW_SIZE)
            if len(ds) > 0:
                train_datasets.append(ds)
    
    train_full = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_full, batch_size=16, shuffle=True)
    
    # 3. モデル学習
    model = LSTMModel(input_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 50 # 時間短縮のため50にしていますが、必要に応じて増やしてください
    print("Start Training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_full):.4f}")

    # モデル保存
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 4. 閾値決定
    model.eval()
    criterion_test = nn.MSELoss(reduction='none')
    train_losses = []
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            out = model(x_batch)
            loss_batch = criterion_test(out, y_batch).mean(dim=1)
            train_losses.extend(loss_batch.numpy())
    
    train_losses = np.array(train_losses)
    threshold = train_losses.mean() + 3 * train_losses.std()
    
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))
    print(f"Threshold saved: {threshold:.4f}")

    # この学習を「最新」として記録
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LATEST_PATH, "w") as f:
        f.write(RUN_ID)
    print(f"Marked this run as latest: {RUN_ID}")

if __name__ == "__main__":
    main()