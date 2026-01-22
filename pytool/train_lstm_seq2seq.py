from pathlib import Path
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
HIDDEN_SIZE = 64
NUM_LAYERS = 2
CSV_LIST = [
    "cleaned_sensor_data_20251203_000616.csv", 
    "cleaned_sensor_data_20251203_232723.csv", 
    "cleaned_sensor_data_20251211_143553.csv", 
    "cleaned_sensor_data_20251211_144659.csv", 
    "cleaned_sensor_data_20251211_145537.csv",
    "cleaned_sensor_data_20251212_001425.csv",
    "cleaned_sensor_data_20251212_003309.csv", 
    "cleaned_sensor_data_20251213_004316.csv", 
    "cleaned_sensor_data_20251213_160339.csv",
    "cleaned_sensor_data_20251214_001933.csv",
    "cleaned_sensor_data_20251218_235445.csv",
    "cleaned_sensor_data_20251219_235637.csv",
    "cleaned_sensor_data_20251222_214718.csv",
    "cleaned_sensor_data_20251224_001236.csv",
    "cleaned_sensor_data_20251224_164339.csv",
    "cleaned_sensor_data_20251225_235932.csv",
    "cleaned_sensor_data_20260105_201035.csv",
    "cleaned_sensor_data_20260106_201740.csv",
    "cleaned_sensor_data_20260107_110329.csv",
    "cleaned_sensor_data_20260107_133048.csv",
    "cleaned_sensor_data_20260107_133202.csv",
    "cleaned_sensor_data_20260108_140956.csv"
]
TEST_CSV = [
            "cleaned_sensor_data_20260108_140956.csv",
            "cleaned_sensor_data_20260114_234600.csv"
            ]

BASE_DIR = Path(__file__).resolve().parents[1]  # プロジェクトルート
MODELS_DIR = BASE_DIR / "autoencoder_models"
DATA_DIR = BASE_DIR / "cleaned_data"

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ARTIFACT_DIR = MODELS_DIR / RUN_ID
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "lstm_autoencoder_model.pth"
SCALER_X_PATH = ARTIFACT_DIR / "scaler_x.pkl"
SCALER_Y_PATH = ARTIFACT_DIR / "scaler_y.pkl"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.txt"

LATEST_PATH = MODELS_DIR / "latest.txt"

#CSVファイルのパスを解決する関数
def resolve_csv_path(name: str) -> Path | None:
    candidates = [
        DATA_DIR / name,   # cleaned_data優先
        BASE_DIR / name,   # ルート直下も一応見る
        Path.cwd() / name  # 実行場所互換
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

#LSTMAutoencoderモデルの定義
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h, c) = self.encoder(x)               # (L,B,H)
        b, t, _ = x.shape
        dec_in = torch.zeros((b, t, self.out.out_features), device=x.device, dtype=x.dtype)  # (B,T,2)
        dec_out, _ = self.decoder(dec_in, (h, c)) # (B,T,H)
        recon = self.out(dec_out)                 # (B,T,2)
        return recon

#データからltsmに入力するデータとその次の時刻の教師データを返すクラス
class SensorDataset(Dataset):
    def __init__(self, df, scaler_x, scaler_y, window_size=10):
        self.window_size = window_size
        feature_cols = ["MovingRange", "HeartRate", "BreathingRate"]
        target_cols = ["HeartRate", "BreathingRate"]
        features = df[feature_cols].values
        targets = df[target_cols].values
        
        self.features_scaled = scaler_x.transform(features)
        self.targets_scaled = scaler_y.transform(targets)
        #要素数の取得(データ数からwindow_sizeを減算して何回学習できるかの数を求める)
        self.length = len(df) - window_size
        if self.length < 0: self.length = 0

    def __len__(self):
        return self.length
    
    #xは現在の学習サンプル、yは次の学習サンプルを定義している
    def __getitem__(self, idx):
        x = self.features_scaled[idx: idx + self.window_size]   # (T,3)
        y = self.targets_scaled[idx: idx + self.window_size]    # (T,2)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

#valLossの計算
def evaluate_loss(model, data_loader, criterion):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            out = model(x_batch)
            loss = criterion(out, y_batch)
            bs = x_batch.size(0)
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)

def main():
    # 1. データ読み込みとスケーラー作成
    train_df_list = []
    for name in CSV_LIST:
        if name in TEST_CSV: continue
        path = resolve_csv_path(name)
        if path is not None:
            df = pd.read_csv(str(path), parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
            train_df_list.append(df)
    
    if not train_df_list:
        raise RuntimeError(
            "学習用CSVが見つかりません。"
            " cleaned_data/ 配下にCSVがあるか、CSV_LISTの名前が正しいか確認してください。"
        )
    
    full_train_df = pd.concat(train_df_list, ignore_index=True)
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    feature_cols = ["MovingRange", "HeartRate", "BreathingRate"]
    target_cols = ["HeartRate", "BreathingRate"]
    
    #入力用
    scaler_x.fit(full_train_df[feature_cols].values)
    #出力用
    scaler_y.fit(full_train_df[target_cols].values)
    
    # スケーラー保存
    joblib.dump(scaler_x, str(SCALER_X_PATH))
    joblib.dump(scaler_y, str(SCALER_Y_PATH))
    print("Scalers saved.")

    # 2. Dataset作成
    train_datasets = []
    for name in CSV_LIST:
        if name in TEST_CSV: continue
        path = resolve_csv_path(name)
        if path:
            try:
                df = pd.read_csv(str(path), parse_dates=["Timestamp"])
                
                # 再度チェック: length check
                if df.empty or len(df) <= WINDOW_SIZE:
                    print(f"Skipping {name} (Not enough data: {len(df)} <= {WINDOW_SIZE})")
                    continue
                if not all(c in df.columns for c in feature_cols):
                    continue
                
                df = df.sort_values("Timestamp").reset_index(drop=True)
                ds = SensorDataset(df, scaler_x, scaler_y, window_size=WINDOW_SIZE)
                if len(ds) > 0:
                    train_datasets.append(ds)
            except Exception:
                continue
    
    train_full = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_full, batch_size=16, shuffle=True)
    
    #検証データ（TEST_CSV)
    val_datasets = []
    for name in TEST_CSV:
        val_path = resolve_csv_path(name)
        if val_path is None:
            print(f"Warning: Validation CSV not found: {name}")
            continue
        try:
            val_df = pd.read_csv(str(val_path), parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
            val_ds_single = SensorDataset(val_df, scaler_x, scaler_y, window_size=WINDOW_SIZE)
            # データ長チェック
            if len(val_ds_single) > 0:
                val_datasets.append(val_ds_single)
            else:
                print(f"Skipping val {name} (Not enough data)")
        except Exception as e:
            print(f"Error loading validation file {name}: {e}")
            continue

    if not val_datasets:
        raise RuntimeError(f"検証用CSVが有効に見つかりません: {TEST_CSV}")

    val_ds = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # 3. モデル学習
    model = LSTMAutoencoder(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0011056456020805483)
    
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
        
        train_loss = running_loss / len(train_full)
        val_loss = evaluate_loss(model, val_loader, criterion)

        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, TrainLoss={train_loss:.10f}, ValLoss={val_loss:.10f}")

    # モデル保存
    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")

    # 4. 閾値決定
    model.eval()
    criterion_test = nn.MSELoss(reduction='none')
    train_losses = []
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            out = model(x_batch)
            loss_batch = criterion_test(out, y_batch).mean(dim=(1, 2))  # (B,)
            train_losses.extend(loss_batch.detach().cpu().numpy())
    
    train_losses = np.array(train_losses)
    #3σだと閾値が大きすぎたので99%にする
    threshold = np.percentile(train_losses, 90.0)
    #平均+3σ法でアノマリースコアの閾値を計算
    #threshold = train_losses.mean() + 3 * train_losses.std()

    
    THRESHOLD_PATH.write_text(str(threshold), encoding="utf-8")
    print(f"Threshold saved: {threshold:.4f}")

    # この学習を「最新」として記録
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(RUN_ID, encoding="utf-8")
    print(f"Marked this run as latest: {RUN_ID}")

if __name__ == "__main__":
    main()