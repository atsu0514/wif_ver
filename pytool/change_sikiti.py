from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import joblib
import matplotlib.pyplot as plt
import datetime  # 追加（ファイル下部で使用）

# ==========================================
# ▼▼▼ ここで新しい閾値の％を設定してください ▼▼▼
# ==========================================
# 例: [93.0, 96.0, 99.0] -> 正常データの93%, 96%, 99%が含まれるライン
NEW_PERCENTILES = [83.0, 84.0, 85.0]  # [Low, Mid, High]
# ==========================================

# --- 基本設定 (学習時と合わせる) ---
WINDOW_SIZE = 15
HIDDEN_SIZE = 16
NUM_LAYERS = 2

# 学習に使ったファイル (Train: 正常データの分布を知るために必要)
CSV_LIST = [
    "cleaned_sensor_data_20251203_000616.csv", 
    "cleaned_sensor_data_20251203_232723.csv", 
    "cleaned_sensor_data_20251212_001425.csv",
    "cleaned_sensor_data_20251212_003309.csv", 
    "cleaned_sensor_data_20251213_004316.csv", 
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
]
# 検証に使ったファイル (Validation: 確認用)
TEST_CSV = [
    "cleaned_sensor_data_val_1_20260108_140956.csv",
    "cleaned_sensor_data_val_1_20260114_234600.csv",
    "cleaned_sensor_data_20260115_145841_kakokyuu.csv",
    "cleaned_sensor_data_20260125_114313_nagaeri.csv",
    "cleaned_sensor_data_20260119_222651_mukokyuuy.csv"
]

BASE_DIR = Path(__file__).resolve().parents[1]  # プロジェクトルートを指すように修正
MODELS_DIR = BASE_DIR / "autoencoder_models"
DATA_DIR = BASE_DIR / "cleaned_data"
LATEST_PATH = MODELS_DIR / "latest.txt"

def _resolve_artifact_dir() -> Path:
    # MODELS_DIR を確実に作成しておく
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # latest.txt が無ければ新しい run を作成して latest.txt を書く（自動化）
    if not LATEST_PATH.exists():
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_manual")
        artifact_dir = MODELS_DIR / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        LATEST_PATH.write_text(run_id, encoding="utf-8")
        print(f"latest.txt が無かったため新しいランを作成しました: {LATEST_PATH} -> {artifact_dir}")
        return artifact_dir

    run_id = LATEST_PATH.read_text(encoding="utf-8").strip()
    artifact_dir = MODELS_DIR / run_id
    if not artifact_dir.is_dir():
        artifact_dir.mkdir(parents=True, exist_ok=True)
        print(f"artifact_dir が無かったため作成しました: {artifact_dir}")
    return artifact_dir

ARTIFACT_DIR = _resolve_artifact_dir()
MODEL_PATH = ARTIFACT_DIR / "lstm_autoencoder_model.pth"
SCALER_X_PATH = ARTIFACT_DIR / "scaler_x.pkl"
SCALER_Y_PATH = ARTIFACT_DIR / "scaler_y.pkl"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.txt"

# --- モデル定義 (学習時と同じ構造が必要) ---
class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        b, t, _ = x.shape
        dec_in = torch.zeros((b, t, self.out.out_features), device=x.device, dtype=x.dtype)
        dec_out, _ = self.decoder(dec_in, (h, c))
        recon = self.out(dec_out)
        return recon

# --- Dataset 定義 ---
class SensorDataset(Dataset):
    def __init__(self, df, scaler_x, scaler_y, window_size=10):
        self.window_size = window_size
        feature_cols = ["MovingRange", "HeartRate", "BreathingRate"]
        target_cols = ["HeartRate", "BreathingRate"]
        features = df[feature_cols].values
        targets = df[target_cols].values
        
        self.features_scaled = scaler_x.transform(features)
        self.targets_scaled = scaler_y.transform(targets)
        self.length = len(df) - window_size
        if self.length < 0: self.length = 0

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x = self.features_scaled[idx: idx + self.window_size]
        y = self.targets_scaled[idx: idx + self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def resolve_csv_path(name: str) -> Path | None:
    candidates = [DATA_DIR / name, BASE_DIR / name, Path.cwd() / name]
    for p in candidates:
        if p.exists(): return p
    return None

def main():
    print(f"Loading artifacts from: {ARTIFACT_DIR}")
    
    # 1. Load Model & Scalers
    if not MODEL_PATH.exists():
        print("Model file not found.")
        return

    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    
    model = LSTMEncoderDecoder(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 2. Data Preparation
    def get_dataloader(csv_names):
        datasets = []
        for name in csv_names:
            path = resolve_csv_path(name)
            if not path: continue
            try:
                df = pd.read_csv(str(path), parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
                if len(df) > WINDOW_SIZE:
                    datasets.append(SensorDataset(df, scaler_x, scaler_y, window_size=WINDOW_SIZE))
            except Exception: continue
        if not datasets: return None
        return DataLoader(ConcatDataset(datasets), batch_size=64, shuffle=False)

    print("Preparing Data...")
    train_loader = get_dataloader(CSV_LIST) # Threshold基準用
    val_loader = get_dataloader(TEST_CSV)   # 確認用

    if not train_loader:
        print("Train data load failed.")
        return

    # 3. Calculate Losses (Inference Only)
    criterion = nn.MSELoss(reduction='none')

    def get_losses(loader):
        losses = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                outputs = model(x_batch)
                # バッチごとの平均MSE (train時と合わせる)
                batch_loss = criterion(outputs, y_batch).mean(dim=(1,2))
                losses.extend(batch_loss.numpy())
        return np.array(losses)

    print("Calculating losses on Train data (to determine thresholds)...")
    train_losses = get_losses(train_loader)
    
    print("Calculating losses on Validation data (for visualization)...")
    val_losses = get_losses(val_loader) if val_loader else []

    # 4. Determine New Thresholds
    # 設定したパーセンタイルに基づいて閾値を計算
    new_thresholds = np.percentile(train_losses, NEW_PERCENTILES)
    th_low, th_mid, th_high = new_thresholds

    print("\n" + "="*40)
    print(f"NEW Thresholds (Percentiles: {NEW_PERCENTILES})")
    print(f"  Low : {th_low:.6f}")
    print(f"  Mid : {th_mid:.6f}")
    print(f"  High: {th_high:.6f}")
    print("="*40 + "\n")

    # 5. Save New Thresholds
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        for val in new_thresholds:
            f.write(f"{val}\n")
    print(f"Saved new thresholds to: {THRESHOLD_PATH}")

    # 6. Plot Result
    plt.figure(figsize=(12, 6))
    
    # ヒストグラム
    plt.hist(train_losses, bins=100, alpha=0.6, label='Train (Normal)', color='blue', density=True)
    if len(val_losses) > 0:
        plt.hist(val_losses, bins=100, alpha=0.6, label='Validation (Anomaly Mixed)', color='red', density=True)

    # 閾値ライン
    plt.axvline(th_low, color='orange', linestyle='--', linewidth=2, label=f'New Low ({NEW_PERCENTILES[0]}%)')
    plt.axvline(th_mid, color='green', linestyle='--', linewidth=2, label=f'New Mid ({NEW_PERCENTILES[1]}%)')
    plt.axvline(th_high, color='purple', linestyle='-', linewidth=2, label=f'New High ({NEW_PERCENTILES[2]}%)')

    plt.title(f"Re-calculated Thresholds (Hidden={HIDDEN_SIZE})")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 表示範囲の調整（外れ値に引っ張られすぎないように）
    all_losses = np.concatenate([train_losses, val_losses]) if len(val_losses) > 0 else train_losses
    p99_5 = np.percentile(all_losses, 99.5)
    plt.xlim(0, max(p99_5 * 1.5, th_high * 1.2))

    plot_path = ARTIFACT_DIR / "new_threshold_check.png"
    plt.savefig(plot_path)
    print(f"Saved visualization to: {plot_path}")
    
    # GUI環境なら表示
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()