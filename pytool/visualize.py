from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import joblib
import matplotlib.pyplot as plt

# --- 設定 (train_lstm_seq2seq.py と合わせる) ---
WINDOW_SIZE = 15
HIDDEN_SIZE = 16
NUM_LAYERS = 2

# 学習に使ったファイル (Train)
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
# 検証に使ったファイル (Validation)
TEST_CSV = [
            "cleaned_sensor_data_val_1_20260108_140956.csv",
            "cleaned_sensor_data_val_1_20260114_234600.csv",
            "cleaned_sensor_data_20260115_145841_kakokyuu.csv",
            "cleaned_sensor_data_20260125_114313_nagaeri.csv",
            "cleaned_sensor_data_20260119_222651_mukokyuuy.csv"
            ]

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "autoencoder_models"
DATA_DIR = BASE_DIR / "cleaned_data"
LATEST_PATH = MODELS_DIR / "latest.txt"

def _resolve_artifact_dir() -> Path:
    if not LATEST_PATH.exists():
        raise FileNotFoundError(f"latest.txt が見つかりません: {LATEST_PATH}")
    run_id = LATEST_PATH.read_text(encoding="utf-8").strip()
    artifact_dir = MODELS_DIR / run_id
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact_dir が見つかりません: {artifact_dir}")
    return artifact_dir

ARTIFACT_DIR = _resolve_artifact_dir()
MODEL_PATH = ARTIFACT_DIR / "lstm_autoencoder_model.pth"
SCALER_X_PATH = ARTIFACT_DIR / "scaler_x.pkl"
SCALER_Y_PATH = ARTIFACT_DIR / "scaler_y.pkl"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.txt"

# --- モデル定義 ---
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
    
    # 1. Scaler & Model Load
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    
    model = LSTMEncoderDecoder(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 2. Threshold Load
    try:
        lines = THRESHOLD_PATH.read_text(encoding="utf-8").strip().splitlines()
        thresholds = [float(x) for x in lines]
        th_low, th_mid, th_high = thresholds
        print(f"Loaded Thresholds: Low={th_low}, Mid={th_mid}, High={th_high}")
    except Exception as e:
        print(f"Error loading thresholds: {e}")
        return

    # 3. Data Preparation
    def get_dataloader(csv_names):
        datasets = []
        for name in csv_names:
            path = resolve_csv_path(name)
            if not path: continue
            try:
                df = pd.read_csv(str(path), parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
                if len(df) > WINDOW_SIZE:
                    datasets.append(SensorDataset(df, scaler_x, scaler_y, window_size=WINDOW_SIZE))
            except: continue
        if not datasets: return None
        return DataLoader(ConcatDataset(datasets), batch_size=32, shuffle=False)

    train_loader = get_dataloader([c for c in CSV_LIST if c not in TEST_CSV])
    val_loader = get_dataloader(TEST_CSV)

    if not train_loader:
        print("No training data found.")
        return

    # 4. Calculate Losses
    criterion = nn.MSELoss(reduction='none')

    def get_losses(loader):
        losses = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                outputs = model(x_batch)
                # (batch,)
                # ここでは「窓全体の平均MSE」 or 「最後のステップのMSE」など
                # train_lstm_seq2seq.py のロジックに合わせる必要がある
                # 今回は train.py に合わせて mean(dim=(1,2)) を使用
                batch_loss = criterion(outputs, y_batch).mean(dim=(1,2))
                losses.extend(batch_loss.numpy())
        return np.array(losses)

    print("Calculating Train losses...")
    train_losses = get_losses(train_loader)
    
    print("Calculating Validation losses...")
    val_losses = get_losses(val_loader) if val_loader else []

    # 5. Plot（改良版: 詳細ビュー + 全体ビュー）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})
    bins = 50

    # combine losses safely
    if len(val_losses) > 0:
        all_losses = np.concatenate([train_losses, val_losses])
    else:
        all_losses = train_losses

    # 左: ズーム（メインの塊が見えるように）
    ax1.hist(train_losses, bins=bins, alpha=0.6, label='Train (Normal)', color='blue', density=True)
    if len(val_losses) > 0:
        ax1.hist(val_losses, bins=bins, alpha=0.6, label='Validation (Anomaly Mixed)', color='red', density=True)
    ax1.axvline(th_low, color='orange', linestyle=':', linewidth=2)
    ax1.axvline(th_mid, color='green', linestyle='--', linewidth=2)
    ax1.axvline(th_high, color='purple', linestyle='-', linewidth=2)
    ax1.set_title('Zoomed view (main mass)')
    ax1.set_xlabel("Reconstruction Error (MSE)")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # ズーム範囲を決定 (90パーセンタイルを基準に少し余裕を持たせる)
    p90 = np.percentile(all_losses, 90) if len(all_losses) > 0 else 0.0
    zoom_xlim = max(p90 * 1.2, th_high * 1.2, np.max(all_losses) * 0.05, 1e-8)
    ax1.set_xlim(0, zoom_xlim)

    # 右: 全体ビュー（対数寄りで尾を見やすくする）
    ax2.hist(train_losses, bins=bins, alpha=0.6, label='Train (Normal)', color='blue', density=True)
    if len(val_losses) > 0:
        ax2.hist(val_losses, bins=bins, alpha=0.6, label='Validation (Anomaly Mixed)', color='red', density=True)
    ax2.axvline(th_low, color='orange', linestyle=':', linewidth=2)
    ax2.axvline(th_mid, color='green', linestyle='--', linewidth=2)
    ax2.axvline(th_high, color='purple', linestyle='-', linewidth=2)
    # symlog を使うことで 0 付近は線形、尾部はログ感で見やすくなる
    ax2.set_xscale('symlog', linthresh=1e-6)
    ax2.set_title('Full range view (symlog)')
    ax2.set_xlabel("Reconstruction Error (MSE)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    max_loss = np.max(all_losses) if len(all_losses) > 0 else 1.0
    ax2.set_xlim(0, max_loss * 1.05)

    plot_path = ARTIFACT_DIR / "threshold_distribution_check.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved plot to: {plot_path}")

    # サーバー環境でなければ表示
    plt.show()

if __name__ == "__main__":
    main()