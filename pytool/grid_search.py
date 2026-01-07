from pathlib import Path
from collections import deque
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid

PARAM_GRID = {
    "WINDOW_SIZE": [15, 30, 45],
    "HIDDEN_SIZE": [32, 64, 128],
    "NUM_LAYERS": [1, 2],
    "BATCH_SIZE": [16, 32],
    "LEARNING_RATE": [1e-3, 3e-4],
}


WINDOW_SIZE = 30
HIDDEN_SIZE = 32
NUM_LAYERS = 1

TEST_CSV_LIST = [
    "cleaned_sensor_data_20260107_133513_nonbreath.csv",
    "cleaned_sensor_data_20260107_141840_negaeri.csv",
]

FEATURE_COLS = ["MovingRange", "HeartRate", "BreathingRate"]
TARGET_COLS = ["HeartRate", "BreathingRate"]

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "autoencoder_models"   # train側に合わせる
DATA_DIR = BASE_DIR / "cleaned_data"
LATEST_PATH = MODELS_DIR / "latest.txt"


def resolve_csv_path(name: str) -> Path | None:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p

    candidates = [DATA_DIR / name, BASE_DIR / name, Path.cwd() / name]
    for c in candidates:
        if c.exists():
            return c
    return None


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


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        b, t, _ = x.shape
        dec_in = torch.zeros((b, t, self.out.out_features), device=x.device, dtype=x.dtype)  # (B,T,2)
        dec_out, _ = self.decoder(dec_in, (h, c))
        recon = self.out(dec_out)  # (B,T,2)
        return recon


class RealtimeLikeOfflineEvaluatorAE:
    """
    AE版:
      - WINDOW_SIZE たまったら、その窓の HR/BR 系列を復元
      - anomaly_score = 窓全体の scaled MSE 平均（train時の閾値算出と同じ）
      - score > threshold なら異常
    """
    def __init__(self):
        self.scaler_x = joblib.load(str(SCALER_X_PATH))
        self.scaler_y = joblib.load(str(SCALER_Y_PATH))

        self.model = LSTMAutoencoder(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2)
        self.model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
        self.model.eval()

        self.threshold = float(THRESHOLD_PATH.read_text(encoding="utf-8"))
        self.buffer = deque(maxlen=WINDOW_SIZE)

    def step(self, feature_row_original: np.ndarray):
        self.buffer.append(feature_row_original.astype(float))
        if len(self.buffer) < WINDOW_SIZE:
            return None

        window = np.array(self.buffer)  # (W,3)
        x_scaled = self.scaler_x.transform(window)  # (W,3)

        y_window_original = window[:, 1:3]  # (W,2) = [HeartRate, BreathingRate]
        y_scaled = self.scaler_y.transform(y_window_original)  # (W,2)

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)  # (1,W,3)

        with torch.no_grad():
            recon_scaled = self.model(x_tensor)[0].cpu().numpy()  # (W,2)

        # (修正前) 窓全体の平均MSE
        # score = float(np.mean((recon_scaled - y_scaled) ** 2))

        # (修正案) 最後のステップのMSEだけを見る、または重み付けする
        last_step_error = (recon_scaled[-1] - y_scaled[-1]) ** 2  # (2,)
        score = float(np.mean(last_step_error))  # 心拍・呼吸の平均
        is_anomaly = bool(score > self.threshold)

        # 参考: 最終時刻の復元値（プロット用）
        recon_last_original = self.scaler_y.inverse_transform([recon_scaled[-1]])[0]  # (2,)
        true_last_original = y_window_original[-1]  # (2,)

        return {
            "anomaly_score": score,
            "is_anomaly": is_anomaly,
            "threshold": float(self.threshold),
            "recon_last_heart": float(recon_last_original[0]),
            "recon_last_breath": float(recon_last_original[1]),
            "true_last_heart": float(true_last_original[0]),
            "true_last_breath": float(true_last_original[1]),
        }


class SensorDataset(Dataset):
    def __init__(self, df, scaler_x, scaler_y, window_size):
        self.window_size = window_size
        feature_cols = ["MovingRange", "HeartRate", "BreathingRate"]
        target_cols = ["HeartRate", "BreathingRate"]
        features = df[feature_cols].values
        targets = df[target_cols].values
        self.features_scaled = scaler_x.transform(features)
        self.targets_scaled = scaler_y.transform(targets)
        self.length = max(0, len(df) - window_size)

    def __len__(self): return self.length

    def __getitem__(self, idx):
        x = self.features_scaled[idx: idx + self.window_size]
        y = self.targets_scaled[idx: idx + self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def eval_csv(csv_name: str):
    csv_path = resolve_csv_path(csv_name)
    if csv_path is None:
        print(f"警告: 見つかりません: {csv_name}")
        return

    df = pd.read_csv(str(csv_path), parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    for c in ["Timestamp", *FEATURE_COLS]:
        if c not in df.columns:
            raise ValueError(f"{csv_name} に必要列 {c} がありません")

    evaluator = RealtimeLikeOfflineEvaluatorAE()
    print(f"\n== {csv_name} ==")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print(f"Threshold (scaled recon MSE): {evaluator.threshold:.6f}")

    ts_list = []
    score_list = []
    flag_list = []

    true_hr = []
    recon_hr = []
    true_br = []
    recon_br = []

    for _, r in df.iterrows():
        features = r[FEATURE_COLS].to_numpy(dtype=float)
        res = evaluator.step(features)
        if res is None:
            continue

        ts_list.append(r["Timestamp"])
        score_list.append(float(res["anomaly_score"]))
        flag_list.append(bool(res["is_anomaly"]))

        true_hr.append(res["true_last_heart"])
        recon_hr.append(res["recon_last_heart"])
        true_br.append(res["true_last_breath"])
        recon_br.append(res["recon_last_breath"])

    if len(score_list) == 0:
        print("評価サンプルが0です（データが短い/欠損など）。")
        return

    scores = np.array(score_list, dtype=float)
    flags = np.array(flag_list, dtype=bool)

    true_hr_arr = np.array(true_hr, dtype=float)
    recon_hr_arr = np.array(recon_hr, dtype=float)
    true_br_arr = np.array(true_br, dtype=float)
    recon_br_arr = np.array(recon_br, dtype=float)

    # 复元精度（オリジナルスケール、最後の点）
    heart_mae = float(np.mean(np.abs(true_hr_arr - recon_hr_arr)))
    heart_rmse = float(np.sqrt(np.mean((true_hr_arr - recon_hr_arr) ** 2)))
    breath_mae = float(np.mean(np.abs(true_br_arr - recon_br_arr)))
    breath_rmse = float(np.sqrt(np.mean((true_br_arr - recon_br_arr) ** 2)))

    heart_match = match_rate_percent(true_hr_arr, recon_hr_arr, tol_abs=1.0)
    breath_match = match_rate_percent(true_br_arr, recon_br_arr, tol_abs=1.0)

    print(f"Samples evaluated: {len(scores)}")
    print(f"Heart  MAE={heart_mae:.3f}, RMSE={heart_rmse:.3f}, Match(±1.0rpm)={heart_match:.1f}%")
    print(f"Breath MAE={breath_mae:.3f}, RMSE={breath_rmse:.3f}, Match(±1.0rpm)={breath_match:.1f}%")
    print(f"Anomaly rate: {float(flags.mean() * 100):.2f}%")

    # 可視化（不要なら消してOK）
    x = np.arange(len(scores))

    plt.figure(figsize=(10, 4))
    plt.plot(x, true_hr, label="True HeartRate")
    plt.plot(x, recon_hr, label="Recon HeartRate", alpha=0.7)
    plt.title(f"HeartRate (recon last step): {csv_name}")
    plt.xlabel("Sample index")
    plt.ylabel("HeartRate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x, true_br, label="True BreathingRate")
    plt.plot(x, recon_br, label="Recon BreathingRate", alpha=0.7)
    plt.title(f"BreathingRate (recon last step): {csv_name}")
    plt.xlabel("Sample index")
    plt.ylabel("BreathingRate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Anomaly Score (scaled recon MSE)", color="red")
    plt.axhline(y=evaluator.threshold, color="green", linestyle="--", label=f"Threshold={evaluator.threshold:.6f}")
    plt.title(f"Anomaly Score: {csv_name}")
    plt.xlabel("Sample index")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def match_rate_percent(true_vals: np.ndarray, pred_vals: np.ndarray, tol_abs: float = 1.5) -> float:
    """
    真値の±tol_abs以内を「一致」とみなした割合(%)。
    """
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    abs_err = np.abs(true_vals - pred_vals)
    return float((abs_err <= tol_abs).mean() * 100.0)


# ==========================================
# 学習実行関数
# ==========================================
def train_evaluate(params, train_dfs, val_dfs, scaler_x, scaler_y):
    # パラメータ展開
    ws = params["WINDOW_SIZE"]
    hs = params["HIDDEN_SIZE"]
    nl = params["NUM_LAYERS"]
    bs = params["BATCH_SIZE"]
    lr = params["LEARNING_RATE"]

    # Dataset作成
    train_datasets = [SensorDataset(df, scaler_x, scaler_y, ws) for df in train_dfs if len(df) > ws]
    val_datasets = [SensorDataset(df, scaler_x, scaler_y, ws) for df in val_dfs if len(df) > ws]
    
    if not train_datasets: return float("inf")
    
    train_loader = DataLoader(torch.utils.data.ConcatDataset(train_datasets), batch_size=bs, shuffle=True)
    val_loader = DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size=bs, shuffle=False)

    # モデル
    model = LSTMAutoencoder(3, hs, nl, 2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 学習 (少なめのEpochで比較)
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    
    # 評価
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in val_loader:
            loss = criterion(model(x), y)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
            
    return total_loss / max(count, 1)


# ==========================================
# メイン
# ==========================================
def main():
    for csv_name in TEST_CSV_LIST:
        eval_csv(csv_name)
    
    # データ読み込み (共通部分)
    print("Loading Data...")
    dfs = []
    for name in CSV_LIST:
        p = resolve_csv_path(name)
        if p:
            try:
                df = pd.read_csv(p, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
                if not df.empty: dfs.append(df)
            except: pass
            
    if not dfs:
        print("No Data Found.")
        return

    # Train/Val 分割 (最後の2ファイルを検証用にする)
    train_dfs = dfs[:-2]
    val_dfs = dfs[-2:]
    
    # スケーラー作成
    full_df = pd.concat(train_dfs, ignore_index=True)
    scaler_x = MinMaxScaler().fit(full_df[["MovingRange", "HeartRate", "BreathingRate"]])
    scaler_y = MinMaxScaler().fit(full_df[["HeartRate", "BreathingRate"]])

    # グリッドサーチ実行
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Start Grid Search: {len(combinations)} combinations")
    
    results = []
    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Testing: {params}")
        try:
            score = train_evaluate(params, train_dfs, val_dfs, scaler_x, scaler_y)
            print(f"  -> Val Loss: {score:.6f}")
            results.append((score, params))
        except Exception as e:
            print(f"  -> Error: {e}")
            results.append((float("inf"), params))

    # ベストを探す
    results.sort(key=lambda x: x[0])
    best_score, best_params = results[0]
    
    print("\n=== Result ===")
    print(f"Best Val Loss: {best_score:.6f}")
    print(f"Best Params: {best_params}")


if __name__ == "__main__":
    main()

