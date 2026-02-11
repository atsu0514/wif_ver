# 隠れ層は64のモデルを使用
# 睡眠時の正常データを変更する必要あり（異常が起きてなさそうなデータを探す）
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt


WINDOW_SIZE = 15
HIDDEN_SIZE = 64
NUM_LAYERS = 2

TEST_CSV_LIST = [
    "cleaned_sensor_data_20260107_133513_mukokyuu.csv",
    "cleaned_sensor_data_20260107_141840_negaeri.csv",
    "cleaned_sensor_data_20260110_095603.csv",
    "cleaned_sensor_data_20260111_103644.csv",
    "cleaned_sensor_data_20260115_142922_kakokyuu.csv",
    "cleaned_sensor_data_20260115_145841_kakokyuu.csv",
    "cleaned_sensor_data_20260119_210812mukokyuu.csv",
    "cleaned_sensor_data_20260119_214346_negaeri.csv",
    "cleaned_sensor_data_20260119_222651_mukokyuuy.csv",
    "cleaned_sensor_data_20260122_215339_negaeri.csv",
    "cleaned_sensor_data_20260122_222017_kakokyuu.csv"
    "cleaned_sensor_data_20260116_000242.csv",
    "cleaned_sensor_data_20260124_225707kakokyuu.csv"
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


class LSTMEncoderDecoder(nn.Module):
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

        self.model = LSTMEncoderDecoder(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2)
        self.model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
        self.model.eval()

        # 閾値読み込み（複数行対応）
        lines = THRESHOLD_PATH.read_text(encoding="utf-8").strip().splitlines()
        try:
            self.thresholds = [float(x) for x in lines]
            if len(self.thresholds) == 1:
                # 互換性のため、1つしかなければそれをlow/mid/highすべてにする
                self.thresholds = [self.thresholds[0]] * 3
        except Exception:
            # 万が一読み込めない場合のフォールバック（例: デフォルト値）
            print("Warning: Failed to parse thresholds, using fallback 0.1")
            self.thresholds = [0.1, 0.1, 0.1]
            
        print(f"Loaded thresholds: {self.thresholds}")
        
        # 判定用には一番緩い(値が大きい) High(99%) をメインで使うか、
        # もしくは Low(90%) を使うか決める必要があります。今回はHighを使います。
        self.threshold_main = self.thresholds[-1] 

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

        # ---------------------------------------------------------
        # (修正) 整数丸めによる誤差計算 (Original Scale)
        # ---------------------------------------------------------
        # 1. 逆変換 (W,2)
        recon_original = self.scaler_y.inverse_transform(recon_scaled)
        
        # 2. 整数に丸める
        recon_int = np.round(recon_original)           # 予測値
        true_int = np.round(y_window_original)         # 正解値(もともと整数だが念のため)
        
        # 3. 最後のステップのMSE (Integer Scale)
        # 心拍・呼吸それぞれの差の二乗の平均
        diff = recon_int[-1] - true_int[-1]
        score = float(np.mean(diff ** 2))
        
        # 判定にはメインの閾値を使用
        is_anomaly = bool(score > self.threshold_main)

        # 参考: 最終時刻の復元値（プロット用）
        # 丸めた値をプロット用として保持する
        recon_last_val = recon_int[-1]
        true_last_val = true_int[-1]

        return {
            "anomaly_score": score,
            "is_anomaly": is_anomaly,
            "threshold_low": self.thresholds[0], # 追加
            "threshold_mid": self.thresholds[1], # 追加
            "threshold_high": self.thresholds[2], # 追加
            "recon_last_heart": float(recon_last_val[0]),
            "recon_last_breath": float(recon_last_val[1]),
            "true_last_heart": float(true_last_val[0]),
            "true_last_breath": float(true_last_val[1]),
        }


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
    # print(f"Threshold (scaled recon MSE): {evaluator.threshold:.6f}") # 削除

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

    # 復元精度（オリジナルスケール、最後の点）
    heart_mae = float(np.mean(np.abs(true_hr_arr - recon_hr_arr)))
    heart_rmse = float(np.sqrt(np.mean((true_hr_arr - recon_hr_arr) ** 2)))
    breath_mae = float(np.mean(np.abs(true_br_arr - recon_br_arr)))
    breath_rmse = float(np.sqrt(np.mean((true_br_arr - recon_br_arr) ** 2)))

    heart_match = match_rate_percent(true_hr_arr, recon_hr_arr, tol_abs=0.2)
    breath_match = match_rate_percent(true_br_arr, recon_br_arr, tol_abs=0.2)

    print(f"Samples evaluated: {len(scores)}")
    print(f"Heart  MAE={heart_mae:.3f}, RMSE={heart_rmse:.3f}, Match(±0.2rpm)={heart_match:.1f}%")
    print(f"Breath MAE={breath_mae:.3f}, RMSE={breath_rmse:.3f}, Match(±0.2rpm)={breath_match:.1f}%")
    print(f"Anomaly rate: {float(flags.mean() * 100):.2f}%")

    # 可視化（不要なら消してOK）
    x = np.arange(len(scores))

    # 目盛り設定用の関数
    def set_fine_ticks(ax_obj, x_data):
        # データ点数に応じて間隔を調整 (例: 50点ごと)
        step = max(1, len(x_data) // 30)  # 全体を20分割程度にする
        ax_obj.set_xticks(np.arange(0, len(x_data), step))
        # 縦線グリッドを入れて見やすくする
        ax_obj.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)

    plt.figure(figsize=(10, 4))
    plt.plot(x, true_hr, label="True HeartRate")
    plt.plot(x, recon_hr, label="Recon HeartRate", alpha=0.7)
    plt.title(f"HeartRate (recon last step): {csv_name}")
    plt.xlabel("Sample index")
    plt.ylabel("HeartRate")
    set_fine_ticks(plt.gca(), x)  # 目盛り適用
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x, true_br, label="True BreathingRate")
    plt.plot(x, recon_br, label="Recon BreathingRate", alpha=0.7)
    plt.title(f"BreathingRate (recon last step): {csv_name}")
    plt.xlabel("Sample index")
    plt.ylabel("BreathingRate")
    set_fine_ticks(plt.gca(), x)  # 目盛り適用
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Anomaly Score (Int MSE)", color="red")
    
    # 3本の線を引く
    th_low = evaluator.thresholds[0]
    th_mid = evaluator.thresholds[1]
    th_high = evaluator.thresholds[2]
    
    plt.axhline(y=th_low, color="orange", linestyle=":", label=f"Low (92%)={th_low:.6f}")
    plt.axhline(y=th_mid, color="blue", linestyle="--", label=f"Mid (87%)={th_mid:.6f}")
    plt.axhline(y=th_high, color="green", linestyle="-", label=f"High (95%)={th_high:.6f}")
    
    plt.title(f"Anomaly Score: {csv_name}")
    plt.xlabel("Sample index")
    plt.ylabel("Score")
    set_fine_ticks(plt.gca(), x)  # 目盛り適用
    plt.legend()
    plt.tight_layout()
    plt.show()


def match_rate_percent(true_vals: np.ndarray, pred_vals: np.ndarray, tol_abs: float = 0.5) -> float:
    """
    真値の±tol_abs以内を「一致」とみなした割合(%)。
    """
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    abs_err = np.abs(true_vals - pred_vals)
    return float((abs_err <= tol_abs).mean() * 100.0)


def main():
    for csv_name in TEST_CSV_LIST:
        eval_csv(csv_name)


if __name__ == "__main__":
    main()

