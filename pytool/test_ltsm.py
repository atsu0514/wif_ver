import os
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt


# =========================
# 設定（必要ならここだけ編集）
# =========================
WINDOW_SIZE = 15
HIDDEN_SIZE = 32
NUM_LAYERS = 1

# テストしたいCSV（1つだけでもOK）
TEST_CSV_LIST = [
    "sensor_data_20251203_232723.csv",
]

FEATURE_COLS = ["MovingRange", "HeartRate", "BreathingRate"]   # train/realtime と同じ
TARGET_COLS = ["HeartRate", "BreathingRate"]


# =========================
# モデル成果物の解決（realtime と同じ）
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # プロジェクトルート
MODELS_DIR = os.path.join(BASE_DIR, "models")
LATEST_PATH = os.path.join(MODELS_DIR, "latest.txt")


def _resolve_artifact_dir() -> str:
    if not os.path.exists(LATEST_PATH):
        raise FileNotFoundError(f"latest.txt が見つかりません: {LATEST_PATH}")
    with open(LATEST_PATH, "r", encoding="utf-8") as f:
        run_id = f.read().strip()
    artifact_dir = os.path.join(MODELS_DIR, run_id)
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"artifact_dir が見つかりません: {artifact_dir}")
    return artifact_dir


ARTIFACT_DIR = _resolve_artifact_dir()
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lstm_model.pth")
SCALER_X_PATH = os.path.join(ARTIFACT_DIR, "scaler_x.pkl")  # pkl（質問のpkiはたぶんtypo）
SCALER_Y_PATH = os.path.join(ARTIFACT_DIR, "scaler_y.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.txt")


# =========================
# モデル定義（train/realtime と同じ構造）
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# =========================
# realtime_ltsm_line と同じ誤差計算で評価するクラス
# =========================
class RealtimeLikeOfflineEvaluator:
    """
    realtime_ltsm_line.py と同じロジック：
      - WINDOW_SIZE たまったら「次の心拍/呼吸」を予測
      - 1ステップ前の予測(last_prediction_scaled) と 今回実測(target_scaled) のMSEを anomaly_score とする
      - anomaly_score > threshold なら異常
    """
    def __init__(self):
        self.scaler_x = joblib.load(SCALER_X_PATH)
        self.scaler_y = joblib.load(SCALER_Y_PATH)

        self.model = LSTMModel(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            self.threshold = float(f.read())

        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.last_prediction_scaled = None  # 1ステップ前に予測した「現在」のscaled値（次のstepで評価される）

    def step(self, feature_row_original: np.ndarray):
        """
        feature_row_original: [MovingRange, HeartRate, BreathingRate] (オリジナルスケール)
        戻り値:
          - warmup中は None
          - それ以外は dict（pred_next と pred_current、anomaly_score など）
        """
        self.buffer.append(feature_row_original.astype(float))
        if len(self.buffer) < WINDOW_SIZE:
            return None

        input_data = np.array(self.buffer)  # [W, 3]
        input_scaled = self.scaler_x.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)  # [1, W, 3]

        with torch.no_grad():
            pred_next_scaled = self.model(input_tensor).numpy()[0]  # [2] scaled

        pred_next_original = self.scaler_y.inverse_transform([pred_next_scaled])[0]  # [2] original

        anomaly_score = None
        is_anomaly = None
        pred_current_original = None

        # realtime と同じ：前回の予測と今回実測のズレ（scaled空間MSE）
        if self.last_prediction_scaled is not None:
            current_target_original = np.array([feature_row_original[1], feature_row_original[2]]).reshape(1, -1)
            current_target_scaled = self.scaler_y.transform(current_target_original)[0]  # [2]

            mse = float(np.mean((self.last_prediction_scaled - current_target_scaled) ** 2))
            anomaly_score = mse
            is_anomaly = bool(anomaly_score > self.threshold)

            pred_current_original = self.scaler_y.inverse_transform([self.last_prediction_scaled])[0]  # [2]

        # 次stepで評価される「今回の予測」を保存
        self.last_prediction_scaled = pred_next_scaled

        return {
            "pred_next_heart": float(pred_next_original[0]),
            "pred_next_breath": float(pred_next_original[1]),
            "pred_current_heart": None if pred_current_original is None else float(pred_current_original[0]),
            "pred_current_breath": None if pred_current_original is None else float(pred_current_original[1]),
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "threshold": float(self.threshold),
        }

# ±10%で誤差を許容
# def match_rate_percent(true_vals: np.ndarray, pred_vals: np.ndarray, tol_ratio: float = 0.10) -> float:
#     true_vals = np.asarray(true_vals, dtype=float)
#     pred_vals = np.asarray(pred_vals, dtype=float)
#     abs_err = np.abs(true_vals - pred_vals)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         rel_err = abs_err / np.where(true_vals == 0, 1.0, np.abs(true_vals))
#     return float((rel_err <= tol_ratio).mean() * 100.0)

# ±3rpmで誤差を許容
def match_rate_percent(true_vals: np.ndarray, pred_vals: np.ndarray, tol_abs: float = 3.0) -> float:
    """
    真値の±tol_abs（例: 3.0）以内を「一致」とみなした割合(%)。
    """
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    abs_err = np.abs(true_vals - pred_vals)
    return float((abs_err <= tol_abs).mean() * 100.0)


def eval_csv(csv_name: str):
    csv_path = os.path.join(BASE_DIR, csv_name)
    if not os.path.exists(csv_path):
        print(f"警告: 見つかりません: {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    for c in ["Timestamp", *FEATURE_COLS]:
        if c not in df.columns:
            raise ValueError(f"{csv_name} に必要列 {c} がありません")

    evaluator = RealtimeLikeOfflineEvaluator()
    print(f"\n== {csv_name} ==")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print(f"Threshold (scaled MSE): {evaluator.threshold:.6f}")

    ts_list = []
    true_hr_list = []
    pred_hr_list = []
    true_br_list = []
    pred_br_list = []
    score_list = []
    flag_list = []

    for _, r in df.iterrows():
        features = r[FEATURE_COLS].to_numpy(dtype=float)  # [MovingRange, HeartRate, BreathingRate]
        res = evaluator.step(features)
        if res is None:
            continue

        # 評価できるのは「1ステップ前の予測（pred_current_*） vs 今回実測」
        if res["pred_current_heart"] is None:
            continue

        ts_list.append(r["Timestamp"])
        true_hr_list.append(float(r["HeartRate"]))
        pred_hr_list.append(float(res["pred_current_heart"]))
        true_br_list.append(float(r["BreathingRate"]))
        pred_br_list.append(float(res["pred_current_breath"]))

        score = float(res["anomaly_score"]) if res["anomaly_score"] is not None else np.nan
        is_anom = bool(res["is_anomaly"]) if res["is_anomaly"] is not None else False
        score_list.append(score)
        flag_list.append(is_anom)

    if len(true_hr_list) == 0:
        print("評価サンプルが0です（データが短い/欠損など）。")
        return

    true_hr = np.array(true_hr_list)
    pred_hr = np.array(pred_hr_list)
    true_br = np.array(true_br_list)
    pred_br = np.array(pred_br_list)
    scores = np.array(score_list, dtype=float)
    flags = np.array(flag_list, dtype=bool)

    # 予測精度（オリジナルスケール）
    heart_mae = float(np.mean(np.abs(true_hr - pred_hr)))
    heart_rmse = float(np.sqrt(np.mean((true_hr - pred_hr) ** 2)))
    breath_mae = float(np.mean(np.abs(true_br - pred_br)))
    breath_rmse = float(np.sqrt(np.mean((true_br - pred_br) ** 2)))

    heart_match10 = match_rate_percent(true_hr, pred_hr, tol_abs=5.0)
    breath_match10 = match_rate_percent(true_br, pred_br, tol_abs=3.0)

    print(f"Samples evaluated: {len(true_hr)}")
    print(f"Heart  MAE={heart_mae:.3f}, RMSE={heart_rmse:.3f}, Match(±5rpm)={heart_match10:.1f}%")
    print(f"Breath MAE={breath_mae:.3f}, RMSE={breath_rmse:.3f}, Match(±3rpm)={breath_match10:.1f}%")
    print(f"Anomaly rate: {float(flags.mean() * 100):.2f}%")

    # 可視化（必要なければ消してOK）
    x = np.arange(len(true_hr))

    plt.figure(figsize=(10, 4))
    plt.plot(x, true_hr, label="True HeartRate")
    plt.plot(x, pred_hr, label="Pred HeartRate", alpha=0.7)
    plt.title(f"HeartRate: {csv_name}")
    plt.xlabel("Sample index")
    plt.ylabel("HeartRate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x, true_br, label="True BreathingRate")
    plt.plot(x, pred_br, label="Pred BreathingRate", alpha=0.7)
    plt.title(f"BreathingRate: {csv_name}")
    plt.xlabel("Sample index")
    plt.ylabel("BreathingRate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Anomaly Score (scaled MSE)", color="red")
    plt.axhline(y=evaluator.threshold, color="green", linestyle="--", label=f"Threshold={evaluator.threshold:.6f}")
    plt.title(f"Anomaly Score: {csv_name}")
    plt.xlabel("Sample index")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    for csv_name in TEST_CSV_LIST:
        eval_csv(csv_name)


if __name__ == "__main__":
    main()