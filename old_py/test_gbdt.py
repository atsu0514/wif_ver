from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# =========================
# 設定（必要ならここだけ編集）
# =========================
WINDOW_SIZE = 15

# テストしたいCSV（1つだけでもOK）
TEST_CSV_LIST = [
    "sensor_data_20251203_232723.csv",
    # "cleaned_data/cleaned_sensor_data_20251203_232723.csv",
]

# 省略時は meta.txt の model= を使用（なければエラー）
MODEL_NAME = None  # "lightgbm" / "xgboost" / "catboost" / None

HR_TOL = 5.0  # bpm
BR_TOL = 3.0  # rpm

PLOT = True

FEATURE_COLS = ["MovingRange", "HeartRate", "BreathingRate"]
TARGET_COLS = ["HeartRate", "BreathingRate"]


# =========================
# 成果物の解決（latest.txt）
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "GBDT"
DATA_DIR = BASE_DIR / "cleaned_data"
LATEST_PATH = MODELS_DIR / "latest.txt"


def resolve_csv_path(name: str) -> Path | None:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        DATA_DIR / name,   # cleaned_data優先
        BASE_DIR / name,   # ルート直下も一応見る
        Path.cwd() / name  # 実行場所互換
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _resolve_artifact_dir() -> Path:
    if not LATEST_PATH.exists():
        raise FileNotFoundError(f"latest.txt が見つかりません: {LATEST_PATH}")
    run_id = LATEST_PATH.read_text(encoding="utf-8").strip()
    d = MODELS_DIR / run_id
    if not d.is_dir():
        raise FileNotFoundError(f"artifact_dir not found: {d}")
    return d


def build_supervised_xy(df: pd.DataFrame, window_size: int):
    feats = df[FEATURE_COLS].to_numpy(dtype=float)
    tars = df[TARGET_COLS].to_numpy(dtype=float)

    n = len(df)
    if n <= window_size:
        return None, None

    X = []
    y = []
    for i in range(window_size, n):
        window = feats[i - window_size:i]
        X.append(window.reshape(-1))
        y.append(tars[i])
    return np.asarray(X), np.asarray(y)


def match_rate_abs(true_vals: np.ndarray, pred_vals: np.ndarray, tol_abs: float) -> float:
    abs_err = np.abs(np.asarray(true_vals) - np.asarray(pred_vals))
    return float((abs_err <= tol_abs).mean() * 100.0)


def _resolve_model_name(artifact_dir: Path) -> str:
    if MODEL_NAME is not None:
        return MODEL_NAME

    meta_path = artifact_dir / "meta.txt"
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("model="):
                return line.strip().split("=", 1)[1]

    raise RuntimeError("modelが特定できません。MODEL_NAME を設定するか、meta.txt に model= を入れてください。")


def eval_csv(artifact_dir: Path, model_name: str, model, scaler_x, scaler_y, threshold: float, csv_name: str):
    csv_path = resolve_csv_path(csv_name)
    if csv_path is None:
        print(f"警告: 見つかりません: {csv_name}")
        return

    df = pd.read_csv(str(csv_path), parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    for c in ["Timestamp", *FEATURE_COLS]:
        if c not in df.columns:
            raise ValueError(f"{csv_name} に必要列 {c} がありません")

    X, y_true = build_supervised_xy(df, WINDOW_SIZE)
    if X is None:
        print(f"{csv_name}: データが短く評価できません。")
        return

    Xs = scaler_x.transform(X)
    y_true_scaled = scaler_y.transform(y_true)

    y_pred_scaled = model.predict(Xs)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    hr_true, br_true = y_true[:, 0], y_true[:, 1]
    hr_pred, br_pred = y_pred[:, 0], y_pred[:, 1]

    hr_mae = float(np.mean(np.abs(hr_true - hr_pred)))
    hr_rmse = float(np.sqrt(np.mean((hr_true - hr_pred) ** 2)))
    br_mae = float(np.mean(np.abs(br_true - br_pred)))
    br_rmse = float(np.sqrt(np.mean((br_true - br_pred) ** 2)))

    hr_match = match_rate_abs(hr_true, hr_pred, tol_abs=HR_TOL)
    br_match = match_rate_abs(br_true, br_pred, tol_abs=BR_TOL)

    scores = np.mean((y_pred_scaled - y_true_scaled) ** 2, axis=1)
    flags = scores > threshold

    print(f"\n== {csv_name} ==")
    print(f"Artifacts: {artifact_dir}")
    print(f"Model: {model_name}")
    print(f"Threshold (scaled MSE): {threshold:.6f}")
    print(f"Samples evaluated: {len(y_true)}")
    print(f"Heart  MAE={hr_mae:.3f}, RMSE={hr_rmse:.3f}, Match(±{HR_TOL}bpm)={hr_match:.1f}%")
    print(f"Breath MAE={br_mae:.3f}, RMSE={br_rmse:.3f}, Match(±{BR_TOL}rpm)={br_match:.1f}%")
    print(f"Anomaly rate: {float(flags.mean() * 100):.2f}%")

    if PLOT:
        x = np.arange(len(y_true))

        plt.figure(figsize=(10, 4))
        plt.plot(x, hr_true, label="True HR")
        plt.plot(x, hr_pred, label="Pred HR", alpha=0.7)
        plt.title(f"HeartRate: {csv_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x, br_true, label="True BR")
        plt.plot(x, br_pred, label="Pred BR", alpha=0.7)
        plt.title(f"BreathingRate: {csv_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(scores, label="Anomaly Score (scaled MSE)", color="red")
        plt.axhline(y=threshold, color="green", linestyle="--", label=f"Threshold={threshold:.6f}")
        plt.title(f"Anomaly Score: {csv_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    artifact_dir = _resolve_artifact_dir()
    model_name = _resolve_model_name(artifact_dir)

    model_path = artifact_dir / f"{model_name}_model.pkl"
    scaler_x_path = artifact_dir / "scaler_x.pkl"
    scaler_y_path = artifact_dir / "scaler_y.pkl"
    threshold_path = artifact_dir / "threshold.txt"

    model = joblib.load(str(model_path))
    scaler_x = joblib.load(str(scaler_x_path))
    scaler_y = joblib.load(str(scaler_y_path))
    threshold = float(threshold_path.read_text(encoding="utf-8"))

    for csv_name in TEST_CSV_LIST:
        eval_csv(artifact_dir, model_name, model, scaler_x, scaler_y, threshold, csv_name)


if __name__ == "__main__":
    main()