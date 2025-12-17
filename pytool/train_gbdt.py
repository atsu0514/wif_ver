import os
import argparse
import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor

# ===== 設定（必要なら編集）=====
WINDOW_SIZE = 15
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

FEATURE_COLS = ["MovingRange", "HeartRate", "BreathingRate"]
TARGET_COLS = ["HeartRate", "BreathingRate"]

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "GBDT")  # 保存先フォルダ
LATEST_PATH = os.path.join(MODELS_DIR, "latest.txt")


def build_supervised_xy(df: pd.DataFrame, window_size: int):
    feats = df[FEATURE_COLS].to_numpy(dtype=float)
    tars = df[TARGET_COLS].to_numpy(dtype=float)

    n = len(df)
    if n <= window_size:
        return None, None

    X = []
    y = []
    for i in range(window_size, n):
        window = feats[i - window_size:i]         # [W, 3]
        X.append(window.reshape(-1))              # [W*3]
        y.append(tars[i])                         # [2]
    return np.asarray(X), np.asarray(y)


def make_model(model_name: str, seed: int):
    model_name = model_name.lower()

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        base = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=seed,
        )
        return MultiOutputRegressor(base)

    if model_name == "xgboost":
        from xgboost import XGBRegressor
        base = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            objective="reg:squarederror",
            n_jobs=0,
        )
        return MultiOutputRegressor(base)

    if model_name == "catboost":
        from catboost import CatBoostRegressor
        base = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
        )
        return MultiOutputRegressor(base)

    raise ValueError("model must be one of: lightgbm, xgboost, catboost")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lightgbm", "xgboost", "catboost"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold-percentile", type=float, default=99.0)
    args = ap.parse_args()

    # 1) データ読み込み（trainのみ）
    train_df_list = []
    for name in CSV_LIST:
        if name == TEST_CSV:
            continue
        path = os.path.join(BASE_DIR, name)
        if not os.path.exists(path):
            print(f"警告: {name} が見つかりません。スキップします。")
            continue
        df = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        train_df_list.append(df)

    if not train_df_list:
        raise RuntimeError("学習用CSVがありません。CSV_LIST/TEST_CSV を確認してください。")

    full_train_df = pd.concat(train_df_list, ignore_index=True)

    # 2) X/Y を作成（窓を平坦化した教師ありデータ）
    X_train, y_train = build_supervised_xy(full_train_df, WINDOW_SIZE)
    if X_train is None:
        raise RuntimeError("学習用サンプルが作れません（データが短すぎる等）。")

    # 3) スケーラ（LSTM/GRUと同じくyはスケールして誤差・閾値計算に使う）
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # 4) モデル学習
    model = make_model(args.model, seed=args.seed)
    print(f"Training {args.model} on {X_train_scaled.shape} ...")
    model.fit(X_train_scaled, y_train_scaled)

    # 5) train誤差分布から閾値（scaled MSE）
    pred_train_scaled = model.predict(X_train_scaled)              # [N, 2]
    train_losses = np.mean((pred_train_scaled - y_train_scaled) ** 2, axis=1)  # [N]
    threshold = float(np.percentile(train_losses, args.threshold_percentile))
    print(f"Threshold (p{args.threshold_percentile}): {threshold:.6f}")

    # 6) 成果物保存（run_idディレクトリに保存）
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = os.path.join(MODELS_DIR, run_id)
    os.makedirs(artifact_dir, exist_ok=True)

    joblib.dump(model, os.path.join(artifact_dir, f"{args.model}_model.pkl"))
    joblib.dump(scaler_x, os.path.join(artifact_dir, "scaler_x.pkl"))
    joblib.dump(scaler_y, os.path.join(artifact_dir, "scaler_y.pkl"))
    with open(os.path.join(artifact_dir, "threshold.txt"), "w", encoding="utf-8") as f:
        f.write(str(threshold))
    with open(os.path.join(artifact_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"model={args.model}\nwindow_size={WINDOW_SIZE}\nfeatures={FEATURE_COLS}\n")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        f.write(run_id)

    print(f"Saved artifacts to: {artifact_dir}")
    print(f"Marked latest: {run_id}")


if __name__ == "__main__":
    main()