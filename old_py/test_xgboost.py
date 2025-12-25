import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


# 学習とテストを同じファイルから 8:2 で分割する例
CSV_PATH = "sensor_data_20251203_232723.csv"
WINDOW_SIZE = 100  # 何ステップ分の履歴を特徴量に含めるか


def make_supervised(df: pd.DataFrame, window_size: int = 10):
    """
    時系列データを教師あり学習用の X, y に変換する。
    X: 過去 window_size ステップ分の特徴量を横に並べたもの
       (Presence, Movement, MovingRange, BreathingRate)
    y: 直後1ステップの [HeartRate, BreathingRate] の2次元
    """
    features = df[["Presence", "Movement", "MovingRange", "BreathingRate"]].values
    heart = df["HeartRate"].values
    breath = df["BreathingRate"].values

    X_list = []
    y_list = []

    for i in range(len(df) - window_size - 1):
        window_feat = features[i: i + window_size].reshape(-1)
        X_list.append(window_feat)
        y_list.append([heart[i + window_size], breath[i + window_size]])

    X = np.array(X_list)   # shape: (N, 4*window_size)
    y = np.array(y_list)   # shape: (N, 2)
    return X, y


def calc_match_rate(true, pred, tol_ratio=0.10):
    """
    真値の±tol_ratio (例: 0.10=±10%) 以内を「一致」とみなした割合(%)を返す。
    """
    true = np.array(true)
    pred = np.array(pred)
    abs_err = np.abs(true - pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = abs_err / np.where(true == 0, 1, np.abs(true))
    match = (rel_err <= tol_ratio)
    return match.mean() * 100.0


def main():
    csv_path = os.path.join(os.getcwd(), CSV_PATH)
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # 教師ありデータに変換
    X, y = make_supervised(df, window_size=WINDOW_SIZE)

    # 時系列の前半 80% を train、後半 20% を test にする
    n_total = len(X)
    n_train = int(n_total * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # スケーリング（特徴量のみ）：fit は train のみ
    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    # XGBoost モデル（2出力回帰）
    base_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model = MultiOutputRegressor(base_model)

    # 学習
    model.fit(X_train_scaled, y_train)

    # 学習データでのRMSE
    pred_train = model.predict(X_train_scaled)
    mse_train_heart = mean_squared_error(y_train[:, 0], pred_train[:, 0])
    mse_train_breath = mean_squared_error(y_train[:, 1], pred_train[:, 1])
    rmse_train_heart = mse_train_heart ** 0.5
    rmse_train_breath = mse_train_breath ** 0.5

    # テストデータでのRMSE
    pred_test = model.predict(X_test_scaled)
    mse_test_heart = mean_squared_error(y_test[:, 0], pred_test[:, 0])
    mse_test_breath = mean_squared_error(y_test[:, 1], pred_test[:, 1])
    rmse_test_heart = mse_test_heart ** 0.5
    rmse_test_breath = mse_test_breath ** 0.5

    print("=== RMSE (HeartRate) [XGBoost] ===")
    print(f"Train RMSE: {rmse_train_heart:.3f}")
    print(f"Test  RMSE: {rmse_test_heart:.3f}")
    print("=== RMSE (BreathingRate) [XGBoost] ===")
    print(f"Train RMSE: {rmse_train_breath:.3f}")
    print(f"Test  RMSE: {rmse_test_breath:.3f}")

    # テストデータの真値と予測
    heart_true = y_test[:, 0]
    breath_true = y_test[:, 1]
    heart_pred = pred_test[:, 0]
    breath_pred = pred_test[:, 1]

    # 一致率（±10%以内）
    heart_match = calc_match_rate(heart_true, heart_pred, tol_ratio=0.10)
    breath_match = calc_match_rate(breath_true, breath_pred, tol_ratio=0.10)

    print("=== Match Rate (±10%以内, XGBoost, Test Only) ===")
    print(f"HeartRate  match: {heart_match:.1f}%")
    print(f"Breathing match: {breath_match:.1f}%")

    # グラフ用インデックス（test部分）
    idx_offset = WINDOW_SIZE + 1 + n_train
    time_idx = np.arange(idx_offset, idx_offset + len(heart_true))

    # 心拍数の真値 vs 予測
    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, heart_true, label="True HeartRate (test)")
    plt.plot(time_idx, heart_pred, label="Pred HeartRate (XGBoost, test)", alpha=0.7)
    plt.xlabel("Time index (relative)")
    plt.ylabel("HeartRate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 呼吸数の真値 vs 予測
    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, breath_true, label="True BreathingRate (test)")
    plt.plot(time_idx, breath_pred, label="Pred BreathingRate (XGBoost, test)", alpha=0.7)
    plt.xlabel("Time index (relative)")
    plt.ylabel("BreathingRate")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()