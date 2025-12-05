import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


CSV_PATH = "sensor_data_20251203_000616.csv"
WINDOW_SIZE = 10


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

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def main():
    csv_path = os.path.join(os.getcwd(), CSV_PATH)
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    X, y = make_supervised(df, window_size=WINDOW_SIZE)

    # 特徴量のみスケーリング
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)

    # 時系列の前半 80% を train、後半 20% を test にする
    n_total = len(X_scaled)
    n_train = int(n_total * 0.8)
    X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    base_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model = MultiOutputRegressor(base_model)

    model.fit(X_train, y_train)

    # RMSE
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    rmse_train_heart = mean_squared_error(y_train[:, 0], pred_train[:, 0]) ** 0.5
    rmse_train_breath = mean_squared_error(y_train[:, 1], pred_train[:, 1]) ** 0.5
    rmse_test_heart = mean_squared_error(y_test[:, 0], pred_test[:, 0]) ** 0.5
    rmse_test_breath = mean_squared_error(y_test[:, 1], pred_test[:, 1]) ** 0.5

    print("=== RMSE (HeartRate) ===")
    print(f"Train RMSE: {rmse_train_heart:.3f}")
    print(f"Test  RMSE: {rmse_test_heart:.3f}")
    print("=== RMSE (BreathingRate) ===")
    print(f"Train RMSE: {rmse_train_breath:.3f}")
    print(f"Test  RMSE: {rmse_test_breath:.3f}")

    # 時系列順で可視化
    heart_true = y_test[:, 0]
    breath_true = y_test[:, 1]
    heart_pred = pred_test[:, 0]
    breath_pred = pred_test[:, 1]

    idx_offset = WINDOW_SIZE + 1 + n_train
    time_idx = np.arange(idx_offset, idx_offset + len(heart_true))

    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, heart_true, label="True HeartRate")
    plt.plot(time_idx, heart_pred, label="Pred HeartRate (LightGBM)", alpha=0.7)
    plt.xlabel("Time index (relative)")
    plt.ylabel("HeartRate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, breath_true, label="True BreathingRate")
    plt.plot(time_idx, breath_pred, label="Pred BreathingRate (LightGBM)", alpha=0.7)
    plt.xlabel("Time index (relative)")
    plt.ylabel("BreathingRate")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()