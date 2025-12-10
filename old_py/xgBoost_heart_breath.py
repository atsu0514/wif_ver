import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


TRAIN_CSV = "sensor_data_20251203_000616.csv"  # 学習用CSV
TEST_CSV  = "sensor_data_20251126_143734.csv"   # テスト用CSV
WINDOW_SIZE = 50  # 何ステップ分の履歴を特徴量に含めるか


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

    X = np.array(X_list)          # shape: (N, 4*window_size)
    y = np.array(y_list)          # shape: (N, 2)
    return X, y


def main():
    # === CSV 読み込み（学習用・テスト用） ===
    train_path = os.path.join(os.getcwd(), TRAIN_CSV)
    test_path  = os.path.join(os.getcwd(), TEST_CSV)

    df_train = pd.read_csv(train_path, parse_dates=["Timestamp"])
    df_test  = pd.read_csv(test_path,  parse_dates=["Timestamp"])

    df_train = df_train.sort_values("Timestamp").reset_index(drop=True)
    df_test  = df_test.sort_values("Timestamp").reset_index(drop=True)

    # === 教師ありデータに変換 ===
    X_train, y_train = make_supervised(df_train, window_size=WINDOW_SIZE)
    X_test,  y_test  = make_supervised(df_test,  window_size=WINDOW_SIZE)

    # スケーリング（特徴量のみ）: 学習データで fit → 両方に transform
    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled  = scaler_x.transform(X_test)

    # === XGBoost モデル定義（2出力回帰） ===
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

    # === 学習 ===
    model.fit(X_train_scaled, y_train)

    # === 学習データでのRMSE ===
    pred_train = model.predict(X_train_scaled)
    mse_train_heart = mean_squared_error(y_train[:, 0], pred_train[:, 0])
    mse_train_breath = mean_squared_error(y_train[:, 1], pred_train[:, 1])
    rmse_train_heart = mse_train_heart ** 0.5
    rmse_train_breath = mse_train_breath ** 0.5

    # === テストデータでのRMSE ===
    pred_test = model.predict(X_test_scaled)
    mse_test_heart = mean_squared_error(y_test[:, 0], pred_test[:, 0])
    mse_test_breath = mean_squared_error(y_test[:, 1], pred_test[:, 1])
    rmse_test_heart = mse_test_heart ** 0.5
    rmse_test_breath = mse_test_breath ** 0.5

    print("=== RMSE (HeartRate) ===")
    print(f"Train RMSE: {rmse_train_heart:.3f}")
    print(f"Test  RMSE: {rmse_test_heart:.3f}")
    print("=== RMSE (BreathingRate) ===")
    print(f"Train RMSE: {rmse_train_breath:.3f}")
    print(f"Test  RMSE: {rmse_test_breath:.3f}")

    # === テストデータでの予測 vs 実測の可視化 ===
    heart_true = y_test[:, 0]
    breath_true = y_test[:, 1]
    heart_pred = pred_test[:, 0]
    breath_pred = pred_test[:, 1]

    # 可視化用のインデックス（テストデータ内の相対位置）
    time_idx = np.arange(WINDOW_SIZE + 1, WINDOW_SIZE + 1 + len(heart_true))

    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, heart_true, label="True HeartRate")
    plt.plot(time_idx, heart_pred, label="Pred HeartRate (XGBoost)", alpha=0.7)
    plt.xlabel("Time index (test, relative)")
    plt.ylabel("HeartRate")
    plt.title("HeartRate: True vs Predicted (Test, XGBoost)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(time_idx, breath_true, label="True BreathingRate")
    plt.plot(time_idx, breath_pred, label="Pred BreathingRate (XGBoost)", alpha=0.7)
    plt.xlabel("Time index (test, relative)")
    plt.ylabel("BreathingRate")
    plt.title("BreathingRate: True vs Predicted (Test, XGBoost)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()