import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


CSV_PATH = "sensor_data_20251203_000616.csv"  # パスは必要に応じて調整
WINDOW_SIZE = 20  # 何ステップ分の履歴を特徴量に含めるか


def make_supervised(df: pd.DataFrame, window_size: int = 10):
    """
    時系列データを「教師あり学習」用の X, y に変換する。
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
        # 過去 window_size ステップ分の4特徴量をすべて連結 → 4*window_size 次元
        window_feat = features[i : i + window_size].reshape(-1)
        X_list.append(window_feat)

        # その直後の HeartRate と BreathingRate を目的変数に
        y_list.append([heart[i + window_size], breath[i + window_size]])

    X = np.array(X_list)          # shape: (N, 4*window_size)
    y = np.array(y_list)          # shape: (N, 2)
    return X, y


def detect_anomalies(heart_true, breath_true, heart_pred, breath_pred,
                     z_th_heart=3.0, z_th_breath=3.0):
    """
    予測誤差から簡易な異常フラグを立てる。
    - 各系列の誤差を z-score 化し、しきい値を超えたら異常と判定。
    """
    # 誤差
    err_heart = heart_true - heart_pred
    err_breath = breath_true - breath_pred

    # 平均・標準偏差
    mu_h = np.mean(err_heart)
    sd_h = np.std(err_heart) + 1e-8
    mu_b = np.mean(err_breath)
    sd_b = np.std(err_breath) + 1e-8

    # z-score
    z_heart = (err_heart - mu_h) / sd_h
    z_breath = (err_breath - mu_b) / sd_b

    # フラグ（どちらかがしきい値を超えたら異常）
    abnormal_heart = np.abs(z_heart) > z_th_heart
    abnormal_breath = np.abs(z_breath) > z_th_breath
    abnormal = np.logical_or(abnormal_heart, abnormal_breath)

    return z_heart, z_breath, abnormal


def main():
    # === CSV 読み込み ===
    csv_path = os.path.join(os.getcwd(), CSV_PATH)  # カレントディレクトリ基準
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # === 教師ありデータに変換 ===
    X, y = make_supervised(df, window_size=WINDOW_SIZE)

    # スケーリング（特徴量のみ）
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)

    # 学習 / 検証に分割
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=True, random_state=42
    )

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
    model.fit(X_train, y_train)

    # === 評価（RMSE） ===
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    mse_train_heart = mean_squared_error(y_train[:, 0], pred_train[:, 0])
    mse_val_heart = mean_squared_error(y_val[:, 0], pred_val[:, 0])
    rmse_train_heart = mse_train_heart ** 0.5
    rmse_val_heart = mse_val_heart ** 0.5

    mse_train_breath = mean_squared_error(y_train[:, 1], pred_train[:, 1])
    mse_val_breath = mean_squared_error(y_val[:, 1], pred_val[:, 1])
    rmse_train_breath = mse_train_breath ** 0.5
    rmse_val_breath = mse_val_breath ** 0.5

    print("=== RMSE (HeartRate) ===")
    print(f"Train RMSE: {rmse_train_heart:.3f}")
    print(f"Val   RMSE: {rmse_val_heart:.3f}")
    print("=== RMSE (BreathingRate) ===")
    print(f"Train RMSE: {rmse_train_breath:.3f}")
    print(f"Val   RMSE: {rmse_val_breath:.3f}")

    # === 全データで予測 ===
    X_all_scaled = scaler_x.transform(X)
    pred_all = model.predict(X_all_scaled)

    heart_true = y[:, 0]
    breath_true = y[:, 1]
    heart_pred = pred_all[:, 0]
    breath_pred = pred_all[:, 1]

    # === 異常検知 ===
    z_heart, z_breath, abnormal = detect_anomalies(
        heart_true, breath_true, heart_pred, breath_pred,
        z_th_heart=3.0, z_th_breath=3.0,
    )

    print(f"異常検知されたポイント数: {abnormal.sum()} / {len(abnormal)}")

    # === 結果を DataFrame にまとめる（必要ならCSV保存も可） ===
    result_df = pd.DataFrame({
        "Heart_true": heart_true,
        "Heart_pred": heart_pred,
        "Breath_true": breath_true,
        "Breath_pred": breath_pred,
        "z_heart": z_heart,
        "z_breath": z_breath,
        "abnormal": abnormal.astype(int),
    })
    # 例: result_df.to_csv("heart_breath_anomaly_result.csv", index=False)

    # 可視化用インデックス
    time_idx = np.arange(WINDOW_SIZE + 1, WINDOW_SIZE + 1 + len(heart_true))

    # === 心拍数: 真値 vs 予測 + 異常マーカー ===
    plt.figure(figsize=(12, 4))
    plt.plot(time_idx, heart_true, label="True HeartRate")
    plt.plot(time_idx, heart_pred, label="Pred HeartRate (XGBoost)", alpha=0.7)

    # 異常ポイントを赤丸で表示
    idx_abn = time_idx[abnormal]
    heart_abn = heart_true[abnormal]
    plt.scatter(idx_abn, heart_abn, color="red", label="Anomaly", zorder=5)

    plt.xlabel("Time index (relative)")
    plt.ylabel("HeartRate")
    plt.title("HeartRate: True vs Predicted + Anomaly")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === 呼吸数: 真値 vs 予測 + 異常マーカー ===
    plt.figure(figsize=(12, 4))
    plt.plot(time_idx, breath_true, label="True BreathingRate")
    plt.plot(time_idx, breath_pred, label="Pred BreathingRate (XGBoost)", alpha=0.7)

    breath_abn = breath_true[abnormal]
    plt.scatter(idx_abn, breath_abn, color="red", label="Anomaly", zorder=5)

    plt.xlabel("Time index (relative)")
    plt.ylabel("BreathingRate")
    plt.title("BreathingRate: True vs Predicted + Anomaly")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === 異常スコア（z-score）の推移プロット ===
    plt.figure(figsize=(12, 4))
    plt.plot(time_idx, z_heart, label="z-score HeartRate")
    plt.plot(time_idx, z_breath, label="z-score BreathingRate")
    plt.axhline(3.0, color="red", linestyle="--", alpha=0.5)
    plt.axhline(-3.0, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Time index (relative)")
    plt.ylabel("z-score of error")
    plt.title("Anomaly Score (z-score of prediction error)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()