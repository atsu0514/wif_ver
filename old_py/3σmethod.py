import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==== 設定 ====
# 異常検知に使う CSV ファイル
# 例1: センサー生データ + 何らかの「基準値」からの差分を取って使う
# 例2: xgBoost_heart_breath の結果CSV (True/Pred が入っている) を使う
CSV_PATH = "sensor_data_20251203_000616.csv"  # 実際のファイル名に合わせて変更

# しきい値 (3σ)
Z_TH_HEART = 3.0
Z_TH_BREATH = 3.0

WINDOW = 30  # 移動平均の窓サイズ
Z_TH = 3.0   # 3σ


def detect_anomalies_3sigma(heart_true, breath_true, heart_pred, breath_pred,
                            z_th_heart=3.0, z_th_breath=3.0):
    """
    3σ法（3-sigma rule）で異常を検出する。
    - 予測誤差 e = true - pred を計算
    - e の平均 μ, 標準偏差 σ を求める
    - z = (e - μ) / σ の絶対値が 3 を超えたら異常と判定
    """
    # 誤差
    err_heart = heart_true - heart_pred
    err_breath = breath_true - breath_pred

    # 平均・標準偏差（ゼロ除算防止で微小値を足す）
    mu_h = np.mean(err_heart)
    sd_h = np.std(err_heart) + 1e-8
    mu_b = np.mean(err_breath)
    sd_b = np.std(err_breath) + 1e-8

    # z-score
    z_heart = (err_heart - mu_h) / sd_h
    z_breath = (err_breath - mu_b) / sd_b

    # 3σ法: |z| > 3 を異常とする
    abnormal_heart = np.abs(z_heart) > z_th_heart
    abnormal_breath = np.abs(z_breath) > z_th_breath

    # どちらかが異常なら「異常」とする
    abnormal = np.logical_or(abnormal_heart, abnormal_breath)

    return z_heart, z_breath, abnormal


def main():
    csv_path = os.path.join(os.getcwd(), CSV_PATH)
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    heart = df["HeartRate"].values
    breath = df["BreathingRate"].values
    time_idx = np.arange(len(heart))

    # --- 移動平均を「基準値」として計算 ---
    heart_ma = pd.Series(heart).rolling(WINDOW, center=False).mean().to_numpy()
    breath_ma = pd.Series(breath).rolling(WINDOW, center=False).mean().to_numpy()

    # 最初の WINDOW-1 点は移動平均が NaN なのでスキップ
    valid = ~np.isnan(heart_ma) & ~np.isnan(breath_ma)
    heart = heart[valid]
    breath = breath[valid]
    heart_ma = heart_ma[valid]
    breath_ma = breath_ma[valid]
    time_idx = time_idx[valid]

    # 誤差
    err_heart = heart - heart_ma
    err_breath = breath - breath_ma

    # 3σ用の平均・標準偏差
    mu_h, sd_h = err_heart.mean(), err_heart.std() + 1e-8
    mu_b, sd_b = err_breath.mean(), err_breath.std() + 1e-8

    z_heart = (err_heart - mu_h) / sd_h
    z_breath = (err_breath - mu_b) / sd_b

    abn_heart = np.abs(z_heart) > Z_TH
    abn_breath = np.abs(z_breath) > Z_TH
    abnormal = np.logical_or(abn_heart, abn_breath)

    print(f"3σ法で検出された異常点数: {abnormal.sum()} / {len(abnormal)}")

    # 心拍
    plt.figure(figsize=(12, 4))
    plt.plot(time_idx, heart, label="HeartRate")
    plt.plot(time_idx, heart_ma, label="Moving Avg (Heart)", alpha=0.7)
    plt.scatter(time_idx[abn_heart], heart[abn_heart],
                color="red", label="Anomaly (Heart 3σ)", zorder=5)
    plt.legend(); plt.tight_layout(); plt.show()

    # 呼吸
    plt.figure(figsize=(12, 4))
    plt.plot(time_idx, breath, label="BreathingRate")
    plt.plot(time_idx, breath_ma, label="Moving Avg (Breath)", alpha=0.7)
    plt.scatter(time_idx[abn_breath], breath[abn_breath],
                color="red", label="Anomaly (Breath 3σ)", zorder=5)
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()