import os
import socket
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from collections import deque
import csv  # 追加

# --- 設定 ---
WINDOW_SIZE = 15
HIDDEN_SIZE = 32
NUM_LAYERS = 1

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # プロジェクトルート
MODELS_DIR = os.path.join(BASE_DIR, "models")
LATEST_PATH = os.path.join(MODELS_DIR, "latest.txt")

# ログCSVファイル（実行時刻付き）
LOG_CSV_PATH = os.path.join(
    BASE_DIR,
    f"sensor_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

def _resolve_artifact_dir():
    if not os.path.exists(LATEST_PATH):
        raise FileNotFoundError(f"latest.txt が見つかりません: {LATEST_PATH}")
    with open(LATEST_PATH, "r") as f:
        run_id = f.read().strip()
    artifact_dir = os.path.join(MODELS_DIR, run_id)
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"artifact ディレクトリがありません: {artifact_dir}")
    return artifact_dir

ARTIFACT_DIR = _resolve_artifact_dir()
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lstm_model.pth")
SCALER_X_PATH = os.path.join(ARTIFACT_DIR, "scaler_x.pkl")
SCALER_Y_PATH = os.path.join(ARTIFACT_DIR, "scaler_y.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.txt")

# 擬似リアルタイム用のテストデータ（オフライン検証に使う）
TEST_CSV = "sensor_data_20251203_232723.csv"

# センサーからのリアルタイム受信用（receive.py と同じ設定にあわせる）
HOST = "172.20.10.2"
PORT = 7007


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class RealtimePredictor:
    def __init__(self):
        # モデルとスケーラーのロード
        self.scaler_x = joblib.load(SCALER_X_PATH)
        self.scaler_y = joblib.load(SCALER_Y_PATH)

        self.model = LSTMModel(input_size=5,
                               hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYERS,
                               output_size=2)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

        with open(THRESHOLD_PATH, "r") as f:
            self.threshold = float(f.read())

        print(f"Loaded model and threshold: {self.threshold:.4f}")

        # データバッファ（直近 window_size 分のデータを保持）
        self.buffer = deque(maxlen=WINDOW_SIZE)

        # 予測結果保持用（次の時刻の正解と比較するため）
        self.last_prediction = None  # [HeartRate, BreathingRate] (scaled)

    def process_new_data(self, data_row):
        """
        data_row: [Presence, Movement, MovingRange, HeartRate, BreathingRate]
                  のリストまたは配列
        """
        # 1. バッファに追加
        self.buffer.append(data_row)

        # バッファが溜まるまでは予測できない
        if len(self.buffer) < WINDOW_SIZE:
            return None

        # 2. 入力データの作成とスケーリング
        input_data = np.array(self.buffer)  # [window, 5]
        input_scaled = self.scaler_x.transform(input_data)

        # Tensor化 [1, window, 5]
        input_tensor = torch.tensor(input_scaled,
                                    dtype=torch.float32).unsqueeze(0)

        # 3. 予測実行
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).numpy()[0]  # [2]

        # スケールを元に戻す（人間が読む用）
        pred_original = self.scaler_y.inverse_transform([pred_scaled])[0]

        # 4. 異常検知（前回の予測と、今回の実測値を比較）
        anomaly_score = 0.0
        is_anomaly = False

        if self.last_prediction is not None:
            # 今回の実測値（HeartRate, BreathingRate）をスケーリング
            current_target = np.array(data_row[-2:]).reshape(1, -1)
            current_target_scaled = self.scaler_y.transform(current_target)[0]

            # MSE計算 (前回の予測 vs 今回の実測)
            mse = np.mean((self.last_prediction - current_target_scaled) ** 2)
            anomaly_score = mse

            if anomaly_score > self.threshold:
                is_anomaly = True

        # 次回比較用に今回の予測を保存
        self.last_prediction = pred_scaled

        return {
            "pred_heart": float(pred_original[0]),
            "pred_breath": float(pred_original[1]),
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
        }


# parse_sensor_line を 5項目用に変更
def parse_sensor_line(data_line):
    """
    presence, movement, moving_range, breathing_rate, heart_rate
    の 5要素カンマ区切り文字列をパースする
    """
    try:
        parts = data_line.strip().split(',')
        if len(parts) < 5:
            print(f"データ形式エラー: 期待5要素, 実際{len(parts)}要素 -> {data_line}")
            return None

        return {
            "timestamp": datetime.datetime.now(),
            "presence": int(parts[0]),
            "movement": int(parts[1]),
            "moving_range": int(parts[2]),
            "breathing_rate": int(parts[3]),
            "heart_rate": int(parts[4]),
        }
    except Exception as e:
        print(f"parse_sensor_line エラー: {e}, data={data_line}")
        return None


# ====== モード1: ソケットからのリアルタイム推論 ======
def run_realtime_from_socket():
    predictor = RealtimePredictor()

    # CSVログ初期化
    log_file = open(LOG_CSV_PATH, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    # CSV ヘッダ（FallState, DwellState を削除）
    log_writer.writerow([
        "Timestamp",
        "Presence", "Movement", "MovingRange",
        "BreathingRate", "HeartRate",
        "PredHeart", "PredBreath",
        "AnomalyScore", "IsAnomaly",
    ])

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.settimeout(1.0)
            s.listen()
            print(f"LSTMリアルタイムサーバ待機中: {HOST}:{PORT}")

            while True:
                try:
                    conn, addr = s.accept()
                    with conn:
                        print(f"接続確立: {addr}")
                        conn.settimeout(1.0)
                        while True:
                            try:
                                data = conn.recv(1024)
                                if not data:
                                    print("クライアント切断")
                                    break

                                decoded = data.decode("utf-8").strip()
                                lines = decoded.split('\n')
                                for line in lines:
                                    if not line.strip():
                                        continue

                                    parsed = parse_sensor_line(line)
                                    if parsed is None:
                                        continue

                                    # LSTM学習時と同じ特徴量の順番に並べる
                                    features = [
                                        parsed["presence"],
                                        parsed["movement"],
                                        parsed["moving_range"],
                                        parsed["heart_rate"],
                                        parsed["breathing_rate"],
                                    ]

                                    result = predictor.process_new_data(features)
                                    if result is None:
                                        continue

                                    ts_obj = parsed["timestamp"]
                                    ts = ts_obj.strftime("%H:%M:%S")
                                    ts_full = ts_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
                                    actual_h = parsed["heart_rate"]
                                    actual_b = parsed["breathing_rate"]
                                    pred_h = result["pred_heart"]
                                    pred_b = result["pred_breath"]
                                    score = result["anomaly_score"]
                                    is_anom = result["is_anomaly"]
                                    alert = "<<< ANOMALY >>>" if is_anom else ""

                                    # コンソール出力
                                    print(
                                        f"{ts} | H(act, pred)=({actual_h:3d}, {pred_h:5.1f}) "
                                        f"B(act, pred)=({actual_b:3d}, {pred_b:5.1f}) "
                                        f"| score={score:.5f} {alert}"
                                    )

                                    # CSV 1行書き込み（fall/dwell を削除）
                                    log_writer.writerow([
                                        ts_full,
                                        parsed["presence"],
                                        parsed["movement"],
                                        parsed["moving_range"],
                                        parsed["breathing_rate"],
                                        parsed["heart_rate"],
                                        f"{pred_h:.3f}",
                                        f"{pred_b:.3f}",
                                        f"{score:.6f}",
                                        int(is_anom),
                                    ])
                                    log_file.flush()

                            except socket.timeout:
                                continue
                            except ConnectionResetError:
                                print("クライアント接続リセット")
                                break
                            except Exception as e:
                                print(f"データ受信エラー: {e}")
                                break

                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print("\nCtrl+C を受信したため終了します")
                    break
                except Exception as e:
                    print(f"サーバーエラー: {e}")
                    break
    finally:
        log_file.close()
        print(f"ログCSVを閉じました: {LOG_CSV_PATH}")


# ====== モード2: CSVからの擬似リアルタイム（オフライン検証用） ======
def main_simulation():
    predictor = RealtimePredictor()

    print(f"Reading {TEST_CSV} for simulation...")
    df = pd.read_csv(TEST_CSV)

    feature_cols = ["Presence", "Movement", "MovingRange", "HeartRate", "BreathingRate"]
    data_stream = df[feature_cols].values

    print("Start Real-time Simulation...")
    print("TimeIdx | Actual(H, B) | Pred(H, B) | AnomalyScore | Alert")
    print("-" * 70)

    for i, row in enumerate(data_stream):
        result = predictor.process_new_data(row)

        if result:
            actual_h, actual_b = row[3], row[4]
            pred_h, pred_b = result["pred_heart"], result["pred_breath"]
            score = result["anomaly_score"]
            alert = "!!! ANOMALY !!!" if result["is_anomaly"] else ""

            print(
                f"{i:04d} | {actual_h:5.1f}, {actual_b:5.1f} | "
                f"{pred_h:5.1f}, {pred_b:5.1f} | {score:.4f} | {alert}"
            )


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Run train_ltsm.py first.")
    else:
        # 実センサーからのリアルタイム推論
        run_realtime_from_socket()
        # オフライン検証したい場合はこちらを呼ぶ:
        # main_simulation()