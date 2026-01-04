import os
import socket
import datetime
import time
import threading
import logging
import numpy as np
import torch
import torch.nn as nn
import joblib
from collections import deque
import csv
from flask import Flask, request, abort
from dotenv import load_dotenv
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ==========================================
# 1. 設定・環境変数の読み込み
# ==========================================
WINDOW_SIZE = 15
HIDDEN_SIZE = 32
NUM_LAYERS = 1
NOTIFY_COOLDOWN = 10  # 異常通知の抑制時間(秒)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LATEST_PATH = os.path.join(MODELS_DIR, "latest.txt")
LOG_CSV_PATH = os.path.join(BASE_DIR, f"sensor_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# .env読み込み (linetoolフォルダにあると仮定)
ENV_PATH = os.path.join(BASE_DIR, "linetool", ".env")
load_dotenv(ENV_PATH)

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")  # プッシュ通知先

# 共有変数（Botが最新状態を返信するために使用）
latest_status = {
    "timestamp": None,
    "heart_rate": 0,
    "breathing_rate": 0,
    "pred_heart": 0.0,
    "pred_breath": 0.0,
    "anomaly_score": 0.0,
    "is_anomaly": False
}
status_lock = threading.Lock()  # ★これがあるか確認

# ==========================================
# 2. LINE Bot / Flask サーバー設定
# ==========================================
app = Flask(__name__)
# ログを少し静かにする
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

line_bot_api = None
handler = None

if CHANNEL_SECRET and CHANNEL_ACCESS_TOKEN:
    line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(CHANNEL_SECRET)
else:
    print("警告: LINE設定が不足しています。Bot機能は動作しません。")

@app.route("/", methods=["GET"])
def index():
    return "Realtime LSTM Bot is running.", 200

@app.route("/callback", methods=["POST"])
def callback():
    if not handler: return "Config Error", 500
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    text = event.message.text.strip()
    
    if text == "状態" or text == "ステータス":
        # ★追加: ロックを取得して安全にデータをコピー
        with status_lock:
            current_status = latest_status.copy()

        if current_status["timestamp"] is None:
            reply = "まだデータを受信していません。"
        else:
            ts_str = current_status["timestamp"].strftime("%H:%M:%S")
            anom_str = "⚠️異常検知中" if current_status["is_anomaly"] else "✅正常"
            reply = (
                f"【現在時刻: {ts_str}】\n"
                f"状態: {anom_str}\n"
                f"心拍数: {current_status['heart_rate']} (予測 {current_status['pred_heart']:.1f})\n"
                f"呼吸数: {current_status['breathing_rate']} (予測 {current_status['pred_breath']:.1f})\n"
                f"スコア: {current_status['anomaly_score']:.4f}"
                f"活動状態: {current_status['movement']}"
            )
    elif text == "ヘルプ":
        reply = "「状態」と送ると最新のセンサー値を返します。"
    else:
        reply = f"「{text}」ですね。最新情報を知りたい場合は「状態」と送ってください。"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

def run_flask_server():
    port = int(os.getenv("PORT", 8000))
    print(f"LINE Webhook Server starting on port {port}...")
    app.run(host="0.0.0.0", port=port, use_reloader=False)

def send_push_notification(message):
    """異常検知時にプッシュ通知を送る"""
    if not line_bot_api or not LINE_USER_ID:
        return
    try:
        line_bot_api.push_message(LINE_USER_ID, TextSendMessage(text=message))
        print(">> LINE通知送信完了")
    except Exception as e:
        print(f">> LINE送信エラー: {e}")

# ==========================================
# 3. LSTM 推論ロジック
# ==========================================
def _resolve_artifact_dir():
    if not os.path.exists(LATEST_PATH):
        raise FileNotFoundError(f"latest.txt が見つかりません: {LATEST_PATH}")
    with open(LATEST_PATH, "r") as f:
        run_id = f.read().strip()
    artifact_dir = os.path.join(MODELS_DIR, run_id)
    return artifact_dir

try:
    ARTIFACT_DIR = _resolve_artifact_dir()
    MODEL_PATH = os.path.join(ARTIFACT_DIR, "lstm_model.pth")
    SCALER_X_PATH = os.path.join(ARTIFACT_DIR, "scaler_x.pkl")
    SCALER_Y_PATH = os.path.join(ARTIFACT_DIR, "scaler_y.pkl")
    THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.txt")
except Exception as e:
    print(f"モデルパス解決エラー: {e}")
    exit()

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class RealtimePredictor:
    def __init__(self):
        self.scaler_x = joblib.load(SCALER_X_PATH)
        self.scaler_y = joblib.load(SCALER_Y_PATH)
        # ★ input_size=3 になっているか確認
        self.model = LSTMModel(3, HIDDEN_SIZE, NUM_LAYERS, 2)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()
        with open(THRESHOLD_PATH, "r") as f:
            self.threshold = float(f.read())
        print(f"Loaded model. Threshold: {self.threshold:.4f}")
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.last_prediction = None

    def process_new_data(self, data_row):
        self.buffer.append(data_row)
        if len(self.buffer) < WINDOW_SIZE: return None
        
        input_data = np.array(self.buffer)
        input_scaled = self.scaler_x.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).numpy()[0]
        
        pred_original = self.scaler_y.inverse_transform([pred_scaled])[0]
        
        anomaly_score = 0.0
        is_anomaly = False
        if self.last_prediction is not None:
            current_target = np.array(data_row[-2:]).reshape(1, -1)
            current_target_scaled = self.scaler_y.transform(current_target)[0]
            mse = np.mean((self.last_prediction - current_target_scaled) ** 2)
            anomaly_score = mse
            if anomaly_score > self.threshold:
                is_anomaly = True
        
        self.last_prediction = pred_scaled
        return {
            "pred_heart": float(pred_original[0]),
            "pred_breath": float(pred_original[1]),
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
        }

def parse_sensor_line(data_line):
    try:
        parts = data_line.strip().split(',')
        if len(parts) < 5:
            return None
        return {
            "timestamp": datetime.datetime.now(),
            "presence": int(parts[0]),
            "movement": int(parts[1]),
            "moving_range": int(parts[2]),
            "breathing_rate": int(parts[3]),
            "heart_rate": int(parts[4]),
        }
    except:
        return None

# ==========================================
# 4. メイン処理 (ソケット受信ループ)
# ==========================================
def run_realtime_system():
    HOST = "172.20.10.2"
    PORT = 7007
    
    predictor = RealtimePredictor()
    last_notify_time = 0
    
    # CSVオープン
    log_file = open(LOG_CSV_PATH, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "Timestamp","Presence","Movement","MovingRange",
        "BreathingRate","HeartRate",
        "PredHeart","PredBreath","AnomalyScore","IsAnomaly"
    ])

    print(f"Waiting for connection on {HOST}:{PORT}...")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.settimeout(1.0)
            s.listen()
            
            while True:
                try:
                    conn, addr = s.accept()
                    with conn:
                        print(f"Connected: {addr}")
                        conn.settimeout(1.0)
                        while True:
                            try:
                                data = conn.recv(1024)
                                if not data: break
                                
                                lines = data.decode("utf-8").strip().split('\n')
                                for line in lines:
                                    if not line.strip(): continue
                                    parsed = parse_sensor_line(line)
                                    if not parsed: continue
                                    
                                    # 推論
                                    # ★ features が3要素になっているか確認
                                    features = [parsed["moving_range"], parsed["heart_rate"], parsed["breathing_rate"]]
                                    result = predictor.process_new_data(features)
                                    if not result: continue
                                    
                                    # 結果の展開
                                    ts = parsed["timestamp"]
                                    score = result["anomaly_score"]
                                    is_anom = result["is_anomaly"]
                                    
                                    # 共有変数の更新（Bot用）
                                    global latest_status
                                    new_status = {
                                        "timestamp": ts,
                                        "heart_rate": parsed["heart_rate"],
                                        "breathing_rate": parsed["breathing_rate"],
                                        "movement": parsed["movement"],
                                        "pred_heart": result["pred_heart"],
                                        "pred_breath": result["pred_breath"],
                                        "anomaly_score": score,
                                        "is_anomaly": is_anom
                                    }
                                    
                                    # ★追加: ロックを取得して更新
                                    with status_lock:
                                        latest_status = new_status
                                    
                                    # コンソール表示
                                    alert = "<<< ANOMALY >>>" if is_anom else ""
                                    print(f"{ts.strftime('%H:%M:%S')} | Score={score:.4f} {alert}")
                                    
                                    # ★ 異常時のLINEプッシュ通知
                                    if is_anom:
                                        now = time.time()
                                        if now - last_notify_time > NOTIFY_COOLDOWN:
                                            msg = (f"⚠️ 異常検知 ⚠️\n"
                                                   f"時刻: {ts.strftime('%H:%M:%S')}\n"
                                                   f"スコア: {score:.4f}\n"
                                                   f"心拍: {parsed['heart_rate']} (予測 {result['pred_heart']:.1f})"
                                                   f"活動状態: {parsed['movement']}")
                                            send_push_notification(msg)
                                            last_notify_time = now
                                    
                                    # CSV保存
                                    log_writer.writerow([
                                        ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                        parsed["presence"], parsed["movement"], parsed["moving_range"],
                                        parsed["breathing_rate"], parsed["heart_rate"],
                                        f"{result['pred_heart']:.3f}", f"{result['pred_breath']:.3f}",
                                        f"{score:.6f}", int(is_anom)
                                    ])
                                    log_file.flush()
                                    
                            except socket.timeout: continue
                            except Exception as e: 
                                print(f"Data Error: {e}")
                                break
                except socket.timeout: continue
                except KeyboardInterrupt: break
                except Exception as e: print(f"Server Error: {e}")
    finally:
        log_file.close()
        print("System Shutdown.")

if __name__ == "__main__":
    # 1. Flaskサーバーを別スレッドで起動（LINE Webhook用）
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    
    # 2. メインスレッドでソケット受信＆推論ループを実行
    run_realtime_system()