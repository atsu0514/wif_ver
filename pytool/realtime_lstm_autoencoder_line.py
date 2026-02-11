import os
import socket
import datetime
import time
import threading
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import joblib
from collections import deque
import csv
from flask import Flask, request, abort

# 警告を無視する設定
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
    # SDK v3の警告特定クラスがあれば無視設定に追加
    try:
        from linebot.deprecations import LineBotSdkDeprecatedIn30
        warnings.filterwarnings("ignore", category=LineBotSdkDeprecatedIn30)
    except ImportError:
        pass
except ImportError:
    print("Error: line-bot-sdk is not installed.")

# ==========================================
# 1. 設定・環境変数の読み込み
# ==========================================
WINDOW_SIZE = 15
HIDDEN_SIZE = 16
NUM_LAYERS = 2
NOTIFY_COOLDOWN = 10 

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "autoencoder_models")
LATEST_PATH = os.path.join(MODELS_DIR, "latest.txt")
LOG_CSV_PATH = os.path.join(BASE_DIR, f"sensor_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


# .env読み込み
ENV_PATH = os.path.join(BASE_DIR, "linetool", ".env")

if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    # コメント除去などを簡易的に行う
                    val = val.split("#")[0].strip()
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Warning: Failed to load .env file: {e}")

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

latest_status = {
    "timestamp": None,
    "heart_rate": 0,
    "breathing_rate": 0,
    "pred_heart": 0.0,
    "pred_breath": 0.0,
    "anomaly_score": 0.0,
    "is_anomaly": False,
    "movement": 0
}
status_lock = threading.Lock()

# ==========================================
# 2. LINE Bot / Flask サーバー設定
# ==========================================
app = Flask(__name__)
# Flaskのアクセスログを抑制（コンソールを見やすくするため）
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

line_bot_api = None
handler = None

if CHANNEL_SECRET and CHANNEL_ACCESS_TOKEN:
    try:
        line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
        handler = WebhookHandler(CHANNEL_SECRET)
    except Exception as e:
        print(f"LINE Bot Init Error: {e}")
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

if handler:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_text_message(event: MessageEvent):
        # 受信ログを表示（デバッグ用）
        print(f"\n[LINE Recv] {event.message.text}")
        
        text = event.message.text.strip()
        reply = None

        if text == "状態" or text == "ステータス":
            with status_lock:
                current_status = latest_status.copy()

            if current_status["timestamp"] is None:
                reply = "データ受信待機中です..."
            else:
                ts_str = current_status["timestamp"].strftime("%H:%M:%S")
                status_label = "⚠️異常検知中" if current_status["is_anomaly"] else "✅正常"
                
                # フォーマット統一: 手動確認時
                reply = (
                    f"【ステータス確認】\n"
                    f"時刻: {ts_str}\n"
                    f"判定: {status_label}\n"
                    f"スコア: {current_status['anomaly_score']:.4f}\n"
                    f"心拍: {current_status['heart_rate']} (予測: {current_status['pred_heart']:.1f})\n"
                    f"呼吸: {current_status['breathing_rate']} (予測: {current_status['pred_breath']:.1f})"
                )
        elif text == "ヘルプ":
            reply = "「状態」と送ると最新情報を返します。"
        
        if reply:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            print(f"[LINE Sent] {reply.splitlines()[0]}...")

def run_flask_server():
    port = int(os.getenv("PORT", 8000))
    print(f"--- LINE Webhook Server running on port {port} ---")
    print(f" * Check local status: http://localhost:{port}/")
    print(f" * Note: Use ngrok to expose port {port} for LINE Webhook.")
    app.run(host="0.0.0.0", port=port, use_reloader=False)

def send_push_notification(message):
    if not line_bot_api or not LINE_USER_ID:
        return
    try:
        line_bot_api.push_message(LINE_USER_ID, TextSendMessage(text=message))
        print(">> PUSH通知送信完了")
    except Exception as e:
        print(f">> PUSH送信エラー: {e}")

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
    MODEL_PATH = os.path.join(ARTIFACT_DIR, "lstm_autoencoder_model.pth")
    SCALER_X_PATH = os.path.join(ARTIFACT_DIR, "scaler_x.pkl")
    SCALER_Y_PATH = os.path.join(ARTIFACT_DIR, "scaler_y.pkl")
    THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.txt")
except Exception as e:
    print(f"モデルパス解決エラー: {e}")
    exit()

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        b, t, _ = x.shape
        dec_in = torch.zeros((b, t, self.out.out_features), device=x.device, dtype=x.dtype)
        dec_out, _ = self.decoder(dec_in, (h, c))
        recon = self.out(dec_out)
        return recon


class RealtimePredictor:
    def __init__(self, threshold_type="high"):
        self.scaler_x = joblib.load(SCALER_X_PATH)
        self.scaler_y = joblib.load(SCALER_Y_PATH)
        self.model = LSTMEncoderDecoder(3, HIDDEN_SIZE, NUM_LAYERS, 2)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()
        # --- 複数閾値対応 ---
        try:
            with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            vals = [float(x) for x in lines if x.strip()]
            if not vals:
                print("Warning: Threshold file empty, using default 0.1")
                vals = [0.1, 0.1, 0.1]
        except Exception as e:
            print(f"Warning: Failed to load threshold: {e}. Using default 0.1")
            vals = [0.1, 0.1, 0.1]
        # 閾値タイプ選択
        self.thresholds = {
            "low": vals[0] if len(vals) > 0 else 0.1,
            "mid": vals[1] if len(vals) > 1 else vals[-1],
            "high": vals[2] if len(vals) > 2 else vals[-1]
        }
        self.threshold_type = threshold_type
        self.threshold = self.thresholds.get(threshold_type, vals[-1])
        print(f"Loaded model. Using Threshold ({threshold_type}): {self.threshold:.8f}")
        self.buffer = deque(maxlen=WINDOW_SIZE)

    def process_new_data(self, feature_row):
        self.buffer.append(feature_row)
        if len(self.buffer) < WINDOW_SIZE: return None
        
        x_raw = np.array(self.buffer)
        x_scaled = self.scaler_x.transform(x_raw)
        
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred_scaled = self.model(x_tensor).numpy()[0]
        
        y_true_raw = x_raw[:, 1:3]
        y_true_scaled = self.scaler_y.transform(y_true_raw)
        
        # MSE計算
        mse_score = np.mean((y_pred_scaled - y_true_scaled) ** 2)
        is_anomaly = mse_score > self.threshold
        
        y_pred_last_scaled = y_pred_scaled[-1].reshape(1, -1)
        y_pred_last_raw = self.scaler_y.inverse_transform(y_pred_last_scaled)[0]
        
        return {
            "pred_heart": float(y_pred_last_raw[0]),
            "pred_breath": float(y_pred_last_raw[1]),
            "anomaly_score": float(mse_score),
            "is_anomaly": bool(is_anomaly),
        }

def parse_sensor_line(data_line):
    try:
        parts = data_line.strip().split(',')
        if len(parts) < 5: return None
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

    # --- 閾値設定 ---
    predictor = RealtimePredictor(threshold_type="high") 

    log_file = open(LOG_CSV_PATH, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "Timestamp","Presence","Movement","MovingRange",
        "BreathingRate","HeartRate",
        "PredHeart","PredBreath","AnomalyScore","IsAnomaly"
    ])

    # 通知クールダウン用タイマー初期化
    last_notify_time = 0 

    print(f"Waiting for sensor on {HOST}:{PORT}...")
    
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
                                    
                                    features = [parsed["moving_range"], parsed["heart_rate"], parsed["breathing_rate"]]
                                    result = predictor.process_new_data(features)
                                    if not result: continue
                                    
                                    ts = parsed["timestamp"]
                                    score = result["anomaly_score"]
                                    is_anom = result["is_anomaly"]
                                    
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
                                    
                                    with status_lock:
                                        latest_status = new_status
                                    
                                    alert = "<<< ANOMALY >>>" if is_anom else ""
                                    print(
                                        f"{ts.strftime('%H:%M:%S')} | 心拍={parsed['heart_rate']} 呼吸={parsed['breathing_rate']} "
                                        f"(予測: 呼吸={result['pred_breath']:.3f}, 心拍={result['pred_heart']:.3f}) {alert}"
                                    )
                                    
                                    if is_anom:
                                        now = time.time()
                                        if now - last_notify_time > NOTIFY_COOLDOWN:
                                            # フォーマット統一: 自動通知時
                                            msg = (
                                                f"⚠️ 異常検知アラート ⚠️\n"
                                                f"時刻: {ts.strftime('%H:%M:%S')}\n"
                                                f"判定: ⚠️異常検知中\n"
                                                f"スコア: {score:.4f}\n"
                                                f"心拍: {parsed['heart_rate']} (予測: {result['pred_heart']:.1f})\n"
                                                f"呼吸: {parsed['breathing_rate']} (予測: {result['pred_breath']:.1f})"
                                            )
                                            
                                            send_push_notification(msg)
                                            last_notify_time = now
                                    
                                    log_writer.writerow([
                                        ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                        parsed["presence"], parsed["movement"], parsed["moving_range"],
                                        parsed["breathing_rate"], parsed["heart_rate"],
                                        f"{result['pred_heart']:.3f}", f"{result['pred_breath']:.3f}",
                                        f"{score:.8f}", int(is_anom)
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
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    
    time.sleep(1)
    run_realtime_system()