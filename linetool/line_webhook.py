import os
import logging
from flask import Flask, request, abort
from dotenv import load_dotenv
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# このファイルと同じ場所の .env を確実に読み込む
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    print("エラー: 環境変数 LINE_CHANNEL_SECRET と LINE_CHANNEL_ACCESS_TOKEN を設定してください。")
    exit()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ブラウザ確認用（ルート）
@app.route("/", methods=["GET"])
def index():
    return "LINE Bot webhook is running. Health: /health, Webhook: POST /callback", 200

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body[:500]}")  # 長すぎるログを抑制
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.warning("署名が不正です。Secret/TokenやWebhook URLを確認してください。")
        abort(400)
    except Exception as e:
        app.logger.exception(f"エラーが発生しました: {e}")
        abort(500)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    user_text = (event.message.text or "").strip()
    if user_text == "こんにちは":
        reply_text = "こんにちは！"
    elif user_text == "天気":
        reply_text = "今日の天気は晴れです。（これはサンプルです）"
    elif "ありがとう" in user_text:
        reply_text = "どういたしまして！"
    else:
        reply_text = f"「{user_text}」ですね。"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

#ポート転送でHTTPと公開に設定するとできます。