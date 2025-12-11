import socket
import threading
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import numpy as np
import japanize_matplotlib
import csv
 
class C1001RealTimePlot:
    def __init__(self, host="172.20.10.2", port=7007, max_points=100):
        self.host = host
        self.port = port
        self.max_points = max_points
        self.server_running = True
       
        # CSVファイル名
        self.csv_filename = f"sensor_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()  # ← この中で open して保持するように変更

        # データバッファの初期化
        self.time_buffer = deque(maxlen=max_points)
        self.presence_data = deque([0] * max_points, maxlen=max_points)
        self.movement_data = deque([0] * max_points, maxlen=max_points)
        self.moving_range_data = deque([0] * max_points, maxlen=max_points)
        self.breathing_data = deque([0] * max_points, maxlen=max_points)
        self.heart_rate_data = deque([0] * max_points, maxlen=max_points)
        # self.fall_data = ...
        # self.dwell_data = ...  ← これらは削除
       
        # プロットの設定
        self.setup_plot()
 
    # ★追加: CSVファイルの初期化（ヘッダー書き込み）
    def init_csv(self):
        # ここで open して、writer を保持する
        self.csv_file = open(self.csv_filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Timestamp','Presence','Movement','MovingRange',
            'BreathingRate','HeartRate'
        ])
        print(f"CSV保存を開始しました: {self.csv_filename}")

    # ★追加: データをCSVに追記するメソッド
    def save_to_csv(self, data):
        # 毎回 open しないで、そのまま書き込む
        self.csv_writer.writerow([
            data['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f'), # ミリ秒まで記録
            data['presence'], data['movement'], data['moving_range'],
            data['breathing_rate'], data['heart_rate']
        ])
        # 必要なら即ディスク反映したい場合だけ
        # self.csv_file.flush()
   
    def setup_plot(self):
        """プロットの初期設定"""
        plt.rcParams['font.size'] = 9
        self.fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        ((self.ax1, self.ax2),
         (self.ax3, self.ax4),
         (self.ax5, self.ax6),
         (self.ax7, self.ax8)) = axes
       
        # プロット1: Presence (存在検知)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, marker='o', markersize=3)
        self.ax1.set_title('① Presence - 存在検知')
        self.ax1.set_ylabel('状態 (0:不在, 1:存在)')
        self.ax1.set_ylim(-0.1, 1.5)
        self.ax1.set_yticks([0, 1])
        self.ax1.set_yticklabels(['不在', '存在'])
        self.ax1.grid(True, alpha=0.3)
       
        # プロット2: Movement (活動状態)
        self.line2, = self.ax2.plot([], [], 'g-', linewidth=2, marker='s', markersize=3)
        self.ax2.set_title('② Movement - 活動状態')
        self.ax2.set_ylabel('状態 (0:静止, 1:微動, 2:活動)')
        self.ax2.set_ylim(-0.2, 2.2)
        self.ax2.set_yticks([0, 1, 2])
        self.ax2.set_yticklabels(['静止', '微動', '活動'])
        self.ax2.grid(True, alpha=0.3)
       
        # プロット3: Moving Range (活動範囲)
        self.line3, = self.ax3.plot([], [], 'r-', linewidth=2, marker='^', markersize=3)
        self.ax3.set_title('③ Moving Range - 活動範囲')
        self.ax3.set_ylabel('範囲値')
        self.ax3.set_ylim(0, 150)
        self.ax3.grid(True, alpha=0.3)
       
        # プロット4: Breathing Rate (呼吸数)
        self.line4, = self.ax4.plot([], [], 'purple', linewidth=2, marker='d', markersize=3)
        self.ax4.set_title('④ Breathing Rate - 呼吸数')
        self.ax4.set_ylabel('回/分')
        self.ax4.set_ylim(0, 35)
        self.ax4.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='下限 (10)')
        self.ax4.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='上限 (25)')
        self.ax4.legend(fontsize=8)
        self.ax4.grid(True, alpha=0.3)
       
        # プロット5: Heart Rate (心拍数)
        self.line5, = self.ax5.plot([], [], 'orange', linewidth=2, marker='*', markersize=3)
        self.ax5.set_title('⑤ Heart Rate - 心拍数')
        self.ax5.set_ylabel('BPM')
        self.ax5.set_ylim(40, 130)
        self.ax5.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='下限 (60)')
        self.ax5.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='上限 (100)')
        self.ax5.legend(fontsize=8)
        self.ax5.grid(True, alpha=0.3)

        # プロット8: 総合ビュー（正規化）
        self.line8a, = self.ax8.plot([], [], 'b-', label='Presence', linewidth=1.2)
        self.line8b, = self.ax8.plot([], [], 'g-', label='Movement', linewidth=1.2)
        self.line8c, = self.ax8.plot([], [], 'r-', label='MovingRange', linewidth=1.2)
        self.line8d, = self.ax8.plot([], [], 'purple', label='Breathing', linewidth=1.2)
        self.line8e, = self.ax8.plot([], [], 'orange', label='HeartRate', linewidth=1.2)
        self.ax8.set_title('⑧ 総合ビュー（正規化）')
        self.ax8.set_ylim(0, 1.1)
        self.ax8.legend(fontsize=8, ncol=3)
        self.ax8.grid(True, alpha=0.3)
        plt.tight_layout()
        self.fig.suptitle('C1001 mmWave センサー リアルタイムモニタリング', fontsize=14, fontweight='bold', y=1.02)
       
    def parse_sensor_data(self, data_line):
        try:
            parts = data_line.strip().split(',')
            if len(parts) >= 5:  # ★ 5 以上あればOK
                return {
                    'timestamp': datetime.datetime.now(),
                    'presence': int(parts[0]),
                    'movement': int(parts[1]),
                    'moving_range': int(parts[2]),
                    'breathing_rate': int(parts[3]),
                    'heart_rate': int(parts[4]),
                }
            else:
                print(f"データ形式エラー: 期待5要素以上, 実際{len(parts)}要素")
                return None
        except ValueError as e:
            print(f"データ変換エラー: {e}, データ: {data_line}")
            return None
        except Exception as e:
            print(f"データ解析エラー: {e}")
            return None
   
    def update_buffers(self, data):
        """データバッファの更新"""
        if data is None:
            return
           
        # ★追加: CSVに保存
        self.save_to_csv(data)

        # データバッファに追加
        self.presence_data.append(data['presence'])
        self.movement_data.append(data['movement'])
        self.moving_range_data.append(data['moving_range'])
        self.breathing_data.append(data['breathing_rate'])
        self.heart_rate_data.append(data['heart_rate'])
        # タイムスタンプの更新
        self.time_buffer.append(data['timestamp'])
   
    def update_plot(self, frame):
        """プロットの更新"""
        try:
            current_length = len(self.presence_data)
           
            if current_length == 0:
                return []
               
            # 時間軸の作成（最新のデータから）
            time_indices = list(range(max(0, self.max_points - current_length), self.max_points))
           
            # プロット1: Presence
            self.line1.set_data(time_indices, list(self.presence_data))
            self.ax1.set_xlim(time_indices[0], time_indices[-1])
           
            # プロット2: Movement
            self.line2.set_data(time_indices, list(self.movement_data))
            self.ax2.set_xlim(time_indices[0], time_indices[-1])
           
            # プロット3: Moving Range
            self.line3.set_data(time_indices, list(self.moving_range_data))
            self.ax3.set_xlim(time_indices[0], time_indices[-1])
           
            # プロット4: Breathing Rate
            self.line4.set_data(time_indices, list(self.breathing_data))
            self.ax4.set_xlim(time_indices[0], time_indices[-1])
           
            # プロット5: Heart Rate
            self.line5.set_data(time_indices, list(self.heart_rate_data))
            self.ax5.set_xlim(time_indices[0], time_indices[-1])

            # 総合ビュー（正規化）
            if current_length > 0:
                # 各データを0-1の範囲に正規化
                presence_norm = list(self.presence_data)  # 0-1なのでそのまま
                movement_norm = [x / 2.0 for x in self.movement_data]  # 0-2を0-1に
                moving_range_norm = [min(x / 100.0, 1.0) for x in self.moving_range_data]  # 0-100を0-1に
                breathing_norm = [min(x / 30.0, 1.0) for x in self.breathing_data]  # 0-30を0-1に
                heart_norm = [min(x / 120.0, 1.0) for x in self.heart_rate_data]  # 0-120を0-1に

                self.line8a.set_data(time_indices, presence_norm)
                self.line8b.set_data(time_indices, movement_norm)
                self.line8c.set_data(time_indices, moving_range_norm)
                self.line8d.set_data(time_indices, breathing_norm)
                self.line8e.set_data(time_indices, heart_norm)
                self.ax8.set_xlim(time_indices[0], time_indices[-1])
           
            # ステータス表示を更新
            latest_time = self.time_buffer[-1] if self.time_buffer else datetime.datetime.now()
            self.fig.suptitle(
                f'C1001 mmWave センサー リアルタイムモニタリング\n'
                f'最終更新: {latest_time.strftime("%H:%M:%S")} | '
                f'データ数: {current_length}/{self.max_points} | '
                f'Presence: {self.presence_data[-1]} | Movement: {self.movement_data[-1]} | '
                f'Breathing: {self.breathing_data[-1]} | Heart: {self.heart_rate_data[-1]}',
                fontsize=12
            )
           
            # 健康状態アラートの表示
            alerts = self.check_health_alerts()
            if alerts:
                alert_text = " | ".join(alerts)
                self.fig.text(0.02, 0.02, f"⚠️ {alert_text}",
                            fontsize=10, color='red', weight='bold',
                            transform=self.fig.transFigure)
           
        except Exception as e:
            print(f"プロット更新エラー: {e}")
           
        return [
            self.line1, self.line2, self.line3, self.line4, self.line5,
            self.line8a, self.line8b, self.line8c, self.line8d, self.line8e
        ]
   
    def check_health_alerts(self):
        """健康状態アラートのチェック"""
        if len(self.breathing_data) == 0:
            return []
           
        latest_breathing = self.breathing_data[-1]
        latest_heart = self.heart_rate_data[-1]
       
        alerts = []
       
        # 呼吸数チェック
        if latest_breathing < 10 and latest_breathing > 0:
            alerts.append(f"低呼吸: {latest_breathing}回/分")
        elif latest_breathing > 25:
            alerts.append(f"過呼吸: {latest_breathing}回/分")
        elif latest_breathing == 0 and self.presence_data[-1] == 1:
            alerts.append("呼吸検出なし")
           
        # 心拍数チェック
        if latest_heart < 60 and latest_heart > 0:
            alerts.append(f"低心拍: {latest_heart}BPM")
        elif latest_heart > 100:
            alerts.append(f"高心拍: {latest_heart}BPM")
        elif latest_heart == 0 and self.presence_data[-1] == 1:
            alerts.append("心拍検出なし")
           
        return alerts
   
    def wifi_server(self):
        """TCPサーバーでESP32からのデータを受信"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.settimeout(1.0)  # タイムアウト設定
            s.listen()
            print(f"サーバー待機中: {self.host}:{self.port}")
 
            while self.server_running:
                try:
                    conn, addr = s.accept()
                    with conn:
                        print(f"接続確立: {addr}")
                        conn.settimeout(1.0)
                        while self.server_running:
                            try:
                                data = conn.recv(1024)
                                if not data:
                                    print("クライアント切断")
                                    break
                               
                                decoded = data.decode("utf-8").strip()
                                # print(data)
                                print(datetime.datetime.now(),",",decoded)
                                # 複数行のデータを処理
                                lines = decoded.split('\n')
                                for line in lines:
                                    if line.strip():
                                        parsed_data = self.parse_sensor_data(line)
                                        self.update_buffers(parsed_data)
                                       
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
                except Exception as e:
                    if self.server_running:
                        print(f"サーバーエラー: {e}")
                    break
   
    def start_realtime_plot(self):
        """リアルタイムプロットの開始"""
        # サーバースレッドを開始
        server_thread = threading.Thread(target=self.wifi_server, daemon=True)
        server_thread.start()
       
        print("リアルタイムプロットを開始します...")
        print("Ctrl+Cで終了")
       
        # アニメーションの開始
        ani = FuncAnimation(
            self.fig, self.update_plot, interval=50, # ★ここを100から50に変更して描画更新を速くする
            blit=True, cache_frame_data=False
        )
       
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nプログラムを終了します")
        finally:
            self.server_running = False
            # ★ 追加：CSV を閉じる
            self.csv_file.close()
 
# メイン実行関数
def main():
    print("C1001 mmWave センサー リアルタイムプロット")
    print("=" * 50)
    print("監視データ:")
    print("  [0] Presence (存在検知): 0=不在, 1=存在")
    print("  [1] Movement (活動状態): 0=静止, 1=微動, 2=活動")
    print("  [2] Moving Range (活動範囲): 数値")
    print("  [3] Breathing Rate (呼吸数): 回/分 (正常: 10-25)")
    print("  [4] Heart Rate (心拍数): BPM (正常: 60-100)")
    print("=" * 50)
   
    # リアルタイムプロットのインスタンス作成
    plotter = C1001RealTimePlot(host="172.20.10.2", port=7007, max_points=100)# filepath: c:\Users\br211408\Documents\PlatformIO\Projects\wif_ver\src\receive.py
   
    # リアルタイムプロットを開始
    try:
        plotter.start_realtime_plot()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
 
if __name__ == "__main__":
    main()
