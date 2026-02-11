import matplotlib.pyplot as plt
import numpy as np

def plot_rmse_comparison():
    # --- 日本語フォント設定 (macOS用) ---
    plt.rcParams['font.family'] = 'Hiragino Sans' # 必要に応じて 'Meiryo' 等に変更

    # --- データ定義 ---
    labels = ["睡眠時\n(正常)", "活動時\n(正常)", "無呼吸\n(異常)", "過呼吸\n(異常)", "激しい体動\n(異常)"]
    
    # 画像の数値データ
    rmse_heart = [0.334, 0.309, 0.323, 0.36, 0.57]
    rmse_breath = [0.076, 0.056, 0.056, 0.074, 0.137]

    # --- 共通のY軸最大値を決定 ---
    # 両方のデータの最大値を取得し、少し余裕を持たせる
    max_val = max(max(rmse_heart), max(rmse_breath))
    y_limit = max_val + 0.08  # 0.65程度になるように調整

    # --- グラフ描画設定 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    color_heart = '#5b9bd5'
    color_breath = '#ed7d31'

    # 1. 左側のグラフ：心拍数の誤差
    ax1 = axes[0]
    bars1 = ax1.bar(labels, rmse_heart, color=color_heart, width=0.6)
    
    ax1.set_title("状態別：心拍数の誤差 (RMSE)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_ylim(0, y_limit)  # 共通のY軸制限
    ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax1.set_axisbelow(True)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.005, 
            f'{height}', 
            ha='center', va='bottom', fontsize=14
        )

    # 2. 右側のグラフ：呼吸数の誤差
    ax2 = axes[1]
    bars2 = ax2.bar(labels, rmse_breath, color=color_breath, width=0.6)
    
    ax2.set_title("状態別：呼吸数の誤差 (RMSE)", fontsize=16, fontweight='bold')
    ax2.set_ylabel("RMSE", fontsize=12)
    ax2.set_ylim(0, y_limit)  # 共通のY軸制限
    ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax2.set_axisbelow(True)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.005, 
            f'{height}', 
            ha='center', va='bottom', fontsize=14
        )

    # --- レイアウト調整と表示 ---
    plt.tight_layout()
    print("グラフを表示します...")
    plt.show()

if __name__ == "__main__":
    plot_rmse_comparison()