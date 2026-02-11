import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- フォント設定 ---
# japanize_matplotlib が効かない場合の代替策として、
# システムに入っている日本語フォントを直接指定する方法があります。
# Windows: 'MS Gothic', Mac: 'Hiragino Sans', Linux: 'Noto Sans CJK JP' など
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# データ定義
categories = ['睡眠時\n(正常)', '活動時\n(正常)', '無呼吸\n(異常)', '過呼吸\n(異常)', '激しい体動\n(異常)']
heart_rate_rmse = [0.334, 0.309, 0.323, 0.36, 0.57]
resp_rate_rmse = [0.076, 0.056, 0.056, 0.074, 0.137]

# グラフの作成（1行2列で配置）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. 心拍数のグラフ
ax1.bar(categories, heart_rate_rmse, color='#5B9BD5', width=0.6)
ax1.set_title('状態別：心拍数の誤差 (RMSE)', fontsize=14, fontweight='bold')
ax1.set_ylabel('RMSE')
ax1.set_ylim(0, 0.65)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# データラベル（心拍数）
for i, v in enumerate(heart_rate_rmse):
    ax1.text(i, v + 0.01, str(v), ha='center', fontsize=12)

# 2. 呼吸数のグラフ
ax2.bar(categories, resp_rate_rmse, color='#ED7D31', width=0.6)
ax2.set_title('状態別：呼吸数の誤差 (RMSE)', fontsize=14, fontweight='bold')
ax2.set_ylabel('RMSE')
ax2.set_ylim(0, 0.16)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# データラベル（呼吸数）
for i, v in enumerate(resp_rate_rmse):
    ax2.text(i, v + 0.002, str(v), ha='center', fontsize=12)

plt.tight_layout()
plt.show()