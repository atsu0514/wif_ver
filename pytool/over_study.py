import re
import matplotlib.pyplot as plt

# ログデータをここに貼り付けます
log_data = """
Epoch 1/50, TrainLoss=0.0028417839, ValLoss=0.0003635737
Epoch 2/50, TrainLoss=0.0004684038, ValLoss=0.0001873443
Epoch 3/50, TrainLoss=0.0003203447, ValLoss=0.0001224138
Epoch 4/50, TrainLoss=0.0002552547, ValLoss=0.0001046121
Epoch 5/50, TrainLoss=0.0002024493, ValLoss=0.0000729509
Epoch 6/50, TrainLoss=0.0001826702, ValLoss=0.0000820557
Epoch 7/50, TrainLoss=0.0001547136, ValLoss=0.0000669839
Epoch 8/50, TrainLoss=0.0001327408, ValLoss=0.0000640656
Epoch 9/50, TrainLoss=0.0001165293, ValLoss=0.0000401917
Epoch 10/50, TrainLoss=0.0001001211, ValLoss=0.0000358520
Epoch 11/50, TrainLoss=0.0000988138, ValLoss=0.0000466989
Epoch 12/50, TrainLoss=0.0000883649, ValLoss=0.0000291546
Epoch 13/50, TrainLoss=0.0000731647, ValLoss=0.0000443353
Epoch 14/50, TrainLoss=0.0000750468, ValLoss=0.0000234342
Epoch 15/50, TrainLoss=0.0000542230, ValLoss=0.0000304380
Epoch 16/50, TrainLoss=0.0000531139, ValLoss=0.0001597214
Epoch 17/50, TrainLoss=0.0000444653, ValLoss=0.0000148055
Epoch 18/50, TrainLoss=0.0000463387, ValLoss=0.0000143317
Epoch 19/50, TrainLoss=0.0000414151, ValLoss=0.0000112780
Epoch 20/50, TrainLoss=0.0000299450, ValLoss=0.0000102986
Epoch 21/50, TrainLoss=0.0000307062, ValLoss=0.0000157462
Epoch 22/50, TrainLoss=0.0000266147, ValLoss=0.0000075871
Epoch 23/50, TrainLoss=0.0000192124, ValLoss=0.0000139798
Epoch 24/50, TrainLoss=0.0000214628, ValLoss=0.0000306102
Epoch 25/50, TrainLoss=0.0000238741, ValLoss=0.0000332158
Epoch 26/50, TrainLoss=0.0000184323, ValLoss=0.0000123210
Epoch 27/50, TrainLoss=0.0000169560, ValLoss=0.0000105539
Epoch 28/50, TrainLoss=0.0000182217, ValLoss=0.0000065137
Epoch 29/50, TrainLoss=0.0000191021, ValLoss=0.0000040156
Epoch 30/50, TrainLoss=0.0000146212, ValLoss=0.0000041283
Epoch 31/50, TrainLoss=0.0000166795, ValLoss=0.0000050089
Epoch 32/50, TrainLoss=0.0000111007, ValLoss=0.0000043586
Epoch 33/50, TrainLoss=0.0000163515, ValLoss=0.0000035290
Epoch 34/50, TrainLoss=0.0000152116, ValLoss=0.0000252076
Epoch 35/50, TrainLoss=0.0000094961, ValLoss=0.0000020997
Epoch 36/50, TrainLoss=0.0000102034, ValLoss=0.0000032318
Epoch 37/50, TrainLoss=0.0000110026, ValLoss=0.0000384030
Epoch 38/50, TrainLoss=0.0000115099, ValLoss=0.0000039939
Epoch 39/50, TrainLoss=0.0000135641, ValLoss=0.0000177397
Epoch 40/50, TrainLoss=0.0000095712, ValLoss=0.0000023098
Epoch 41/50, TrainLoss=0.0000070314, ValLoss=0.0000126235
Epoch 42/50, TrainLoss=0.0000078309, ValLoss=0.0000033304
Epoch 43/50, TrainLoss=0.0000073895, ValLoss=0.0000010994
Epoch 44/50, TrainLoss=0.0000133498, ValLoss=0.0000025764
Epoch 45/50, TrainLoss=0.0000087025, ValLoss=0.0000035515
Epoch 46/50, TrainLoss=0.0000088821, ValLoss=0.0000020871
Epoch 47/50, TrainLoss=0.0000050255, ValLoss=0.0000136776
Epoch 48/50, TrainLoss=0.0000048088, ValLoss=0.0000040207
Epoch 49/50, TrainLoss=0.0000049623, ValLoss=0.0000055352
Epoch 50/50, TrainLoss=0.0000048163, ValLoss=0.0000017800
"""

def main():
    epochs = []
    train_losses = []
    val_losses = []

    # ログデータの解析
    for line in log_data.strip().split('\n'):
        # 例: Epoch 1/50, TrainLoss=0.0028, ValLoss=0.0004
        match = re.search(r'Epoch (\d+)/(\d+), TrainLoss=([0-9\.]+), ValLoss=([0-9\.]+)', line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(3))
            val_loss = float(match.group(4))
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    if not epochs:
        print("データが見つかりませんでした。ログ形式を確認してください。")
        return

    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='.', linestyle='-')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='.', linestyle='-')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # 縦軸を対数表示に切り替えたい場合はコメントアウトを外してください
    plt.yscale('log')

    print("グラフを表示します...")
    plt.show()

if __name__ == "__main__":
    main()