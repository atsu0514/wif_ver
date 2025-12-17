import os
import pandas as pd
import glob

# 設定
INPUT_DIR = os.path.dirname(os.path.dirname(__file__))  # プロジェクトルート
OUTPUT_DIR = os.path.join(INPUT_DIR, "cleaned_data")    # 出力先フォルダ
THRESHOLD_HEART_RATE = 100
SKIP_POINTS = 4  # 異常値の後にスキップするデータ数

def clean_csv_files():
    # 出力フォルダ作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"フォルダ作成: {OUTPUT_DIR}")

    # CSVファイル一覧取得
    csv_files = glob.glob(os.path.join(INPUT_DIR, "sensor_data_*.csv"))
    
    if not csv_files:
        print("CSVファイルが見つかりません。")
        return

    print(f"{len(csv_files)} 個のファイルを処理します...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            
            # フィルタリング処理
            valid_indices = []
            skip_counter = 0
            
            for i in range(len(df)):
                heart_rate = df.loc[i, "HeartRate"]
                
                # 1. 異常値検知 (HeartRate > 100)
                if heart_rate > THRESHOLD_HEART_RATE:
                    skip_counter = SKIP_POINTS  # カウンタをリセット（ここから4つ飛ばす）
                    continue  # この行は削除
                
                # 2. 復帰待ち期間 (直後の4ポイント)
                if skip_counter > 0:
                    skip_counter -= 1
                    continue  # この行も削除
                
                # 3. 正常データ
                valid_indices.append(i)
            
            # 新しいDataFrame作成
            cleaned_df = df.loc[valid_indices].reset_index(drop=True)
            
            # 保存
            output_path = os.path.join(OUTPUT_DIR, f"cleaned_{file_name}")
            cleaned_df.to_csv(output_path, index=False)
            
            deleted_count = len(df) - len(cleaned_df)
            print(f"完了: {file_name} (削除: {deleted_count}行)")
            
        except Exception as e:
            print(f"エラー: {file_name} - {e}")

if __name__ == "__main__":
    clean_csv_files()