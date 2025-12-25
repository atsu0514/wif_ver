import os
import pandas as pd
import glob

# 設定
INPUT_DIR = os.path.dirname(os.path.dirname(__file__))  # プロジェクトルート
OUTPUT_DIR = os.path.join(INPUT_DIR, "cleaned_data")    # 出力先フォルダ
THRESHOLD_HEART_RATE = 100

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
            
            # HeartRate を「削除せず上限100でキープ」
            df["HeartRate"] = pd.to_numeric(df["HeartRate"], errors="coerce")
            clamped_count = int((df["HeartRate"] > THRESHOLD_HEART_RATE).sum())
            df["HeartRate"] = df["HeartRate"].clip(upper=THRESHOLD_HEART_RATE)

            cleaned_df = df  # 行は削除しない
            
            # 保存
            output_path = os.path.join(OUTPUT_DIR, f"cleaned_{file_name}")
            cleaned_df.to_csv(output_path, index=False)
            
            deleted_count = len(df) - len(cleaned_df)
            print(f"完了: {file_name} (削除: {deleted_count}行)")
            
        except Exception as e:
            print(f"エラー: {file_name} - {e}")

if __name__ == "__main__":
    clean_csv_files()