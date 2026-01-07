import os
import pandas as pd
import glob
import datetime

# 設定
INPUT_DIR = os.path.dirname(os.path.dirname(__file__))  # プロジェクトルート
OUTPUT_DIR = os.path.join(INPUT_DIR, "cleaned_data")    # 出力先フォルダ
THRESHOLD_HEART_RATE = 100
DROP_INITIAL_SECONDS = 60  # 開始1分間を削除

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
            # Timestampを日付型として読み込む
            df = pd.read_csv(file_path, parse_dates=["Timestamp"])
            
            # データが空の場合はスキップ
            if df.empty:
                print(f"スキップ (空ファイル): {file_name}")
                continue
                
            # Timestampでソート
            df = df.sort_values("Timestamp").reset_index(drop=True)
            
            # 開始時刻を取得
            start_time = df["Timestamp"].iloc[0]
            cutoff_time = start_time + datetime.timedelta(seconds=DROP_INITIAL_SECONDS)
            
            # 開始1分以降のデータのみ残す
            cleaned_df = df[df["Timestamp"] >= cutoff_time].copy()
            
            # もしコメントアウトされていた心拍数制限を戻すならここで行う
            # cleaned_df["HeartRate"] = pd.to_numeric(cleaned_df["HeartRate"], errors="coerce")
            # cleaned_df["HeartRate"] = cleaned_df["HeartRate"].clip(upper=THRESHOLD_HEART_RATE)

            # 保存
            output_path = os.path.join(OUTPUT_DIR, f"cleaned_{file_name}")
            cleaned_df.to_csv(output_path, index=False)
            
            deleted_count = len(df) - len(cleaned_df)
            print(f"完了: {file_name} (削除: {deleted_count}行, 開始時刻: {start_time}, カットオフ: {cutoff_time})")
            
        except Exception as e:
            print(f"エラー: {file_name} - {e}")

if __name__ == "__main__":
    clean_csv_files()