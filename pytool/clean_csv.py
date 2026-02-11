from pathlib import Path
import os
import pandas as pd
import glob
import datetime
import argparse

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parents[1]  # wif_ver（プロジェクトルート）
INPUT_DIR = str(BASE_DIR)
OUTPUT_DIR = str(BASE_DIR / "cleaned_data")    # 出力先フォルダ

THRESHOLD_HEART_RATE = 100

DROP_INITIAL_SECONDS = 60  # 開始1分間を削除
DROP_FINAL_SECONDS = 30    # 終了30秒間を削除

# デフォルト: 空リストだと INPUT_DIR 内の sensor_data_*.csv を処理します
TARGET_FILES = ["sensor_data_20260126_211612_mukokyuu.csv"]

def clean_csv_files(target_files=None, pattern=None, list_only=False):
    """
    CSVクリーニング処理

    - target_files: リスト of ファイル名またはパス（優先）
    - pattern: glob パターン（例: "sensor_data_202601*.csv"）
    - list_only: True の場合、処理対象の一覧のみ表示して終了
    """
    # 出力フォルダ作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"フォルダ作成: {OUTPUT_DIR}")

    files_to_process = []

    # CLIで明示されたファイルがある場合
    if target_files:
        print(f"ターゲット指定モード（CLI指定）: {len(target_files)} 件")
        for fname in target_files:
            if os.path.isabs(fname):
                fpath = fname
            else:
                fpath = os.path.join(INPUT_DIR, fname)

            if os.path.exists(fpath):
                files_to_process.append(fpath)
            else:
                print(f"警告: ファイルが見つかりません -> {fpath}")
    elif pattern:
        print(f"パターン検索モード: {pattern}")
        search_pattern = os.path.join(INPUT_DIR, pattern)
        files_to_process = glob.glob(search_pattern)
    elif TARGET_FILES:
        # 設定ファイルの TARGET_FILES を使う
        print(f"ターゲット指定モード（設定）: {len(TARGET_FILES)} 件")
        for fname in TARGET_FILES:
            if os.path.isabs(fname):
                fpath = fname
            else:
                fpath = os.path.join(INPUT_DIR, fname)

            if os.path.exists(fpath):
                files_to_process.append(fpath)
            else:
                print(f"警告: ファイルが見つかりません -> {fpath}")
    else:
        # 指定がない場合（全件モード）
        print("全件処理モード: sensor_data_*.csv を検索します")
        search_pattern = os.path.join(INPUT_DIR, "sensor_data_*.csv")
        files_to_process = glob.glob(search_pattern)

    if not files_to_process:
        print(f"処理対象のCSVファイルが見つかりません。\n検索パス: {os.path.join(INPUT_DIR, 'sensor_data_*.csv')}")
        return

    # list_only の場合は一覧表示して終了
    if list_only:
        print("処理対象ファイル一覧:")
        for p in files_to_process:
            print("  -", p)
        return

    print(f"{len(files_to_process)} 個のファイルを処理します...")

    for file_path in files_to_process:
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

            # 開始時刻と終了時刻を取得
            start_time = df["Timestamp"].iloc[0]
            end_time = df["Timestamp"].iloc[-1]

            # カットオフ時刻の計算
            cutoff_start = start_time + datetime.timedelta(seconds=DROP_INITIAL_SECONDS)
            cutoff_end = end_time - datetime.timedelta(seconds=DROP_FINAL_SECONDS)

            # フィルタリング実行
            cleaned_df = df[(df["Timestamp"] >= cutoff_start) & (df["Timestamp"] < cutoff_end)].copy()

            # データが空になってしまった場合のガード
            if cleaned_df.empty:
                print(f"警告: 削除範囲が広すぎてデータが残りませんでした -> {file_name}")
                continue

            # 心拍数のクリッピング（必要に応じて）
            cleaned_df["HeartRate"] = pd.to_numeric(cleaned_df["HeartRate"], errors="coerce")
            cleaned_df["HeartRate"] = cleaned_df["HeartRate"].clip(upper=THRESHOLD_HEART_RATE)

            # 保存
            output_path = os.path.join(OUTPUT_DIR, f"cleaned_{file_name}")
            cleaned_df.to_csv(output_path, index=False)

            deleted_count = len(df) - len(cleaned_df)
            print(f"完了: {file_name} (元: {len(df)}行 -> 処理後: {len(cleaned_df)}行, 削除: {deleted_count}行)")

        except Exception as e:
            print(f"エラー: {file_name} - {e}")

def _parse_args():
    parser = argparse.ArgumentParser(description="CSVクリーニングツール")
    parser.add_argument('--files', '-f', nargs='+', help='処理するファイル名またはパスを指定（複数可）。相対パスはプロジェクトルート基準）')
    parser.add_argument('--pattern', '-p', help='globパターンで検索（例: sensor_data_202601*.csv）')
    parser.add_argument('--list', action='store_true', help='処理対象ファイルを一覧表示して終了')
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    # CLI引数があればそれを優先して処理
    clean_csv_files(target_files=args.files, pattern=args.pattern, list_only=args.list)