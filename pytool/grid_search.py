from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import optuna


# =========================
# 設定
# =========================
FEATURE_COLS = ["MovingRange", "HeartRate", "BreathingRate"]
TARGET_COLS = ["HeartRate", "BreathingRate"]

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "cleaned_data"

CSV_LIST = sorted([p.name for p in DATA_DIR.glob("cleaned_sensor_data_*.csv")])

# 設定変更: Epoch数を増やし、Early Stopping用の忍耐値を設定
SEARCH_EPOCHS = 200  # 十分に大きく変更
PATIENCE = 15        # 改善が見られなくなってから何回待つか
N_TRIALS = 50


def resolve_csv_path(name: str) -> Path | None:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p

    candidates = [DATA_DIR / name, BASE_DIR / name, Path.cwd() / name]
    for c in candidates:
        if c.exists():
            return c
    return None


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x)
        b, t, _ = x.shape
        dec_in = torch.zeros((b, t, self.out.out_features), device=x.device, dtype=x.dtype)
        dec_out, _ = self.decoder(dec_in, (h, c))
        recon = self.out(dec_out)
        return recon


class SensorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler_x: MinMaxScaler, scaler_y: MinMaxScaler, window_size: int):
        self.window_size = int(window_size)

        features_df = df[FEATURE_COLS]
        targets_df = df[TARGET_COLS]

        self.features_scaled = scaler_x.transform(features_df)
        self.targets_scaled = scaler_y.transform(targets_df)

        self.length = max(0, len(df) - self.window_size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        x = self.features_scaled[idx : idx + self.window_size]
        y = self.targets_scaled[idx : idx + self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_all_dfs() -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []

    for name in CSV_LIST:
        p = resolve_csv_path(name)
        if not p:
            continue
        try:
            df = pd.read_csv(p, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
            if df.empty:
                continue
            if not all(c in df.columns for c in ["Timestamp", *FEATURE_COLS]):
                continue
            dfs.append(df)
        except Exception:
            continue

    return dfs


def objective(trial: optuna.Trial, train_dfs, val_dfs, scaler_x, scaler_y, device: torch.device) -> float:
    window_size = trial.suggest_categorical("WINDOW_SIZE", [15, 30, 45])
    hidden_size = trial.suggest_categorical("HIDDEN_SIZE", [32, 64])
    num_layers = trial.suggest_int("NUM_LAYERS", 1, 2)
    batch_size = trial.suggest_categorical("BATCH_SIZE", [16, 32])
    learning_rate = trial.suggest_float("LEARNING_RATE", 1e-4, 1e-2, log=True)

    train_datasets = [SensorDataset(df, scaler_x, scaler_y, window_size) for df in train_dfs if len(df) > window_size]
    val_datasets = [SensorDataset(df, scaler_x, scaler_y, window_size) for df in val_dfs if len(df) > window_size]

    if not train_datasets or not val_datasets:
        raise optuna.exceptions.TrialPruned()

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False)

    model = LSTMEncoderDecoder(input_size=3, hidden_size=hidden_size, num_layers=num_layers, output_size=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_loss_history = []
    val_loss_history = []
    
    # Early Stopping用変数
    best_val_loss = float('inf')
    no_improve_cnt = 0
    best_epoch = 0

    for epoch in range(SEARCH_EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_n += x.size(0)

        avg_train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                l = criterion(model(x), y).item()
                val_loss_sum += l * x.size(0)
                n += x.size(0)

        avg_val_loss = float(val_loss_sum / max(n, 1))

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # === Early Stopping 判定 ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        # 指定回数連続で改善しなかったらストップ
        if no_improve_cnt >= PATIENCE:
            # print(f"  Early Stopping at Epoch {epoch+1} (Best: {best_epoch+1}, Val: {best_val_loss:.6f})")
            break
        # ==========================

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            trial.set_user_attr("train_loss_history", train_loss_history)
            trial.set_user_attr("val_loss_history", val_loss_history)
            trial.set_user_attr("best_epoch", best_epoch) # 記録
            raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("train_loss_history", train_loss_history)
    trial.set_user_attr("val_loss_history", val_loss_history)
    trial.set_user_attr("best_epoch", best_epoch) # 記録

    # 最も良かった時のLossを返す (最後のEpochの値ではない)
    return best_val_loss


def main():
    print("Loading data...")
    dfs = load_all_dfs()
    print(f"Found {len(dfs)} valid CSV files (out of {len(CSV_LIST)})")

    if len(dfs) < 3:
        print("Not enough data files for train/val split (need >= 3).")
        return

    train_dfs = dfs[:-2]
    val_dfs = dfs[-2:]
    print(f"Train files: {len(train_dfs)}, Val files: {len(val_dfs)}")

    full_df = pd.concat(train_dfs, ignore_index=True)
    scaler_x = MinMaxScaler().fit(full_df[FEATURE_COLS])
    scaler_y = MinMaxScaler().fit(full_df[TARGET_COLS])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(lambda t: objective(t, train_dfs, val_dfs, scaler_x, scaler_y, device), n_trials=N_TRIALS)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    # === 追加修正: 詳細データのCSV保存 ===
    # Optunaの全TrialデータをDataFrame化
    df_trials = study.trials_dataframe()
    # 保存先
    details_path = BASE_DIR / "optuna_details.csv"
    df_trials.to_csv(details_path, index=False)
    print(f"Saved detailed trials to {details_path}")
    # ===================================

    # === 追加修正: ベストなEpoch数も取得して保存 ===
    best_trial = study.best_trial
    best_epoch_num = best_trial.user_attrs.get("best_epoch", "Unknown")
    # ============================================

    result_path = BASE_DIR / "optuna_result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Best Val Loss: {study.best_value}\n")
        f.write(f"Best Params: {study.best_params}\n")
        f.write(f"Best Epoch: {best_epoch_num}\n")  # ← ここを追加
        
    print(f"Saved result to {result_path}")


if __name__ == "__main__":
    main()