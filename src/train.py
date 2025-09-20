import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim import Adam

# Robust imports whether you run "python src/train.py" or from package
try:
    from model import LSTMAutoencoder
    from utils import make_windows, fit_scaler_and_scale, save_scaler
except ImportError:
    from src.model import LSTMAutoencoder
    from src.utils import make_windows, fit_scaler_and_scale, save_scaler

def parse_args():
    p = argparse.ArgumentParser(description="Train LSTM Autoencoder for hydroponics anomalies")
    p.add_argument("--data", required=True, help="Path to training CSV with timestamp + features.")
    p.add_argument("--val-data", required=False, help="Optional path to external validation CSV.")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--features", nargs="+", required=False,
                   default=["pH", "EC", "DO", "pumpFlow", "farmLightPct"],
                   help="Feature columns to use.")
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction for validation if --val-data not provided.")
    p.add_argument("--q", type=float, default=0.995,
                   help="Quantile for anomaly threshold based on val errors.")
    p.add_argument("--outdir", default="outputs/model", help="Where to save artifacts.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    if args.val_data:
        df_train = pd.read_csv(args.data)
        df_val = pd.read_csv(args.val_data)
    else:
        df_all = pd.read_csv(args.data)
        n = len(df_all)
        split_idx = int(n * (1 - args.val_split))
        df_train = df_all.iloc[:split_idx].copy()
        df_val = df_all.iloc[split_idx:].copy()

    # Validate columns
    required = [args.timestamp_col] + args.features
    for col in required:
        if col not in df_train.columns or col not in df_val.columns:
            raise ValueError(f"Missing column '{col}' in training/validation CSV.")

    # Sort by time
    df_train = df_train.sort_values(args.timestamp_col).reset_index(drop=True)
    df_val = df_val.sort_values(args.timestamp_col).reset_index(drop=True)

    # Scale
    scaler, df_train_s = fit_scaler_and_scale(df_train, args.features)
    df_val_s = df_val.copy()
    df_val_s[args.features] = scaler.transform(df_val[args.features])

    # Windows
    X_train = make_windows(df_train_s, args.features, args.window, args.stride)
    X_val = make_windows(df_val_s, args.features, args.window, args.stride)
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise RuntimeError("Not enough data to form windows. Reduce --window or provide more rows.")

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMAutoencoder(
        n_features=X_train.shape[2],
        hidden_size=args.hidden,
        seq_len=args.window,
        num_layers=args.layers
    ).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    # Loaders
    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds = TensorDataset(torch.from_numpy(X_val))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # Train
    best_val = float("inf")
    best_state = None
    thresh = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for (xb,) in train_dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            y_true_all, y_pred_all = [], []
            for (xb,) in val_dl:
                xb = xb.to(device)
                recon = model(xb)
                loss = crit(recon, xb)
                val_loss += loss.item() * xb.size(0)
                y_true_all.append(xb.cpu().numpy())
                y_pred_all.append(recon.cpu().numpy())
            val_loss /= len(val_ds)
            y_true_all = np.concatenate(y_true_all, axis=0)
            y_pred_all = np.concatenate(y_pred_all, axis=0)
            val_errors = ((y_true_all - y_pred_all) ** 2).mean(axis=(1, 2))
            thresh = float(np.quantile(val_errors, args.q))

        print(
            f"Epoch {epoch}:\n"
            f"  Training reconstruction error (MSE): {tr_loss:.4f}\n"
            f"  Validation reconstruction error (MSE): {val_loss:.4f}\n"
            f"  Anomaly threshold (99.5% quantile): {thresh:.4f}\n"
        )
        if val_loss < best_val or best_state is None:
            best_val = val_loss
            best_state = model.state_dict()

    # Save artifacts
    os.makedirs(args.outdir, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))
    save_scaler(scaler, os.path.join(args.outdir, "scaler.pkl"))

    cfg = {
        "features": args.features,
        "window": args.window,
        "stride": args.stride,
        "hidden": args.hidden,
        "layers": args.layers
    }
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.outdir, "threshold.json"), "w") as f:
        json.dump({"threshold": float(thresh)}, f, indent=2)

    print(f"✅ Training complete. Artifacts saved to {args.outdir}")

if __name__ == "__main__":
    main()