import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def make_windows(df, feature_cols, window, stride):
    X = df[feature_cols].values.astype(np.float32)
    seqs = []
    for start in range(0, len(X) - window + 1, stride):
        seqs.append(X[start:start+window])
    return np.stack(seqs) if seqs else np.empty((0, window, len(feature_cols)), dtype=np.float32)

def fit_scaler_and_scale(train_df, feature_cols):
    scaler = StandardScaler()
    X = train_df[feature_cols].values
    scaler.fit(X)
    df_scaled = train_df.copy()
    df_scaled[feature_cols] = scaler.transform(X)
    return scaler, df_scaled

def save_scaler(scaler, path): joblib.dump(scaler, path)
def load_scaler(path): return joblib.load(path)

def reconstruction_errors(y_true, y_pred): return ((y_true - y_pred)**2).mean(axis=(1,2))
def quantile_threshold(errors, q=0.995): return float(np.quantile(errors, q))

# domain + lighting rules
def check_domain_and_lights(raw_row, photoperiod=(6,22)):
    issues = []
    h = int(raw_row["timestamp"][11:13])
    if raw_row["pH"] < 5.5 or raw_row["pH"] > 6.8:
        issues.append(f"pH={raw_row['pH']} out of range [5.5,6.8]")
    if raw_row["EC"] < 1.2 or raw_row["EC"] > 2.2:
        issues.append(f"EC={raw_row['EC']} out of range [1.2,2.2]")
    if raw_row["DO"] < 4.5 or raw_row["DO"] > 9.0:
        issues.append(f"DO={raw_row['DO']} out of range [4.5,9.0]")
    if raw_row["pumpFlow"] < 7 or raw_row["pumpFlow"] > 13:
        issues.append(f"pumpFlow={raw_row['pumpFlow']} out of range [7,13]")
    if photoperiod[0] <= h < photoperiod[1]:
        if raw_row["farmLightPct"] == 0:
            issues.append("Lights OFF during scheduled ON period")
    else:
        if raw_row["farmLightPct"] > 0:
            issues.append(f"Lights ON at night ({raw_row['farmLightPct']}%)")
    return issues