import os, json, torch
import numpy as np
from .model import LSTMAutoencoder
from .utils import load_scaler, reconstruction_errors

_MODEL, _SCALER, _CFG, _THRESH = None, None, None, None
_MODEL_DIR = "outputs/model"

def _lazy_init():
    global _MODEL, _SCALER, _CFG, _THRESH
    if _MODEL is not None: return

    with open(os.path.join(_MODEL_DIR,"config.json")) as f: _CFG = json.load(f)
    with open(os.path.join(_MODEL_DIR,"threshold.json")) as f: _THRESH = json.load(f)["threshold"]

    _MODEL = LSTMAutoencoder(n_features=len(_CFG["features"]),
                             hidden_size=_CFG["hidden"],
                             seq_len=_CFG["window"])
    state = torch.load(os.path.join(_MODEL_DIR,"model.pt"), map_location="cpu")
    _MODEL.load_state_dict(state, strict=False)
    _MODEL.eval()
    _SCALER = load_scaler(os.path.join(_MODEL_DIR,"scaler.pkl"))

def run(raw_json):
    _lazy_init()
    data = json.loads(raw_json)
    rows = data["inputs"]

    X = np.array([[float(r[f]) for f in _CFG["features"]] for r in rows], dtype=np.float32)
    Xs = _SCALER.transform(X)
    seqs = [Xs[-_CFG["window"]:]]
    seqs = np.stack(seqs).astype(np.float32)

    with torch.no_grad():
        xb = torch.from_numpy(seqs)
        recon = _MODEL(xb)
        total_err = reconstruction_errors(xb.numpy(), recon.numpy())[0]
        feat_errs = ((xb.numpy()-recon.numpy())**2).mean(axis=1)[0]

    return {
        "threshold": _THRESH,
        "scores": [{
            "timestamp": rows[-1]["timestamp"],
            "total_score": float(total_err),
            "is_anomaly": bool(total_err > _THRESH),
            "feature_errors": {f: float(e) for f,e in zip(_CFG["features"], feat_errs)},
            "raw_input": rows[-1]
        }]
    }