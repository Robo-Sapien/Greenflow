import json, pandas as pd
from src import score
from src.utils import check_domain_and_lights

def run_batch(csv_path):
    df = pd.read_csv(csv_path)
    payload = {"inputs": df.to_dict(orient="records")}
    return score.run(json.dumps(payload))

def run_single_record(row_dict):
    # Convert datetime-like objects into ISO strings
    safe_row = {
        k: (v.isoformat() if hasattr(v, "isoformat") else str(v) if isinstance(v, pd.Timestamp) else v)
        for k, v in row_dict.items()
    }
    payload = {"inputs": [safe_row]}
    return score.run(json.dumps(payload))["scores"][0]

if __name__ == "__main__":
    res = run_batch("data/test.csv")
    print(json.dumps(res, indent=2))
    for s in res["scores"]:
        print("Violations:", check_domain_and_lights(s["raw_input"]))