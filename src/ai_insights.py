import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def _expand_feature_names(errors: dict) -> dict:
    mapping = {
        "EC": "Electrical Conductivity",
        "DO": "Dissolved Oxygen",
        "pumpFlow": "Pump Flow",
        "farmLightPct": "Farm Light (%)"
    }
    return {mapping.get(k, k): v for k, v in errors.items()}

def get_insights(anomaly_result: dict) -> str:
    feature_errors = _expand_feature_names(
        anomaly_result.get("feature_reconstruction_errors", anomaly_result.get("feature_errors", {}))
    )

    prompt = f"""
    Indoor farm anomaly detected.

    Device/Twin ID: {anomaly_result['raw_input'].get('deviceId', 'unknown')}
    Timestamp: {anomaly_result['timestamp']}
    Status: {"Anomaly" if anomaly_result.get("is_anomaly") else "Normal"}
    Reconstruction Error: {anomaly_result.get("reconstruction_error", anomaly_result.get("total_score", 0))}

    Feature errors: {feature_errors}
    Domain/Lighting Violations: {anomaly_result.get("violations", [])}

    Provide 2–3 actionable insights in simple language.
    """

    try:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a farm AI insights assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI Insights unavailable: {e})"