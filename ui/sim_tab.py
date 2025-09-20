import streamlit as st
import pandas as pd
from src import run_inference
from src.utils import check_domain_and_lights
from src import ai_insights
from ui.graph import render_farm_graph

def render_sim_tab():
    st.caption("Drive the farm with sliders and see the digital twin + insights react instantly.")

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        device = st.selectbox("Device / Tank", ["Tank-1"], index=0)
        ph = st.slider("pH", 4.5, 7.5, 6.0, 0.1)
    with c2:
        ec = st.slider("Electrical Conductivity (mS/cm)", 0.8, 2.8, 1.8, 0.1)
        do = st.slider("Dissolved Oxygen (mg/L)", 3.0, 10.0, 6.8, 0.1)
    with c3:
        pump = st.slider("Pump Flow (L/min)", 5.0, 15.0, 10.0, 0.5)
        light = st.slider("Farm Light (%)", 0, 100, 50, 5)

    ts = pd.Timestamp.utcnow().isoformat()
    row = {
        "timestamp": ts,
        "deviceId": device,
        "pH": ph,
        "EC": ec,
        "DO": do,
        "pumpFlow": pump,
        "farmLightPct": light,
    }

    # Run inference
    result = run_inference.run_single_record(row)
    recon_err = result.get("reconstruction_error", result.get("total_score", 0.0))
    feat_errs = result.get("feature_reconstruction_errors", result.get("feature_errors", {}))

    violations = check_domain_and_lights(row)
    result["violations"] = violations

    # Metrics
    st.markdown("---")
    cA, cB, cC, cD = st.columns(4)
    with cA: st.metric("Timestamp", ts)
    with cB: st.metric("Device", device)
    with cC: st.metric("Status", "⚠️ ANOMALY" if result["is_anomaly"] else "✅ Normal")
    with cD: st.metric("Reconstruction Error", f"{recon_err:.4f}")

    # Graph
    render_farm_graph(
        state=row,
        is_anomaly=result["is_anomaly"],
        violations=violations,
        title="Digital Twin (Simulator)"
    )

    # Details
    with st.expander("🔎 Feature Reconstruction Errors"):
        st.json(feat_errs)

    with st.expander("🤖 AI Insights"):
        insight = ai_insights.get_insights(result)
        st.info(insight)