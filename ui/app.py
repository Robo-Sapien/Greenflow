import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from live_tab import render_live_tab
from sim_tab import render_sim_tab

def main():
    st.set_page_config(page_title="🌱 Greenflow: Indoor Farm Digital Twin", layout="wide")
    st.title("🌱 Greenflow: Hydroponics Indoor Farm Digital Twin")

    # Environment readiness (for AI Insights)
    ok_endpoint = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))
    ok_key = bool(os.getenv("AZURE_OPENAI_API_KEY"))
    ok_deploy = bool(os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    with st.sidebar:
        st.markdown("### 🔐 Azure AI Foundry")
        st.write(f"Endpoint set: {'✅' if ok_endpoint else '❌'}")
        st.write(f"API Key set: {'✅' if ok_key else '❌'}")
        st.write(f"Deployment set: {'✅' if ok_deploy else '❌'}")
        st.write(f"Model deployed: GPT-4.1-nano (Finetuned)")
        st.write(f"Anomaly Detector: LSTM Autoencoder")

    tab_live, tab_sim = st.tabs(["📡 Live Farm", "🕹️ Simulator"])
    with tab_live:
        render_live_tab()
    with tab_sim:
        render_sim_tab()

if __name__ == "__main__":
    main()