import streamlit as st
import pandas as pd
import time
from src import run_inference
from src.utils import check_domain_and_lights
import src.ai_insights as ai_insights

def _load_live_df():
    """Load test dataset as mock sensor stream."""
    df = pd.read_csv("data/test.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def render_live_tab():
    st.header("Live Farm Dashboard")

    if "live_df" not in st.session_state:
        st.session_state.live_df = _load_live_df()

    if st.button("▶️ Start Stream"):
        placeholder = st.empty()
        chart_data = pd.DataFrame(columns=[
            "timestamp", "pH", "Electrical Conductivity",
            "Dissolved Oxygen", "Pump Flow", "Farm Light (%)"
        ])

        for _, row in st.session_state.live_df.iterrows():
            row_dict = row.to_dict()
            result = run_inference.run_single_record(row_dict)

            # Update chart data
            chart_data = pd.concat([chart_data, pd.DataFrame([{
                "timestamp": row["timestamp"],
                "pH": row["pH"],
                "Electrical Conductivity": row["EC"],
                "Dissolved Oxygen": row["DO"],
                "Pump Flow": row["pumpFlow"],
                "Farm Light (%)": row["farmLightPct"],
            }])], ignore_index=True)

            # Domain & AI
            violations = check_domain_and_lights(result["raw_input"])
            result["violations"] = violations
            insights = ai_insights.get_insights(result)

            # Refresh dashboard
            with placeholder.container():
                st.subheader(f"⏱️ Timestamp: {result['timestamp']}")
                status = "⚠️ Anomaly" if result["is_anomaly"] else "✅ Normal"
                st.markdown(f"### Current Status: {status}")

                # Metrics row
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.metric("pH", f"{row['pH']:.2f}")
                with c2: st.metric("Electrical Conductivity (mS/cm)", f"{row['EC']:.2f}")
                with c3: st.metric("Dissolved Oxygen (mg/L)", f"{row['DO']:.2f}")
                with c4: st.metric("Pump Flow (L/min)", f"{row['pumpFlow']:.1f}")
                with c5: st.metric("Farm Light (%)", f"{row['farmLightPct']}")

                # Trend chart
                st.line_chart(chart_data.set_index("timestamp")[
                    ["pH", "Electrical Conductivity", "Dissolved Oxygen", "Pump Flow", "Farm Light (%)"]
                ])

                # Alerts
                if violations:
                    st.error("🚨 Alerts:\n" + "\n".join([f"- {v}" for v in violations]))

                # AI Insights
                if insights:
                    st.info(f"🤖 AI Insights: {insights}")

            time.sleep(1)  # simulate real-time