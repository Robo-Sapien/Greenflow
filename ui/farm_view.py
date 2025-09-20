import networkx as nx
from pyvis.network import Network
import streamlit as st
import tempfile, os

def render_farm_state(state: dict, insights: list | None, is_anomaly: bool, violations: list | None):
    """
    state: the raw_input dict (timestamp + features)
    insights: list of strings from GPT or heuristics
    is_anomaly: boolean
    violations: domain rule messages (list)
    """
    # Node colors by health
    def color_ok(red: bool, default_color: str):
        return "red" if red else default_color

    G = nx.DiGraph()

    # Tank node (EC primarily)
    ec = state.get("EC", 1.8)
    tank_bad = (ec < 1.2 or ec > 2.2)
    G.add_node("Tank", label=f"Tank\nEC={ec}", color=color_ok(tank_bad, "lightblue"))

    # GrowBed node (pH)
    ph = state.get("pH", 6.0)
    grow_bad = (ph < 5.5 or ph > 6.8)
    G.add_node("GrowBed", label=f"GrowBed\npH={ph}", color=color_ok(grow_bad, "lightgreen"))

    # Pump node (flow)
    pf = state.get("pumpFlow", 10.0)
    pump_bad = (pf < 7 or pf > 13)
    G.add_node("Pump", label=f"Pump\nFlow={pf}", color=color_ok(pump_bad, "green"))

    # Lights node (farmLightPct)
    lp = state.get("farmLightPct", 0)
    light_bad = False  # domain schedule is in violations; keep neutral coloring
    light_color = "yellow" if lp > 0 else "gray"
    G.add_node("Lights", label=f"Lights\n{lp}%", color=color_ok(light_bad, light_color))

    # Edges (flow & influence)
    G.add_edge("Tank", "Pump")
    G.add_edge("Pump", "GrowBed")
    G.add_edge("Lights", "GrowBed")

    # render
    net = Network(height="460px", width="100%", directed=True)
    net.from_nx(G)
    # emphasize whole graph if anomaly
    if is_anomaly:
        for nid in net.nodes:
            nid["borderWidth"] = 3
            nid["borderWidthSelected"] = 4

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        st.components.v1.html(open(f.name).read(), height=520)
        os.unlink(f.name)

    # Panel: violations + insights
    if violations:
        st.error("🚨 Domain Issues\n- " + "\n- ".join(violations))
    if insights:
        st.info("🤖 AI Insights\n- " + "\n- ".join(insights))