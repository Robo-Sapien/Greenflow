import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import os

def _color(bad: bool, base: str) -> str:
    return "red" if bad else base

def render_farm_graph(state: dict, is_anomaly: bool, violations: list[str], title: str = "Farm View"):
    """
    Render a single-tank farm graph (Tank -> Pump -> GrowBed + Lights).
    `state`: latest raw_input dict (must include pH, EC, DO, pumpFlow, farmLightPct).
    """
    st.markdown(f"#### {title}")

    # Build graph
    G = nx.DiGraph()

    ph = float(state.get("pH", 6.0))
    ec = float(state.get("EC", 1.8))
    do = float(state.get("DO", 6.5))
    flow = float(state.get("pumpFlow", 10.0))
    light = float(state.get("farmLightPct", 0))
    device_id = state.get("deviceId", "Tank-1")

    # Node flags
    ph_bad = (ph < 5.5 or ph > 6.8)
    ec_bad = (ec < 1.2 or ec > 2.2)
    do_bad = (do < 4.5 or do > 9.0)
    flow_bad = (flow < 7 or flow > 13)

    # Nodes
    G.add_node(device_id, label=f"{device_id}\nEC={ec}", color=_color(ec_bad, "lightblue"))
    G.add_node("Pump", label=f"Pump\n{flow} L/min", color=_color(flow_bad, "lightgreen"))
    G.add_node("GrowBed", label=f"GrowBed\npH={ph}, DO={do}", color=_color(ph_bad or do_bad, "khaki"))
    G.add_node("Lights", label=f"Lights\n{int(light)}%", color="orange" if light > 0 else "gray")

    # Edges
    G.add_edge(device_id, "Pump")
    G.add_edge("Pump", "GrowBed")
    G.add_edge("Lights", "GrowBed")

    # Render
    net = Network(height="520px", width="100%", directed=True)
    net.from_nx(G)

    if is_anomaly:
        for n in net.nodes:
            n["borderWidth"] = 3
            n["borderWidthSelected"] = 4
            n["color"] = {"background": n["color"], "border": "red"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        html = open(f.name).read()
        os.unlink(f.name)
        st.components.v1.html(html, height=560)

    if violations:
        st.error("🚨 Domain/Lighting Issues:\n" + "\n".join([f"- {v}" for v in violations]))