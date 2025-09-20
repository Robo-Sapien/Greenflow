# src/digital_twin.py
from typing import Dict, Any, List

# ---- Domain ranges for indoor leafy greens (tweak per crop/phase) ----
def domain_ranges() -> Dict[str, Dict[str, float]]:
    return {
        "pH": {"min": 5.5, "max": 6.8},
        "EC": {"min": 1.2, "max": 2.2},     # mS/cm
        "DO": {"min": 4.5, "max": 9.0},     # mg/L
        "pumpFlow": {"min": 7.0, "max": 13.0},  # LPM (example)
        # Light runtime checked by schedule (06:00-22:00 ON)
        "farmLightPct": {"min": 0.0, "max": 100.0}
    }

# ---- Twin Registry (simplified) ----
# You can expand to many tanks, dosers, HVAC, etc.
TWINS: Dict[str, Dict[str, Any]] = {
    "farm-1": {
        "type": "Farm",
        "label": "Farm-1",
        "components": ["tank-1", "tank-2", "pump-1", "light-1", "fan-1", "doser-1"]
    },
    "tank-1": {"type": "Tank", "label": "Tank-1", "props": ["pH", "EC", "DO"]},
    "tank-2": {"type": "Tank", "label": "Tank-2", "props": ["pH", "EC", "DO"]},
    "pump-1": {"type": "Pump", "label": "Main Pump", "props": ["pumpFlow"]},
    "light-1": {"type": "Lighting", "label": "Light Array", "props": ["farmLightPct"]},
    "fan-1":  {"type": "Fan", "label": "Air Fan", "props": ["fanSpeed"]},
    "doser-1":{"type": "Doser", "label": "Nutrient Doser", "props": ["doserActivity"]}
}

# Directed relationships (source -> target)
RELATIONSHIPS: List[Dict[str, str]] = [
    {"from": "farm-1", "to": "tank-1", "name": "hosts"},
    {"from": "farm-1", "to": "tank-2", "name": "hosts"},
    {"from": "farm-1", "to": "pump-1", "name": "hosts"},
    {"from": "farm-1", "to": "light-1", "name": "hosts"},
    {"from": "farm-1", "to": "fan-1",  "name": "hosts"},
    {"from": "farm-1", "to": "doser-1","name": "hosts"},

    # Flow/Influence
    {"from": "pump-1",  "to": "tank-1", "name": "feeds"},
    {"from": "pump-1",  "to": "tank-2", "name": "feeds"},
    {"from": "doser-1", "to": "tank-1", "name": "conditions"},
    {"from": "doser-1", "to": "tank-2", "name": "conditions"},
    {"from": "light-1", "to": "tank-1", "name": "illuminates"},
    {"from": "light-1", "to": "tank-2", "name": "illuminates"},
    {"from": "fan-1",   "to": "tank-1", "name": "circulates_air"},
    {"from": "fan-1",   "to": "tank-2", "name": "circulates_air"},
]

def map_row_to_twin_id(row: Dict[str, Any]) -> str:
    """
    Decide which twin this row belongs to.
    If your data already includes 'twin_id', use it.
    Otherwise, use a routing rule (e.g., by sensor/channel).
    """
    twin_id = row.get("twin_id")
    if twin_id and twin_id in TWINS:
        return twin_id
    # default route (single tank setup)
    return "tank-1"

def get_neighbors(twin_id: str) -> Dict[str, Any]:
    """
    Return upstream/downstream neighbors of a twin based on RELATIONSHIPS.
    """
    upstream = [r["from"] for r in RELATIONSHIPS if r["to"] == twin_id]
    downstream = [r["to"] for r in RELATIONSHIPS if r["from"] == twin_id]
    return {"upstream": upstream, "downstream": downstream}

def get_twin_context(twin_id: str) -> Dict[str, Any]:
    """
    Compact context object for GPT: twin info + neighbors + domain ranges.
    """
    twin = TWINS.get(twin_id, {"type": "Unknown", "label": twin_id, "props": []})
    nn = get_neighbors(twin_id)
    return {
        "twin_id": twin_id,
        "type": twin.get("type"),
        "label": twin.get("label"),
        "props": twin.get("props", []),
        "neighbors": nn,
        "farm": {
            "id": "farm-1",
            "components": TWINS["farm-1"]["components"]
        },
        "domain_ranges": domain_ranges()
    }