"""
Local tools available to the agent during the ReAct loop. Each tool is a
plain Python function — no decorators, no framework magic — registered in
TOOL_REGISTRY with a name, description, and parameter spec that gets
rendered into the agent's system prompt so the LLM knows what it can call
and how.

Backed by SQLite (data/facility_ops.db, created by data/seed_db.py) rather
than flat JSON — this lets tools express real filtered/windowed queries
via parameterized SQL instead of manual Python filtering, which is both
more realistic to an actual facility-ops backend and demonstrates
proper parameterization (no string-formatted SQL, ever, even for a mock
dataset — that habit should be automatic).

Design choice: tools return STRINGS (not raw dicts), because the observation
in a ReAct loop gets fed back into the LLM's context as text. Formatting the
observation clearly here — rather than dumping raw JSON — measurably helps
small local models reason about it correctly on the next step.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

DB_PATH = Path(__file__).parent.parent / "data" / "facility_ops.db"


def _get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found. Run `python data/seed_db.py` first to create and seed it."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def query_badge_scanner_log(zone_id: str, reference_time: str, window_minutes: int = 10) -> str:
    """
    Look up badge scan events for a given zone within +/- window_minutes of
    a reference timestamp (ISO 8601, e.g. "2026-08-01T22:15:00"). Used to
    determine whether a detected individual correlates with a valid badge
    scan.
    """
    try:
        ref_dt = datetime.fromisoformat(reference_time)
    except ValueError:
        return f"ERROR: reference_time '{reference_time}' is not valid ISO 8601 format."

    window_start = (ref_dt - timedelta(minutes=window_minutes)).isoformat(timespec="seconds")
    window_end = (ref_dt + timedelta(minutes=window_minutes)).isoformat(timespec="seconds")

    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT badge_id, employee_name, scan_timestamp, access_granted
            FROM badge_scans
            WHERE zone_id = ? AND scan_timestamp BETWEEN ? AND ?
            ORDER BY scan_timestamp DESC
            """,
            (zone_id, window_start, window_end),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return (
            f"No badge scans found for zone '{zone_id}' within {window_minutes} minutes "
            f"of {reference_time}. No badge correlation exists for this incident."
        )

    lines = [f"Found {len(rows)} badge scan(s) for zone '{zone_id}' near {reference_time}:"]
    for r in rows:
        status = "GRANTED" if r["access_granted"] else "DENIED"
        lines.append(
            f"  - {r['employee_name']} (badge {r['badge_id']}) at {r['scan_timestamp']} — access {status}"
        )
    return "\n".join(lines)


def query_maintenance_schedule(zone_id: str, reference_time: str) -> str:
    """
    Look up maintenance work orders active in a given zone at a reference
    timestamp (ISO 8601). Used to rule out legitimate maintenance activity
    as an explanation for a detected hazard or object.
    """
    try:
        ref_dt = datetime.fromisoformat(reference_time)
    except ValueError:
        return f"ERROR: reference_time '{reference_time}' is not valid ISO 8601 format."

    ref_iso = ref_dt.isoformat(timespec="seconds")

    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT work_order_id, description, scheduled_start, scheduled_end, status
            FROM maintenance_schedule
            WHERE zone_id = ? AND scheduled_start <= ? AND scheduled_end >= ?
            ORDER BY scheduled_start DESC
            """,
            (zone_id, ref_iso, ref_iso),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return (
            f"No active maintenance work orders found for zone '{zone_id}' "
            f"at {reference_time}. No maintenance explanation available."
        )

    lines = [f"Found {len(rows)} active work order(s) for zone '{zone_id}' at {reference_time}:"]
    for r in rows:
        lines.append(
            f"  - {r['work_order_id']}: {r['description']} "
            f"[{r['status']}] scheduled {r['scheduled_start']} to {r['scheduled_end']}"
        )
    return "\n".join(lines)


def get_zone_metadata(zone_id: str) -> str:
    """
    Look up a zone's tier classification (tier_1 = highest sensitivity,
    tier_2 = standard) and whether it's adjacent to a known physical hazard.
    Use this to help decide escalation severity per the SOPs.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT zone_name, tier, is_hazard_adjacent FROM zone_metadata WHERE zone_id = ?",
            (zone_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return f"No metadata found for zone '{zone_id}'. Zone is not registered in the system."

    hazard_note = "IS adjacent to a known hazard area" if row["is_hazard_adjacent"] else "is NOT hazard-adjacent"
    return (
        f"Zone '{zone_id}' ({row['zone_name']}) is classified as {row['tier'].upper()} and {hazard_note}."
    )


@dataclass
class ToolSpec:
    name: str
    func: Callable[..., str]
    description: str
    parameters: dict[str, str]  # param_name -> description, for prompt rendering


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "query_badge_scanner_log": ToolSpec(
        name="query_badge_scanner_log",
        func=query_badge_scanner_log,
        description=(
            "Check badge scan records for a zone around a specific time. "
            "Use this to determine if a detected individual correlates with "
            "a valid badge access event."
        ),
        parameters={
            "zone_id": "the zone identifier, e.g. 'ZONE_LOADING_DOCK', 'ZONE_SERVER_RM'",
            "reference_time": "ISO 8601 timestamp, e.g. '2026-08-01T22:15:00'",
            "window_minutes": "(optional, default 10) search window in minutes",
        },
    ),
    "query_maintenance_schedule": ToolSpec(
        name="query_maintenance_schedule",
        func=query_maintenance_schedule,
        description=(
            "Check active maintenance work orders for a zone at a specific "
            "time. Use this to rule out legitimate maintenance activity as "
            "an explanation for a detected hazard or object."
        ),
        parameters={
            "zone_id": "the zone identifier, e.g. 'ZONE_LOADING_DOCK'",
            "reference_time": "ISO 8601 timestamp, e.g. '2026-08-01T21:00:00'",
        },
    ),
    "get_zone_metadata": ToolSpec(
        name="get_zone_metadata",
        func=get_zone_metadata,
        description=(
            "Look up a zone's tier classification (tier_1 = highest sensitivity, "
            "tier_2 = standard) and whether it's adjacent to a known physical hazard."
        ),
        parameters={"zone_id": "the zone identifier, e.g. 'ZONE_ELEC_ROOM'"},
    ),
}


def render_tool_descriptions() -> str:
    """Renders the tool registry into the text block injected into the
    agent's system prompt, so the LLM knows what's available."""
    blocks = []
    for spec in TOOL_REGISTRY.values():
        params_str = ", ".join(f"{k} ({v})" for k, v in spec.parameters.items())
        blocks.append(f"- {spec.name}({params_str}): {spec.description}")
    return "\n".join(blocks)