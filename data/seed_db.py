"""
Creates and seeds data/facility_ops.db — the mock corporate data sources
the agent's tools query against (badge scanner logs, maintenance schedule,
zone metadata).

Design note: timestamps are generated RELATIVE to "now" at seed time,
not hardcoded absolute dates. This matters because our test incidents
need to plausibly correlate (or deliberately NOT correlate) with badge
scans and maintenance windows — hardcoded dates from months ago would
never match "now" and every tool call would trivially return "no match",
which defeats the purpose of testing the agent's reasoning over both
outcomes (correlation found vs. not found).

Run this once before using agent/tools.py. Safe to re-run — it drops and
recreates all tables each time.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "facility_ops.db"


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def build_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS badge_scans;
        DROP TABLE IF EXISTS maintenance_schedule;
        DROP TABLE IF EXISTS zone_metadata;

        CREATE TABLE badge_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            badge_id TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            zone_id TEXT NOT NULL,
            scan_timestamp TEXT NOT NULL,
            access_granted INTEGER NOT NULL
        );

        CREATE TABLE maintenance_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_order_id TEXT NOT NULL,
            zone_id TEXT NOT NULL,
            description TEXT NOT NULL,
            scheduled_start TEXT NOT NULL,
            scheduled_end TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE zone_metadata (
            zone_id TEXT PRIMARY KEY,
            zone_name TEXT NOT NULL,
            tier TEXT NOT NULL,
            is_hazard_adjacent INTEGER NOT NULL
        );
    """)


def seed_data(conn: sqlite3.Connection) -> None:
    now = datetime.now()
    cur = conn.cursor()

    # --- Zone metadata ---
    zones = [
    ("ZONE_SERVER_RM", "Server Room B", "tier_1", 0),
    ("ZONE_PERIMETER_N", "North Perimeter Fence Line", "tier_2", 0),
    ("ZONE_LOADING_DOCK", "Loading Dock 3", "tier_2", 1),
    ("ZONE_LOBBY", "Main Lobby", "tier_2", 0),
    ("ZONE_ELEC_ROOM", "Electrical Substation Room", "tier_1", 1),
    ("ZONE_MAIN_PLAZA", "Main Plaza - Building Entrance", "tier_2", 0),  # <-- add this line
]
    cur.executemany(
        "INSERT INTO zone_metadata VALUES (?, ?, ?, ?)", zones
    )

    # --- Badge scans ---
    # A valid, recent scan in the loading dock (~this matches a "trespassing
    # near loading dock" incident and should downgrade severity)
    # A scan in the server room that happened OVER an hour ago (should NOT
    # correlate with a "right now" unauthorized access detection)
    # No scan at all near the electrical room (tests the "no match" path)
    badge_scans = [
        ("B-1042", "Reza Kiani", "ZONE_LOADING_DOCK", iso(now - timedelta(minutes=3)), 1),
        ("B-1042", "Reza Kiani", "ZONE_LOADING_DOCK", iso(now - timedelta(minutes=45)), 1),
        ("B-2201", "Sara Ahmadi", "ZONE_SERVER_RM", iso(now - timedelta(hours=3)), 1),
        ("B-3390", "Night Shift Contractor", "ZONE_LOBBY", iso(now - timedelta(minutes=20)), 1),
    ]
    cur.executemany(
        "INSERT INTO badge_scans (badge_id, employee_name, zone_id, scan_timestamp, access_granted) "
        "VALUES (?, ?, ?, ?, ?)",
        badge_scans,
    )

    # --- Maintenance schedule ---
    # An active work order in the loading dock right now (explains presence)
    # A completed work order in the lobby from earlier (should NOT explain
    # a current abandoned-object detection)
    maintenance = [
        ("WO-5510", "ZONE_LOADING_DOCK", "Scheduled forklift maintenance",
         iso(now - timedelta(minutes=30)), iso(now + timedelta(minutes=30)), "in_progress"),
        ("WO-5498", "ZONE_LOBBY", "Floor cleaning",
         iso(now - timedelta(hours=6)), iso(now - timedelta(hours=5)), "completed"),
    ]
    cur.executemany(
        "INSERT INTO maintenance_schedule "
        "(work_order_id, zone_id, description, scheduled_start, scheduled_end, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        maintenance,
    )

    conn.commit()


def main():
    conn = sqlite3.connect(DB_PATH)
    build_schema(conn)
    seed_data(conn)

    print(f"Seeded {DB_PATH}")
    for table in ("zone_metadata", "badge_scans", "maintenance_schedule"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()


if __name__ == "__main__":
    main()
