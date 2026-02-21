#!/usr/bin/env python3
import argparse
import os
import sqlite3
from typing import Dict, Optional

import questionary

# Store DB under ./database/
DEFAULT_DB_DIR = os.environ.get("ORQESTRA_DB_DIR", "./database")
DEFAULT_DB = os.environ.get("ORQESTRA_DB", os.path.join(DEFAULT_DB_DIR, "orqestra.db"))


# ---------- DB ----------
def ensure_parent_dir(db_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def db_connect(path: str) -> sqlite3.Connection:
    ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO settings(key, value, updated_at)
        VALUES(?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=datetime('now');
        """,
        (key, value),
    )
    conn.commit()


def get_setting(conn: sqlite3.Connection, key: str) -> Optional[str]:
    cur = conn.execute("SELECT value FROM settings WHERE key=?;", (key,))
    row = cur.fetchone()
    return row[0] if row else None


def get_all_settings(conn: sqlite3.Connection) -> Dict[str, str]:
    cur = conn.execute("SELECT key, value FROM settings ORDER BY key;")
    return {k: v for (k, v) in cur.fetchall()}


# ---------- TUI ONBOARDING ----------
def onboard(conn: sqlite3.Connection) -> None:
    existing = get_all_settings(conn)

    questionary.print("\n=== Orqestra Onboarding ===\n", style="bold")

    service_name = questionary.text(
        "Service name:",
        default=existing.get("service.name", "orqestra"),
    ).ask()
    if service_name is None:
        raise SystemExit(130)

    api_base = questionary.text(
        "API base URL:",
        default=existing.get("api.base", "http://127.0.0.1:8788"),
    ).ask()
    if api_base is None:
        raise SystemExit(130)

    log_level = questionary.select(
        "Select log level:",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default=existing.get("log.level", "INFO"),
    ).ask()
    if log_level is None:
        raise SystemExit(130)

    features = questionary.checkbox(
        "Enable features (Space to toggle, Enter to confirm):",
        choices=[
            questionary.Choice(
                "Enable authentication",
                checked=existing.get("auth.enabled", "false") == "true",
            ),
            questionary.Choice(
                "Enable metrics",
                checked=existing.get("metrics.enabled", "false") == "true",
            ),
            questionary.Choice(
                "Enable auto-update",
                checked=existing.get("autoupdate.enabled", "false") == "true",
            ),
        ],
    ).ask()
    if features is None:
        raise SystemExit(130)

    auth_enabled = "Enable authentication" in features
    metrics_enabled = "Enable metrics" in features
    autoupdate_enabled = "Enable auto-update" in features

    auth_token = existing.get("auth.token", "")
    if auth_enabled:
        auth_token = questionary.password("Auth token (hidden):").ask()
        if auth_token is None or auth_token.strip() == "":
            questionary.print("Auth token is required when authentication is enabled.", style="bold fg:red")
            raise SystemExit(2)
    else:
        auth_token = ""

    # NOTE: Now default points to ./database/orqestra.db
    db_path = questionary.text(
        "SQLite DB path:",
        default=existing.get("db.path", DEFAULT_DB),
    ).ask()
    if db_path is None:
        raise SystemExit(130)

    # Save
    set_setting(conn, "service.name", service_name.strip())
    set_setting(conn, "api.base", api_base.strip())
    set_setting(conn, "log.level", log_level.strip())
    set_setting(conn, "db.path", db_path.strip())

    set_setting(conn, "auth.enabled", "true" if auth_enabled else "false")
    set_setting(conn, "auth.token", auth_token.strip())

    set_setting(conn, "metrics.enabled", "true" if metrics_enabled else "false")
    set_setting(conn, "autoupdate.enabled", "true" if autoupdate_enabled else "false")

    questionary.print("\n✅ Saved. Current settings:\n", style="bold")
    for k, v in get_all_settings(conn).items():
        print(f"{k} = {v}")
    print("")


# ---------- CLI ----------
def main() -> int:
    ap = argparse.ArgumentParser(prog="orqestra.py")
    ap.add_argument("--onboard", action="store_true", help="Run interactive onboarding")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite DB path (default: {DEFAULT_DB})")
    ap.add_argument("--show", action="store_true", help="Show all saved settings")
    args = ap.parse_args()

    conn = db_connect(args.db)
    try:
        db_init(conn)

        if args.onboard:
            onboard(conn)
            return 0

        if args.show:
            for k, v in get_all_settings(conn).items():
                print(f"{k} = {v}")
            return 0

        ap.print_help()
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
