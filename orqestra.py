#!/usr/bin/env python3
import argparse
import base64
import os
import sqlite3
from typing import Dict, Optional

import questionary
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Store DB under ./database/
DEFAULT_DB_DIR = os.environ.get("ORQESTRA_DB_DIR", "./database")
DEFAULT_DB = os.environ.get("ORQESTRA_DB", os.path.join(DEFAULT_DB_DIR, "orqestra.db"))
MASTER_KEY_FILE = os.path.join(DEFAULT_DB_DIR, "master.key")


# ---------- FS HELPERS ----------
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ---------- CRYPTO HELPERS ----------
def load_or_create_master_key() -> bytes:
    """
    32 bytes key for AES-256-GCM.
    Priority:
      1) ORQESTRA_MASTER_KEY (base64)
      2) ./database/master.key (base64)
      3) create new and store to file
    """
    env = os.environ.get("ORQESTRA_MASTER_KEY")
    if env:
        key = base64.b64decode(env.encode("utf-8"))
        if len(key) != 32:
            raise ValueError("ORQESTRA_MASTER_KEY must be base64 of 32 bytes.")
        return key

    ensure_parent_dir(MASTER_KEY_FILE)
    if os.path.exists(MASTER_KEY_FILE):
        raw = open(MASTER_KEY_FILE, "rb").read().strip()
        key = base64.b64decode(raw)
        if len(key) != 32:
            raise ValueError("master.key must be base64 of 32 bytes.")
        return key

    key = os.urandom(32)
    with open(MASTER_KEY_FILE, "wb") as f:
        f.write(base64.b64encode(key))
    return key


def encrypt_secret(master_key: bytes, plaintext: str) -> str:
    aes = AESGCM(master_key)
    nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
    ct = aes.encrypt(nonce, plaintext.encode("utf-8"), None)
    blob = nonce + ct
    return "enc:" + base64.b64encode(blob).decode("utf-8")


def decrypt_secret(master_key: bytes, value: str) -> str:
    if not value.startswith("enc:"):
        return value
    blob = base64.b64decode(value[4:].encode("utf-8"))
    nonce, ct = blob[:12], blob[12:]
    aes = AESGCM(master_key)
    pt = aes.decrypt(nonce, ct, None)
    return pt.decode("utf-8")


# ---------- DB ----------
def db_connect(path: str) -> sqlite3.Connection:
    ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
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
    master_key = load_or_create_master_key()

    questionary.print("\n=== Orqestra Onboarding ===\n", style="bold")

    # Basic
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

    # Features
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

    auth_token = ""
    if auth_enabled:
        auth_token = questionary.password("Auth token (hidden):").ask() or ""
        if not auth_token.strip():
            questionary.print("Auth token is required when authentication is enabled.", style="bold fg:red")
            raise SystemExit(2)

    # Providers (OpenClaw-like)
    providers = questionary.checkbox(
        "AI providers to enable (Space to toggle, Enter to confirm):",
        choices=[
            questionary.Choice(
                "OpenAI",
                checked=existing.get("provider.openai.enabled", "false") == "true",
            ),
            questionary.Choice(
                "Anthropic (Claude)",
                checked=existing.get("provider.anthropic.enabled", "false") == "true",
            ),
            questionary.Choice(
                "Google (Gemini)",
                checked=existing.get("provider.google.enabled", "false") == "true",
            ),
            questionary.Choice(
                "Local (Ollama)",
                checked=existing.get("provider.ollama.enabled", "false") == "true",
            ),
        ],
    ).ask()
    if providers is None:
        raise SystemExit(130)

    enable_openai = "OpenAI" in providers
    enable_anthropic = "Anthropic (Claude)" in providers
    enable_google = "Google (Gemini)" in providers
    enable_ollama = "Local (Ollama)" in providers

    # Token prompts (only if enabled)
    openai_key = ""
    anthropic_key = ""
    google_key = ""
    ollama_base = existing.get("provider.ollama.base_url", "http://127.0.0.1:11434")

    if enable_openai:
        openai_key = questionary.password("OpenAI API key (hidden):").ask() or ""
        if not openai_key.strip():
            questionary.print("OpenAI API key is required when OpenAI is enabled.", style="bold fg:red")
            raise SystemExit(2)

    if enable_anthropic:
        anthropic_key = questionary.password("Anthropic API key (hidden):").ask() or ""
        if not anthropic_key.strip():
            questionary.print("Anthropic API key is required when Anthropic is enabled.", style="bold fg:red")
            raise SystemExit(2)

    if enable_google:
        google_key = questionary.password("Google API key (hidden):").ask() or ""
        if not google_key.strip():
            questionary.print("Google API key is required when Google is enabled.", style="bold fg:red")
            raise SystemExit(2)

    if enable_ollama:
        ollama_base = questionary.text("Ollama base URL:", default=ollama_base).ask() or ollama_base

    # DB path (still configurable, default in ./database)
    db_path = questionary.text(
        "SQLite DB path:",
        default=existing.get("db.path", DEFAULT_DB),
    ).ask()
    if db_path is None:
        raise SystemExit(130)

    # --- Save basic ---
    set_setting(conn, "service.name", service_name.strip())
    set_setting(conn, "api.base", api_base.strip())
    set_setting(conn, "log.level", log_level.strip())
    set_setting(conn, "db.path", db_path.strip())

    # --- Save feature flags + auth secret ---
    set_setting(conn, "auth.enabled", "true" if auth_enabled else "false")
    set_setting(conn, "metrics.enabled", "true" if metrics_enabled else "false")
    set_setting(conn, "autoupdate.enabled", "true" if autoupdate_enabled else "false")

    set_setting(conn, "secrets.auth.token", encrypt_secret(master_key, auth_token.strip()) if auth_enabled else "")

    # --- Save provider flags ---
    set_setting(conn, "provider.openai.enabled", "true" if enable_openai else "false")
    set_setting(conn, "provider.anthropic.enabled", "true" if enable_anthropic else "false")
    set_setting(conn, "provider.google.enabled", "true" if enable_google else "false")
    set_setting(conn, "provider.ollama.enabled", "true" if enable_ollama else "false")
    set_setting(conn, "provider.ollama.base_url", ollama_base.strip())

    # --- Save encrypted provider secrets ---
    set_setting(conn, "secrets.openai.api_key", encrypt_secret(master_key, openai_key.strip()) if enable_openai else "")
    set_setting(
        conn,
        "secrets.anthropic.api_key",
        encrypt_secret(master_key, anthropic_key.strip()) if enable_anthropic else "",
    )
    set_setting(conn, "secrets.google.api_key", encrypt_secret(master_key, google_key.strip()) if enable_google else "")

    questionary.print("\n✅ Saved. Current settings (secrets hidden):\n", style="bold")
    all_settings = get_all_settings(conn)
    for k, v in all_settings.items():
        if k.startswith("secrets."):
            print(f"{k} = (hidden)")
        else:
            print(f"{k} = {v}")
    print("")


# ---------- CLI ----------
def main() -> int:
    ap = argparse.ArgumentParser(prog="orqestra.py")
    ap.add_argument("--onboard", action="store_true", help="Run interactive onboarding")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite DB path (default: {DEFAULT_DB})")
    ap.add_argument("--show", action="store_true", help="Show all saved settings (secrets hidden)")
    ap.add_argument("--show-secrets", action="store_true", help="Show decrypted secrets (requires master key)")
    args = ap.parse_args()

    conn = db_connect(args.db)
    try:
        db_init(conn)

        if args.onboard:
            onboard(conn)
            return 0

        if args.show or args.show_secrets:
            settings = get_all_settings(conn)
            if args.show_secrets:
                master_key = load_or_create_master_key()
                for k, v in settings.items():
                    if k.startswith("secrets.") and v:
                        print(f"{k} = {decrypt_secret(master_key, v)}")
                    else:
                        print(f"{k} = {v}")
            else:
                for k, v in settings.items():
                    if k.startswith("secrets."):
                        print(f"{k} = (hidden)")
                    else:
                        print(f"{k} = {v}")
            return 0

        ap.print_help()
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())