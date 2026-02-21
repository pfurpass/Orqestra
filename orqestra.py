#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sqlite3
from typing import Dict, Optional, List, Tuple

import questionary
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# -------------------------
# Paths / Defaults
# -------------------------
DEFAULT_DB_DIR = os.environ.get("ORQESTRA_DB_DIR", "./database")
DEFAULT_DB = os.environ.get("ORQESTRA_DB", os.path.join(DEFAULT_DB_DIR, "orqestra.db"))
MASTER_KEY_FILE = os.path.join(DEFAULT_DB_DIR, "master.key")

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Gemini: use the official "models/..." names
DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash"

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

# Gemini models list (arrow-key select)
GEMINI_MODEL_CHOICES = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemini-pro-latest",
]


# -------------------------
# FS helpers
# -------------------------
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# -------------------------
# Crypto helpers (AES-256-GCM)
# -------------------------
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
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext.encode("utf-8"), None)
    return "enc:" + base64.b64encode(nonce + ct).decode("utf-8")


def decrypt_secret(master_key: bytes, value: str) -> str:
    if not value.startswith("enc:"):
        return value
    blob = base64.b64decode(value[4:].encode("utf-8"))
    nonce, ct = blob[:12], blob[12:]
    aes = AESGCM(master_key)
    return aes.decrypt(nonce, ct, None).decode("utf-8")


# -------------------------
# DB
# -------------------------
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


def get_all_settings(conn: sqlite3.Connection) -> Dict[str, str]:
    cur = conn.execute("SELECT key, value FROM settings ORDER BY key;")
    return {k: v for (k, v) in cur.fetchall()}


# -------------------------
# Provider calls
# -------------------------
class ProviderError(Exception):
    pass


def call_openai(api_key: str, message: str, model: str, base_url: str, timeout: int = 60) -> str:
    url = base_url.rstrip("/") + "/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": message}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise ProviderError(f"OpenAI error {r.status_code}: {r.text}")
    data = r.json()
    try:
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text" and "text" in c:
                    return c["text"]
    except Exception:
        pass
    return json.dumps(data, indent=2)


def call_anthropic(api_key: str, message: str, model: str, timeout: int = 60) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {"model": model, "max_tokens": 1024, "messages": [{"role": "user", "content": message}]}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise ProviderError(f"Anthropic error {r.status_code}: {r.text}")
    data = r.json()
    parts = data.get("content", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
    texts = [t for t in texts if t]
    return "\n".join(texts) if texts else json.dumps(data, indent=2)


def call_gemini(api_key: str, message: str, model: str, timeout: int = 60) -> str:
    # model MUST be "models/...."
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": message}]}]}
    r = requests.post(url, params=params, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise ProviderError(f"Gemini error {r.status_code}: {r.text}")
    data = r.json()
    try:
        cand = (data.get("candidates") or [None])[0] or {}
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        texts = [t for t in texts if t]
        if texts:
            return "\n".join(texts)
    except Exception:
        pass
    return json.dumps(data, indent=2)


def call_ollama(base_url: str, message: str, model: str, timeout: int = 60) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "prompt": message, "stream": False}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise ProviderError(f"Ollama error {r.status_code}: {r.text}")
    data = r.json()
    return str(data["response"]) if "response" in data else json.dumps(data, indent=2)


# -------------------------
# Settings helpers
# -------------------------
def enabled_providers(settings: Dict[str, str]) -> List[str]:
    provs: List[str] = []
    if settings.get("provider.openai.enabled") == "true":
        provs.append("openai")
    if settings.get("provider.anthropic.enabled") == "true":
        provs.append("anthropic")
    if settings.get("provider.gemini.enabled") == "true" or settings.get("provider.google.enabled") == "true":
        provs.append("gemini")
    if settings.get("provider.ollama.enabled") == "true":
        provs.append("ollama")
    return provs


def read_secret(settings: Dict[str, str], master_key: bytes, key: str, fallback_key: Optional[str] = None) -> str:
    v = settings.get(key, "")
    if not v and fallback_key:
        v = settings.get(fallback_key, "")
    return decrypt_secret(master_key, v) if v else ""


# -------------------------
# Message runner
# -------------------------
def run_message(conn: sqlite3.Connection, message: str, provider_filter: Optional[str], timeout: int) -> int:
    settings = get_all_settings(conn)
    provs = enabled_providers(settings)

    if provider_filter:
        pf = provider_filter.strip().lower()
        aliases = {
            "openai": "openai",
            "anthropic": "anthropic",
            "claude": "anthropic",
            "gemini": "gemini",
            "google": "gemini",
            "ollama": "ollama",
            "local": "ollama",
        }
        pf = aliases.get(pf, pf)
        provs = [p for p in provs if p == pf]

    if not provs:
        print("No providers enabled (or filter removed all). Run: python3 orqestra.py --onboard")
        return 2

    master_key = load_or_create_master_key()

    results: List[Tuple[str, str]] = []
    failures: List[Tuple[str, str]] = []

    for p in provs:
        try:
            if p == "openai":
                api_key = read_secret(settings, master_key, "secrets.openai.api_key")
                if not api_key:
                    raise ProviderError("Missing OpenAI API key (secrets.openai.api_key).")
                model = settings.get("provider.openai.model", DEFAULT_OPENAI_MODEL)
                base_url = settings.get("provider.openai.base_url", DEFAULT_OPENAI_BASE_URL)
                out = call_openai(api_key, message, model=model, base_url=base_url, timeout=timeout)
                results.append(("OpenAI", out))

            elif p == "anthropic":
                api_key = read_secret(settings, master_key, "secrets.anthropic.api_key")
                if not api_key:
                    raise ProviderError("Missing Anthropic API key (secrets.anthropic.api_key).")
                model = settings.get("provider.anthropic.model", DEFAULT_ANTHROPIC_MODEL)
                out = call_anthropic(api_key, message, model=model, timeout=timeout)
                results.append(("Anthropic", out))

            elif p == "gemini":
                api_key = read_secret(settings, master_key, "secrets.gemini.api_key", fallback_key="secrets.google.api_key")
                if not api_key:
                    raise ProviderError("Missing Gemini API key (secrets.gemini.api_key).")
                model = settings.get("provider.gemini.model", DEFAULT_GEMINI_MODEL)
                out = call_gemini(api_key, message, model=model, timeout=timeout)
                results.append(("Gemini", out))

            elif p == "ollama":
                base_url = settings.get("provider.ollama.base_url", DEFAULT_OLLAMA_BASE)
                model = settings.get("provider.ollama.model", DEFAULT_OLLAMA_MODEL)
                out = call_ollama(base_url, message, model=model, timeout=timeout)
                results.append(("Ollama", out))

            else:
                raise ProviderError(f"Unknown provider '{p}'")

        except Exception as e:
            failures.append((p, str(e)))

    for name, out in results:
        print(f"\n=== {name} ===\n{out}\n")

    if failures:
        print("\n--- Errors ---")
        for p, err in failures:
            print(f"- {p}: {err}")
        return 3 if not results else 1

    return 0


# -------------------------
# Onboarding (Gemini model via arrow keys ONLY)
# -------------------------
def onboard(conn: sqlite3.Connection) -> None:
    existing = get_all_settings(conn)
    master_key = load_or_create_master_key()

    questionary.print("\n=== Orqestra Onboarding ===\n", style="bold")

    service_name = questionary.text("Service name:", default=existing.get("service.name", "orqestra")).ask()
    if service_name is None:
        raise SystemExit(130)

    api_base = questionary.text("API base URL:", default=existing.get("api.base", "http://127.0.0.1:8788")).ask()
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
            questionary.Choice("Enable authentication", checked=existing.get("auth.enabled", "false") == "true"),
            questionary.Choice("Enable metrics", checked=existing.get("metrics.enabled", "false") == "true"),
            questionary.Choice("Enable auto-update", checked=existing.get("autoupdate.enabled", "false") == "true"),
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

    providers = questionary.checkbox(
        "AI providers to enable (Space to toggle, Enter to confirm):",
        choices=[
            questionary.Choice("OpenAI", checked=existing.get("provider.openai.enabled", "false") == "true"),
            questionary.Choice("Anthropic (Claude)", checked=existing.get("provider.anthropic.enabled", "false") == "true"),
            questionary.Choice("Gemini", checked=(existing.get("provider.gemini.enabled", "false") == "true" or existing.get("provider.google.enabled", "false") == "true")),
            questionary.Choice("Ollama (local)", checked=existing.get("provider.ollama.enabled", "false") == "true"),
        ],
    ).ask()
    if providers is None:
        raise SystemExit(130)

    enable_openai = "OpenAI" in providers
    enable_anthropic = "Anthropic (Claude)" in providers
    enable_gemini = "Gemini" in providers
    enable_ollama = "Ollama (local)" in providers

    # Provider configs
    openai_key = ""
    anthropic_key = ""
    gemini_key = ""

    openai_base_url = existing.get("provider.openai.base_url", DEFAULT_OPENAI_BASE_URL)
    openai_model = existing.get("provider.openai.model", DEFAULT_OPENAI_MODEL)
    anthropic_model = existing.get("provider.anthropic.model", DEFAULT_ANTHROPIC_MODEL)

    gemini_model_existing = existing.get("provider.gemini.model", DEFAULT_GEMINI_MODEL)
    if gemini_model_existing not in GEMINI_MODEL_CHOICES:
        # if it was custom before, fall back to default list
        gemini_model_existing = DEFAULT_GEMINI_MODEL if DEFAULT_GEMINI_MODEL in GEMINI_MODEL_CHOICES else GEMINI_MODEL_CHOICES[0]

    ollama_base = existing.get("provider.ollama.base_url", DEFAULT_OLLAMA_BASE)
    ollama_model = existing.get("provider.ollama.model", DEFAULT_OLLAMA_MODEL)

    if enable_openai:
        openai_base_url = questionary.text("OpenAI base URL:", default=openai_base_url).ask() or openai_base_url
        openai_model = questionary.text("OpenAI model:", default=openai_model).ask() or openai_model
        openai_key = questionary.password("OpenAI API key (hidden):").ask() or ""
        if not openai_key.strip():
            questionary.print("OpenAI API key is required when OpenAI is enabled.", style="bold fg:red")
            raise SystemExit(2)

    if enable_anthropic:
        anthropic_model = questionary.text("Anthropic model:", default=anthropic_model).ask() or anthropic_model
        anthropic_key = questionary.password("Anthropic API key (hidden):").ask() or ""
        if not anthropic_key.strip():
            questionary.print("Anthropic API key is required when Anthropic is enabled.", style="bold fg:red")
            raise SystemExit(2)

    if enable_gemini:
        # ONLY arrow-key selection, no free text
        gemini_model = questionary.select(
            "Gemini model (use arrow keys):",
            choices=GEMINI_MODEL_CHOICES,
            default=gemini_model_existing,
        ).ask()
        if gemini_model is None:
            raise SystemExit(130)

        gemini_key = questionary.password("Gemini API key (hidden):").ask() or ""
        if not gemini_key.strip():
            questionary.print("Gemini API key is required when Gemini is enabled.", style="bold fg:red")
            raise SystemExit(2)

    if enable_ollama:
        ollama_base = questionary.text("Ollama base URL:", default=ollama_base).ask() or ollama_base
        ollama_model = questionary.text("Ollama model:", default=ollama_model).ask() or ollama_model

    db_path = questionary.text("SQLite DB path:", default=existing.get("db.path", DEFAULT_DB)).ask()
    if db_path is None:
        raise SystemExit(130)

    # Save basic
    set_setting(conn, "service.name", service_name.strip())
    set_setting(conn, "api.base", api_base.strip())
    set_setting(conn, "log.level", log_level.strip())
    set_setting(conn, "db.path", db_path.strip())

    # Save features + auth secret
    set_setting(conn, "auth.enabled", "true" if auth_enabled else "false")
    set_setting(conn, "metrics.enabled", "true" if metrics_enabled else "false")
    set_setting(conn, "autoupdate.enabled", "true" if autoupdate_enabled else "false")
    set_setting(conn, "secrets.auth.token", encrypt_secret(master_key, auth_token.strip()) if auth_enabled else "")

    # Save provider flags
    set_setting(conn, "provider.openai.enabled", "true" if enable_openai else "false")
    set_setting(conn, "provider.anthropic.enabled", "true" if enable_anthropic else "false")
    set_setting(conn, "provider.gemini.enabled", "true" if enable_gemini else "false")
    set_setting(conn, "provider.ollama.enabled", "true" if enable_ollama else "false")

    # Save provider config
    set_setting(conn, "provider.openai.base_url", openai_base_url.strip())
    set_setting(conn, "provider.openai.model", openai_model.strip())
    set_setting(conn, "provider.anthropic.model", anthropic_model.strip())
    if enable_gemini:
        set_setting(conn, "provider.gemini.model", gemini_model.strip())
    set_setting(conn, "provider.ollama.base_url", ollama_base.strip())
    set_setting(conn, "provider.ollama.model", ollama_model.strip())

    # Save secrets
    set_setting(conn, "secrets.openai.api_key", encrypt_secret(master_key, openai_key.strip()) if enable_openai else "")
    set_setting(conn, "secrets.anthropic.api_key", encrypt_secret(master_key, anthropic_key.strip()) if enable_anthropic else "")
    set_setting(conn, "secrets.gemini.api_key", encrypt_secret(master_key, gemini_key.strip()) if enable_gemini else "")

    # Cleanup old keys
    set_setting(conn, "provider.google.enabled", "false")
    set_setting(conn, "secrets.google.api_key", "")

    questionary.print("\n✅ Saved. Current settings (secrets hidden):\n", style="bold")
    for k, v in get_all_settings(conn).items():
        print(f"{k} = (hidden)" if k.startswith("secrets.") else f"{k} = {v}")
    print("")


# -------------------------
# CLI
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser(prog="orqestra.py")
    ap.add_argument("--onboard", action="store_true", help="Run interactive onboarding")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite DB path (default: {DEFAULT_DB})")
    ap.add_argument("--message", type=str, help='Send a message to your agent, e.g. --message "Hello"')
    ap.add_argument("--provider", type=str, default=None, help="Filter: openai|anthropic|gemini|ollama")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--show", action="store_true", help="Show all saved settings (secrets hidden)")
    ap.add_argument("--show-secrets", action="store_true", help="Show decrypted secrets (requires master key)")
    args = ap.parse_args()

    conn = db_connect(args.db)
    try:
        db_init(conn)

        if args.onboard:
            onboard(conn)
            return 0

        if args.message is not None:
            return run_message(conn, args.message, provider_filter=args.provider, timeout=args.timeout)

        if args.show or args.show_secrets:
            settings = get_all_settings(conn)
            if args.show_secrets:
                mk = load_or_create_master_key()
                for k, v in settings.items():
                    if k.startswith("secrets.") and v:
                        print(f"{k} = {decrypt_secret(mk, v)}")
                    else:
                        print(f"{k} = {v}")
            else:
                for k, v in settings.items():
                    print(f"{k} = (hidden)" if k.startswith("secrets.") else f"{k} = {v}")
            return 0

        ap.print_help()
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())