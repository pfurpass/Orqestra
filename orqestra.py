#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import sqlite3
import sys
import threading
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import questionary
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# -------------------------
# Paths / Defaults
# -------------------------
DEFAULT_DB_DIR = os.environ.get("ORQESTRA_DB_DIR", "./database")
DEFAULT_DB = os.environ.get("ORQESTRA_DB", os.path.join(DEFAULT_DB_DIR, "orqestra.db"))
MASTER_KEY_FILE = os.path.join(DEFAULT_DB_DIR, "master.key")

WORKSPACE_ROOT = Path(os.environ.get("ORQESTRA_WORKSPACE", "./workspace")).resolve()

# -------------------------
# Provider defaults
# -------------------------
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash"

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

# -------------------------
# Arrow-key selectable model lists (remote providers)
# -------------------------
OPENAI_MODEL_CHOICES = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "o4-mini",
    "o3-mini",
]

ANTHROPIC_MODEL_CHOICES = [
    "claude-sonnet-4-6",
    "claude-opus-4-1",
    "claude-haiku-3-5",
]

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
# Spinner / "waiting" animation
# -------------------------
class Spinner:
    def __init__(self, text: str = "Waiting for the agent"):
        self.text = text
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()

        def run():
            i = 0
            while not self._stop.is_set():
                frame = self._frames[i % len(self._frames)]
                sys.stdout.write(f"\r{frame} {self.text}...")
                sys.stdout.flush()
                i += 1
                time.sleep(0.1)
            # clear line
            sys.stdout.write("\r" + (" " * (len(self.text) + 10)) + "\r")
            sys.stdout.flush()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


# -------------------------
# FS helpers
# -------------------------
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def safe_join(base: Path, rel: str) -> Path:
    # Prevent ../../ escape
    p = (base / rel).resolve()
    if not str(p).startswith(str(base)):
        raise ValueError(f"Refusing to access outside workspace: {rel}")
    return p


# -------------------------
# Crypto helpers (AES-256-GCM)
# -------------------------
def load_or_create_master_key() -> bytes:
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
# Provider calls + model listing
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
    payload = {"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": message}]}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise ProviderError(f"Anthropic error {r.status_code}: {r.text}")
    data = r.json()
    parts = data.get("content", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
    texts = [t for t in texts if t]
    return "\n".join(texts) if texts else json.dumps(data, indent=2)


def call_gemini(api_key: str, message: str, model: str, timeout: int = 60, force_json: bool = False) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": message}]}],
        "generationConfig": {
            "maxOutputTokens": 4096,
        },
    }

    if force_json:
        payload["generationConfig"]["response_mime_type"] = "application/json"

    r = requests.post(url, params=params, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise ProviderError(f"Gemini error {r.status_code}: {r.text}")

    data = r.json()
    try:
        cand = (data.get("candidates") or [None])[0] or {}
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        texts = []
        for p in parts:
            if isinstance(p, dict):
                t = p.get("text", "")
                if t:
                    texts.append(t)
        if texts:
            return "\n".join(texts).strip()
    except Exception:
        pass

    return json.dumps(data, indent=2)


def ollama_list_models(base_url: str, timeout: int = 10) -> List[str]:
    url = base_url.rstrip("/") + "/api/tags"
    r = requests.get(url, timeout=timeout)
    if r.status_code >= 400:
        raise ProviderError(f"Ollama /api/tags error {r.status_code}: {r.text}")
    data = r.json()
    models = []
    for m in data.get("models", []):
        name = m.get("name")
        if name:
            models.append(name)
    return sorted(set(models))


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


def provider_alias(p: str) -> str:
    aliases = {
        "openai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "gemini": "gemini",
        "google": "gemini",
        "ollama": "ollama",
        "local": "ollama",
    }
    return aliases.get(p.strip().lower(), p.strip().lower())


def call_provider_for_text(
    settings: Dict[str, str],
    master_key: bytes,
    provider: str,
    message: str,
    model_override: Optional[str],
    timeout: int,
    force_json: bool = False,
) -> str:
    p = provider_alias(provider)

    if p == "openai":
        api_key = read_secret(settings, master_key, "secrets.openai.api_key")
        if not api_key:
            raise ProviderError("Missing OpenAI API key (secrets.openai.api_key).")
        model = model_override or settings.get("provider.openai.model", DEFAULT_OPENAI_MODEL)
        base_url = settings.get("provider.openai.base_url", DEFAULT_OPENAI_BASE_URL)
        return call_openai(api_key, message, model=model, base_url=base_url, timeout=timeout)

    if p == "anthropic":
        api_key = read_secret(settings, master_key, "secrets.anthropic.api_key")
        if not api_key:
            raise ProviderError("Missing Anthropic API key (secrets.anthropic.api_key).")
        model = model_override or settings.get("provider.anthropic.model", DEFAULT_ANTHROPIC_MODEL)
        return call_anthropic(api_key, message, model=model, timeout=timeout)

    if p == "gemini":
        api_key = read_secret(settings, master_key, "secrets.gemini.api_key", fallback_key="secrets.google.api_key")
        if not api_key:
            raise ProviderError("Missing Gemini API key (secrets.gemini.api_key).")
        model = model_override or settings.get("provider.gemini.model", DEFAULT_GEMINI_MODEL)
        return call_gemini(api_key, message, model=model, timeout=timeout, force_json=force_json)

    if p == "ollama":
        base_url = settings.get("provider.ollama.base_url", DEFAULT_OLLAMA_BASE)
        model = model_override or settings.get("provider.ollama.model", DEFAULT_OLLAMA_MODEL)
        return call_ollama(base_url, message, model=model, timeout=timeout)

    raise ProviderError(f"Unknown provider: {provider}")


# -------------------------
# Message runner (spinner per provider + optional global spinner)
# -------------------------
def run_message(conn: sqlite3.Connection, message: str, provider_filter: Optional[str], model_override: Optional[str], timeout: int) -> int:
    settings = get_all_settings(conn)
    provs = enabled_providers(settings)

    if provider_filter:
        pf = provider_alias(provider_filter)
        provs = [p for p in provs if p == pf]

    if not provs:
        print("No providers enabled (or filter removed all). Run: python3 orqestra.py --onboard")
        return 2

    master_key = load_or_create_master_key()

    results: List[Tuple[str, str]] = []
    failures: List[Tuple[str, str]] = []

    global_spin = Spinner("Waiting for the agent")
    global_spin.start()

    try:
        for p in provs:
            try:
                global_spin.stop()
                print(f"\n=== {p.upper()} ===")
                spin = Spinner(f"Waiting for {p}")
                spin.start()

                out = call_provider_for_text(settings, master_key, p, message, model_override, timeout)

                results.append((p.capitalize(), out))
                spin.stop()
                print("Done.\n")

            except Exception as e:
                try:
                    spin.stop()
                except Exception:
                    pass
                failures.append((p, str(e)))
            finally:
                if p != provs[-1]:
                    global_spin.start()

    finally:
        global_spin.stop()

    for name, out in results:
        print(f"\n--- {name} response ---\n{out}\n")

    if failures:
        print("\n--- Errors ---")
        for p, err in failures:
            print(f"- {p}: {err}")
        return 3 if not results else 1

    return 0


# -------------------------
# Agent mode (writes files + runs commands + fixes until OK)
# -------------------------
AGENT_JSON_SCHEMA = """
Return ONLY valid JSON:

{
  "plan": ["..."],
  "files": [
    {"path": "relative/path", "content": "full file content"},
    ...
  ],
  "commands": [
    ["cmd", "arg1", "arg2"],
    ...
  ],
  "run_instructions": "How the user runs/tests it",
  "notes": "short notes"
}

Rules:
- paths MUST be relative, no absolute paths, no ".."
- put ALL code in files (not in notes)
- commands should be runnable in the workspace directory
"""

# still block the most destructive stuff (even in wide mode)
AGENT_BLOCK_PATTERNS = [
    r"\brm\b\s+-rf\s+/\b",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bshutdown\b|\breboot\b",
]

# default “safe-ish” allowlist (can be disabled via --agent-wide)
AGENT_ALLOWED = {
    "python", "python3",
    "pip", "pip3",
    "npm", "node",
    "pnpm", "yarn",
    "npx",
    "pytest",
    "bash", "sh",
    "git",
    "docker",
    "docker-compose",
    "make",
}


def _sanitize_task_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")[:60]
    return safe or "task"


def agent_make_workspace(task: str) -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ws = (WORKSPACE_ROOT / f"{stamp}-{_sanitize_task_name(task)}").resolve()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def agent_tree(ws: Path, max_entries: int = 250) -> str:
    lines: List[str] = []
    count = 0
    for root, dirs, files in os.walk(ws):
        rootp = Path(root)
        relroot = rootp.relative_to(ws)
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", "__pycache__", ".pytest_cache"}]
        for f in files:
            p = relroot / f
            lines.append(str(p))
            count += 1
            if count >= max_entries:
                lines.append("... (truncated)")
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(empty)"


def agent_parse_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()

    if not raw:
        raise ValueError("Agent returned empty response (no JSON).")

    # Remove common code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    raw = re.sub(r"\s*```$", "", raw.strip())

    # First try direct JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Salvage JSON object from within text
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    preview = raw[:400].replace("\n", "\\n")
    raise ValueError(f"Agent did not return valid JSON. Preview: {preview}")


def agent_write_files(ws: Path, files: List[Dict[str, str]]) -> None:
    for f in files:
        path = (f.get("path") or "").strip()
        content = f.get("content")
        if content is None:
            content = ""
        if not path or ".." in path or path.startswith("/") or path.startswith("\\"):
            raise ValueError(f"Unsafe path from agent: {path}")
        p = safe_join(ws, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(content), encoding="utf-8")


def _shlex_join(cmd: List[str]) -> str:
    def q(s: str) -> str:
        if re.search(r"\s|\"", s):
            return '"' + s.replace('"', '\\"') + '"'
        return s
    return " ".join(q(c) for c in cmd)


def agent_cmd_allowed(cmd: List[str], wide: bool) -> Tuple[bool, str]:
    if not cmd:
        return False, "Empty command"
    raw = " ".join(cmd)
    for pat in AGENT_BLOCK_PATTERNS:
        if re.search(pat, raw):
            return False, f"Blocked pattern: {pat}"
    if wide:
        return True, ""
    if cmd[0] not in AGENT_ALLOWED:
        return False, f"Not allowed in safe mode: {cmd[0]} (use --agent-wide)"
    return True, ""


def agent_run_cmd(ws: Path, cmd: List[str], timeout_s: int, approve: bool, wide: bool) -> Tuple[bool, int, str, str]:
    ok, reason = agent_cmd_allowed(cmd, wide=wide)
    if not ok:
        return False, 127, "", f"Blocked: {reason}"

    if approve:
        print(f"\nAbout to run:\n  {_shlex_join(cmd)}\n")
        ans = input("Run this command? [y/N]: ").strip().lower()
        if ans != "y":
            return False, 126, "", "User declined command"

    p = subprocess.run(
        cmd,
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return p.returncode == 0, p.returncode, p.stdout, p.stderr


def build_agent_prompt(task: str, ws: Path, last_logs: str) -> str:
    tree = agent_tree(ws)
    return f"""
You are Orqestra Agent: you create code, write files, and provide shell commands to build/run/test.

{AGENT_JSON_SCHEMA}

Task:
{task}

Workspace tree:
{tree}

Last execution logs (if any):
{last_logs}
""".strip()


def run_agent(
    conn: sqlite3.Connection,
    task: str,
    provider: Optional[str],
    model_override: Optional[str],
    timeout: int,
    iters: int,
    approve: bool,
    wide: bool,
    cmd_timeout: int,
) -> int:
    settings = get_all_settings(conn)
    provs = enabled_providers(settings)
    if not provs:
        print("No providers enabled. Run: python3 orqestra.py --onboard")
        return 2

    # pick provider: explicit --provider or first enabled
    chosen = provider_alias(provider) if provider else provs[0]
    if chosen not in provs:
        print(f"Provider '{chosen}' not enabled. Enabled: {', '.join(provs)}")
        return 2

    master_key = load_or_create_master_key()

    ws = agent_make_workspace(task)
    print(f"\n[Agent] Workspace: {ws}")

    last_logs = ""
    final_instructions = ""

    for i in range(1, iters + 1):
        print(f"\n[Agent] Iteration {i}/{iters}")
        spin = Spinner("Waiting for the agent")
        spin.start()
        try:
            prompt = build_agent_prompt(task, ws, last_logs)

            # Force JSON for Gemini (safe for others; ignored)
            raw = call_provider_for_text(
                settings, master_key, chosen, prompt, model_override, timeout,
                force_json=True
            )

            # Retry once if empty
            if not (raw or "").strip():
                retry_prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown. No explanations."
                raw = call_provider_for_text(
                    settings, master_key, chosen, retry_prompt, model_override, timeout,
                    force_json=True
                )
        finally:
            spin.stop()

        data = agent_parse_json(raw)

        plan = data.get("plan") or []
        files = data.get("files") or []
        commands = data.get("commands") or []
        final_instructions = (data.get("run_instructions") or "").strip()

        if plan:
            print("\n[Plan]")
            for p in plan:
                print(f"- {p}")

        if files:
            print(f"\n[Agent] Writing {len(files)} file(s)...")
            agent_write_files(ws, files)

        if not commands:
            print("\n[Agent] No commands provided. Stopping.")
            break

        print(f"\n[Agent] Running {len(commands)} command(s)...")
        logs_parts: List[str] = []
        all_ok = True

        for idx, c in enumerate(commands, start=1):
            if not isinstance(c, list) or not all(isinstance(x, str) for x in c):
                all_ok = False
                logs_parts.append(f"[cmd {idx}] INVALID COMMAND: {c}")
                break

            print(f"  -> {_shlex_join(c)}")
            run_spin = Spinner("Executing")
            run_spin.start()
            try:
                ok, rc, out, err = agent_run_cmd(ws, c, timeout_s=cmd_timeout, approve=approve, wide=wide)
            except subprocess.TimeoutExpired:
                run_spin.stop()
                ok, rc, out, err = False, 124, "", f"Command timed out after {cmd_timeout}s"
            finally:
                run_spin.stop()

            status = "OK" if ok else "FAIL"
            print(f"     [{status}] exit={rc}")

            logs_parts.append(
                "\n".join(
                    [
                        f"[cmd {idx}] {_shlex_join(c)}",
                        f"exit={rc}",
                        "stdout:",
                        out[-4000:] if out else "",
                        "stderr:",
                        err[-4000:] if err else "",
                        "-" * 40,
                    ]
                )
            )

            if not ok:
                all_ok = False
                break

        last_logs = "\n".join(logs_parts)

        if all_ok:
            print("\n[Agent] ✅ All commands succeeded.")
            break
        else:
            print("\n[Agent] ❌ Failure detected, asking agent to fix...")

    print("\n=========================")
    print("AGENT DONE")
    print("=========================")
    print(f"Workspace: {ws}")
    if final_instructions:
        print("\nRun instructions:\n")
        print(final_instructions)
    else:
        print("\nNo run instructions returned. Check workspace and logs above.")
    return 0


# -------------------------
# Onboarding (Ollama models fetched from API, all models via arrow keys)
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
            questionary.Choice(
                "Gemini",
                checked=(existing.get("provider.gemini.enabled", "false") == "true" or existing.get("provider.google.enabled", "false") == "true"),
            ),
            questionary.Choice("Ollama (local)", checked=existing.get("provider.ollama.enabled", "false") == "true"),
        ],
    ).ask()
    if providers is None:
        raise SystemExit(130)

    enable_openai = "OpenAI" in providers
    enable_anthropic = "Anthropic (Claude)" in providers
    enable_gemini = "Gemini" in providers
    enable_ollama = "Ollama (local)" in providers

    # OpenAI
    openai_key = ""
    openai_base_url = existing.get("provider.openai.base_url", DEFAULT_OPENAI_BASE_URL)
    openai_model_existing = existing.get("provider.openai.model", DEFAULT_OPENAI_MODEL)
    if openai_model_existing not in OPENAI_MODEL_CHOICES:
        openai_model_existing = OPENAI_MODEL_CHOICES[0]
    if enable_openai:
        openai_base_url = questionary.text("OpenAI base URL:", default=openai_base_url).ask() or openai_base_url
        openai_model = questionary.select("OpenAI model (arrow keys):", choices=OPENAI_MODEL_CHOICES, default=openai_model_existing).ask()
        if openai_model is None:
            raise SystemExit(130)
        openai_key = questionary.password("OpenAI API key (hidden):").ask() or ""
        if not openai_key.strip():
            questionary.print("OpenAI API key is required when OpenAI is enabled.", style="bold fg:red")
            raise SystemExit(2)
    else:
        openai_model = openai_model_existing

    # Anthropic
    anthropic_key = ""
    anthropic_model_existing = existing.get("provider.anthropic.model", DEFAULT_ANTHROPIC_MODEL)
    if anthropic_model_existing not in ANTHROPIC_MODEL_CHOICES:
        anthropic_model_existing = ANTHROPIC_MODEL_CHOICES[0]
    if enable_anthropic:
        anthropic_model = questionary.select("Anthropic model (arrow keys):", choices=ANTHROPIC_MODEL_CHOICES, default=anthropic_model_existing).ask()
        if anthropic_model is None:
            raise SystemExit(130)
        anthropic_key = questionary.password("Anthropic API key (hidden):").ask() or ""
        if not anthropic_key.strip():
            questionary.print("Anthropic API key is required when Anthropic is enabled.", style="bold fg:red")
            raise SystemExit(2)
    else:
        anthropic_model = anthropic_model_existing

    # Gemini
    gemini_key = ""
    gemini_model_existing = existing.get("provider.gemini.model", DEFAULT_GEMINI_MODEL)
    if gemini_model_existing not in GEMINI_MODEL_CHOICES:
        gemini_model_existing = GEMINI_MODEL_CHOICES[0]
    if enable_gemini:
        gemini_model = questionary.select("Gemini model (arrow keys):", choices=GEMINI_MODEL_CHOICES, default=gemini_model_existing).ask()
        if gemini_model is None:
            raise SystemExit(130)
        gemini_key = questionary.password("Gemini API key (hidden):").ask() or ""
        if not gemini_key.strip():
            questionary.print("Gemini API key is required when Gemini is enabled.", style="bold fg:red")
            raise SystemExit(2)
    else:
        gemini_model = gemini_model_existing

    # Ollama (fetch models)
    ollama_base = existing.get("provider.ollama.base_url", DEFAULT_OLLAMA_BASE)
    ollama_model_existing = existing.get("provider.ollama.model", DEFAULT_OLLAMA_MODEL)
    if enable_ollama:
        ollama_base = questionary.text("Ollama base URL:", default=ollama_base).ask() or ollama_base
        try:
            models = ollama_list_models(ollama_base, timeout=10)
        except Exception as e:
            questionary.print(f"Could not fetch Ollama models from {ollama_base}: {e}", style="bold fg:red")
            models = []
        if not models:
            models = sorted(set([ollama_model_existing, DEFAULT_OLLAMA_MODEL]))
        default_pick = ollama_model_existing if ollama_model_existing in models else models[0]
        ollama_model = questionary.select("Ollama model (fetched from API):", choices=models, default=default_pick).ask()
        if ollama_model is None:
            raise SystemExit(130)
    else:
        ollama_model = ollama_model_existing

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

    # Provider flags
    set_setting(conn, "provider.openai.enabled", "true" if enable_openai else "false")
    set_setting(conn, "provider.anthropic.enabled", "true" if enable_anthropic else "false")
    set_setting(conn, "provider.gemini.enabled", "true" if enable_gemini else "false")
    set_setting(conn, "provider.ollama.enabled", "true" if enable_ollama else "false")

    # Provider config
    set_setting(conn, "provider.openai.base_url", openai_base_url.strip())
    set_setting(conn, "provider.openai.model", openai_model.strip())
    set_setting(conn, "provider.anthropic.model", anthropic_model.strip())
    set_setting(conn, "provider.gemini.model", gemini_model.strip())
    set_setting(conn, "provider.ollama.base_url", ollama_base.strip())
    set_setting(conn, "provider.ollama.model", ollama_model.strip())

    # Secrets
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
    ap.add_argument("--model", type=str, default=None, help="Model override for the selected provider")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")

    ap.add_argument("--show", action="store_true", help="Show all saved settings (secrets hidden)")
    ap.add_argument("--show-secrets", action="store_true", help="Show decrypted secrets (requires master key)")
    ap.add_argument("--ollama-models", action="store_true", help="List Ollama models from configured base URL")

    # ---- AGENT MODE ----
    ap.add_argument("--agent", type=str, help='Run coding agent, e.g. --agent "build a web UI with login"')
    ap.add_argument("--agent-iters", type=int, default=6, help="Max fix iterations")
    ap.add_argument("--agent-approve", action="store_true", help="Ask before each command")
    ap.add_argument("--agent-wide", action="store_true", help="Disable allowlist (still blocks sudo + obvious destructive patterns)")
    ap.add_argument("--agent-cmd-timeout", type=int, default=900, help="Per-command timeout seconds")

    args = ap.parse_args()

    conn = db_connect(args.db)
    try:
        db_init(conn)

        if args.onboard:
            onboard(conn)
            return 0

        if args.ollama_models:
            settings = get_all_settings(conn)
            base = settings.get("provider.ollama.base_url", DEFAULT_OLLAMA_BASE)
            for m in ollama_list_models(base):
                print(m)
            return 0

        if args.agent is not None:
            return run_agent(
                conn=conn,
                task=args.agent,
                provider=args.provider,
                model_override=args.model,
                timeout=args.timeout,
                iters=args.agent_iters,
                approve=args.agent_approve,
                wide=args.agent_wide,
                cmd_timeout=args.agent_cmd_timeout,
            )

        if args.message is not None:
            return run_message(conn, args.message, provider_filter=args.provider, model_override=args.model, timeout=args.timeout)

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