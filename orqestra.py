#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import shutil
import signal
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

# Default HTTP timeout for agent mode
DEFAULT_AGENT_TIMEOUT = 300

# Max retries for transient network / timeout errors per iteration
MAX_PROVIDER_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds

# Max file size (bytes) to include in prompt context for existing workspaces
MAX_FILE_SIZE_FOR_CONTEXT = 12000  # ~12KB per file
MAX_TOTAL_CONTEXT_SIZE = 80000     # ~80KB total file content in prompt

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
# Spinner
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


def _is_transient_error(exc: Exception) -> bool:
    transient_types = (
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    )
    if isinstance(exc, transient_types):
        return True
    if isinstance(exc, ProviderError):
        msg = str(exc)
        for code in ("429", "500", "502", "503", "504"):
            if f"error {code}" in msg:
                return True
    return False


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
            "maxOutputTokens": 65536,
            "temperature": 0.2,
        },
    }

    if force_json:
        # keep both keys: some SDKs accept different casing
        payload["generationConfig"]["responseMimeType"] = "application/json"
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


def call_provider_with_retries(
    settings: Dict[str, str],
    master_key: bytes,
    provider: str,
    message: str,
    model_override: Optional[str],
    timeout: int,
    force_json: bool = False,
    max_retries: int = MAX_PROVIDER_RETRIES,
) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_provider_for_text(
                settings, master_key, provider, message, model_override, timeout, force_json=force_json
            )
        except Exception as e:
            last_exc = e
            if not _is_transient_error(e) or attempt == max_retries:
                raise
            wait = RETRY_BACKOFF_BASE * attempt
            print(f"\n[Network] ⚠ {type(e).__name__}: {e}")
            print(f"[Network] Retrying in {wait}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# -------------------------
# Message runner
# -------------------------
def run_message(
    conn: sqlite3.Connection,
    message: str,
    provider_filter: Optional[str],
    model_override: Optional[str],
    timeout: int
) -> int:
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

                out = call_provider_with_retries(settings, master_key, p, message, model_override, timeout)

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
# Agent mode — JSON schema (NO TESTS)
# -------------------------
AGENT_JSON_SCHEMA = r"""
Return ONLY valid JSON:

{
  "plan": ["step1", "step2"],
  "files": [
    {"path": "relative/path", "content_b64": "BASE64(UTF-8 file content)"}
  ],
  "commands": [
    ["cmd", "arg1", "arg2"]
  ],
  "run_instructions": "How the user starts/uses it",
  "notes": "short notes"
}

Rules:
- paths MUST be relative, no absolute paths, no ".."
- files MUST use content_b64 (base64 of UTF-8 content). No raw code anywhere else.
- commands MUST be non-interactive and MUST terminate on their own.
- DO NOT include any test/smoke commands; runtime testing is disabled.
- run_instructions: how the user starts/uses it. Do NOT put these in commands.

Reliability:
- Keep JSON strings SHORT.
- If you need many/large files, return fewer per response and iterate.
"""

# Minimal safety blocks (keep)
AGENT_BLOCK_PATTERNS = [
    r"\brm\b\s+-rf\s+/\b",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bshutdown\b|\breboot\b",
]

# Safe-mode allowlist (wide mode disables this)
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
    "go",
    "cargo",
    "mvn",
    "gradle",
    "dotnet",
    "ruby", "gem", "bundle",
    "php", "composer",
    "javac", "java",
    "gcc", "g++", "clang",
    "cmake",
    "rustc",
}

_TEXT_EXTENSIONS = {
    ".js", ".mjs", ".ts", ".tsx", ".jsx",
    ".py", ".rb", ".go", ".rs", ".java", ".kt", ".scala", ".cs", ".fs", ".vb",
    ".c", ".cpp", ".cc", ".h", ".hpp",
    ".html", ".htm", ".ejs", ".hbs", ".pug", ".vue", ".svelte",
    ".css", ".scss", ".less",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".cfg",
    ".md", ".txt", ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql",
    ".xml", ".pom",
    ".dockerfile", ".dockerignore", ".gitignore",
    ".makefile", ".cmake",
    ".r", ".R", ".jl", ".lua", ".pl", ".pm",
    ".swift", ".m",
    ".tf", ".hcl",
    ".proto",
}


def _sanitize_task_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")[:60]
    return safe or "task"


# -------------------------
# Workspace management
# -------------------------
def _is_inside_workspace_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(WORKSPACE_ROOT.resolve())
        return True
    except ValueError:
        return False


def _has_project_files(path: Path) -> bool:
    indicators = [
        "package.json", "pnpm-lock.yaml", "yarn.lock",
        "pyproject.toml", "requirements.txt", "Pipfile", "setup.py",
        "go.mod", "Cargo.toml",
        "pom.xml", "build.gradle", "settings.gradle", "gradlew",
        "composer.json", "Gemfile",
        "Makefile", "CMakeLists.txt",
        "Dockerfile", "docker-compose.yml",
        "README.md",
    ]
    for name in indicators:
        if (path / name).exists():
            return True
    for c in path.iterdir():
        if c.name.startswith("."):
            continue
        if c.name in {"raw_agent_response.txt", "node_modules", "__pycache__"}:
            continue
        return True
    return False


def _workspace_file_count(ws: Path) -> int:
    skip_dirs = {".git", "node_modules", ".venv", "__pycache__", ".pytest_cache"}
    count = 0
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if f != "raw_agent_response.txt":
                count += 1
    return count


def resolve_workspace(explicit_ws: Optional[str], task: str) -> Tuple[Path, bool]:
    if explicit_ws:
        ws = Path(explicit_ws).resolve()
        if ws.exists() and ws.is_dir():
            print(f"[Agent] Using explicit workspace: {ws}")
            return ws, True
        ws.mkdir(parents=True, exist_ok=True)
        print(f"[Agent] Created explicit workspace: {ws}")
        return ws, False

    cwd = Path.cwd().resolve()
    if _is_inside_workspace_root(cwd) and _has_project_files(cwd):
        print(f"[Agent] Detected existing project in CWD, continuing here: {cwd}")
        return cwd, True

    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ws = (WORKSPACE_ROOT / f"{stamp}-{_sanitize_task_name(task)}").resolve()
    return ws, False


def agent_tree(ws: Path, max_entries: int = 250) -> str:
    if not ws.exists():
        return "(empty -- workspace not yet created)"
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


def _read_workspace_files(ws: Path) -> str:
    if not ws.exists():
        return ""

    skip_dirs = {
        ".git", "node_modules", ".venv", "__pycache__", ".pytest_cache",
        "dist", "build", ".next", "target", "bin", "obj",
        ".gradle", ".idea", ".vscode"
    }
    skip_files = {"raw_agent_response.txt", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"}

    parts: List[str] = []
    total_size = 0

    for root, dirs, files in os.walk(ws):
        dirs[:] = sorted([d for d in dirs if d not in skip_dirs])
        rootp = Path(root)
        relroot = rootp.relative_to(ws)

        for fname in sorted(files):
            if fname in skip_files:
                continue

            fpath = rootp / fname
            relpath = relroot / fname
            suffix = fpath.suffix.lower()

            if suffix not in _TEXT_EXTENSIONS and suffix != "":
                continue
            if suffix == "" and fname.lower() not in {
                "dockerfile", "makefile", "procfile", "gemfile", "rakefile",
                "cmakelists.txt", "vagrantfile", "brewfile", "justfile",
            }:
                continue

            try:
                size = fpath.stat().st_size
            except OSError:
                continue

            if size > MAX_FILE_SIZE_FOR_CONTEXT:
                parts.append(f"\n--- {relpath} (too large: {size} bytes, skipped) ---")
                continue

            if total_size + size > MAX_TOTAL_CONTEXT_SIZE:
                parts.append(f"\n--- (remaining files skipped, context limit reached) ---")
                return "\n".join(parts)

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                total_size += size
                parts.append(f"\n--- {relpath} ---\n{content}")
            except Exception:
                parts.append(f"\n--- {relpath} (could not read) ---")

    return "\n".join(parts) if parts else ""


# -------------------------
# JSON parsing helpers
# -------------------------
def _strip_code_fences(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json|javascript|js|text)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_outer_object(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start: end + 1]
    return raw


def _escape_invalid_controls_inside_strings(s: str) -> str:
    out: List[str] = []
    in_str = False
    esc = False
    for ch in s:
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
                esc = False
            continue
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            out.append(ch)
            in_str = False
            continue
        o = ord(ch)
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20:
            out.append("\\u%04x" % o)
        else:
            out.append(ch)
    return "".join(out)


def agent_parse_json(raw: str) -> Dict[str, Any]:
    raw0 = (raw or "").strip()
    if not raw0:
        raise ValueError("Agent returned empty response (no JSON).")

    s = _strip_code_fences(raw0)
    s = _extract_outer_object(s)

    try:
        return json.loads(s)
    except Exception:
        pass

    repaired = _escape_invalid_controls_inside_strings(s)
    try:
        return json.loads(repaired)
    except Exception:
        pass

    m = re.search(r"\{.*\}\s*$", raw0, flags=re.S)
    if m:
        cand = _escape_invalid_controls_inside_strings(_strip_code_fences(m.group(0)))
        try:
            return json.loads(cand)
        except Exception:
            pass

    preview = s[:900].replace("\n", "\\n")
    raise ValueError(f"Agent did not return valid JSON. Preview: {preview}")


# -------------------------
# File write (robust base64)
# -------------------------
def _b64decode_relaxed(s: str) -> bytes:
    if s is None:
        raise ValueError("content_b64 is None")

    cleaned = re.sub(r"\s+", "", str(s))

    if cleaned.startswith("data:"):
        comma = cleaned.find(",")
        if comma != -1:
            cleaned = cleaned[comma + 1:]

    pad = (-len(cleaned)) % 4
    if pad:
        cleaned += "=" * pad

    try:
        return base64.b64decode(cleaned, validate=False)
    except Exception:
        try:
            return base64.urlsafe_b64decode(cleaned)
        except Exception as e:
            preview = cleaned[:80]
            raise ValueError(f"Invalid base64 (preview='{preview}...'): {e}") from e


def agent_write_files(ws: Path, files: List[Dict[str, str]]) -> None:
    for f in files:
        path = (f.get("path") or "").strip()
        if not path:
            raise ValueError("Agent returned empty file path")
        if path.startswith(("/", "\\")) or ".." in Path(path).parts:
            raise ValueError(f"Unsafe path from agent: {path}")

        b64 = f.get("content_b64", None)

        try:
            if b64 is None:
                content = str(f.get("content") or "")
                data = content.encode("utf-8")
            else:
                data = _b64decode_relaxed(b64)

            p = safe_join(ws, path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

        except Exception as e:
            raise ValueError(f"FILE_WRITE_ERROR for '{path}': {e}") from e


# -------------------------
# Command execution helpers (UNIVERSAL timeout + kill)
# -------------------------
def _shlex_join(cmd: List[str]) -> str:
    def q(s: str) -> str:
        if re.search(r'\s|"', s):
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


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.terminate()
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


def agent_run_cmd(ws: Path, cmd: List[str], timeout_s: int, approve: bool, wide: bool) -> Tuple[bool, int, str, str]:
    ok, reason = agent_cmd_allowed(cmd, wide=wide)
    if not ok:
        return False, 127, "", f"Blocked: {reason}"

    if approve:
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"About to run:\n  {_shlex_join(cmd)}\n")
        ans = input("Run this command? [y/N]: ").strip().lower()
        if ans != "y":
            return False, 126, "", "User declined command"

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ws),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError as e:
        return False, 127, "", f"Command not found: {e}"

    try:
        out, err = proc.communicate(timeout=timeout_s)
        rc = proc.returncode if proc.returncode is not None else 1
        return rc == 0, rc, out or "", err or ""
    except subprocess.TimeoutExpired:
        out_partial = ""
        err_partial = ""
        try:
            out_partial = proc.stdout.read() if proc.stdout else ""
        except Exception:
            pass
        try:
            err_partial = proc.stderr.read() if proc.stderr else ""
        except Exception:
            pass

        _terminate_process_tree(proc)
        note = f"Command timed out after {timeout_s}s and was terminated."
        rc = 124
        return False, rc, out_partial, (err_partial + ("\n" if err_partial else "") + note).strip()


# -------------------------
# Agent prompt
# -------------------------
def build_agent_prompt(task: str, ws: Path, last_logs: str, is_existing_ws: bool) -> str:
    tree = agent_tree(ws)

    file_contents = ""
    if ws.exists() and tree not in ("(empty)", "(empty -- workspace not yet created)"):
        file_contents = _read_workspace_files(ws)

    context = ""
    if is_existing_ws and file_contents:
        context = """
IMPORTANT: You are continuing work in an EXISTING project workspace.
The current file contents are shown below. Review them carefully.
Do NOT recreate files that already exist unless you need to modify them.
When you modify a file, include the COMPLETE updated file content in content_b64.
Build on top of what is already there.
""".strip()

    file_section = ""
    if file_contents:
        file_section = f"\n\nCurrent file contents:\n{file_contents}\n"

    return f"""
You are Orqestra Agent: you create code, write files, and provide shell commands to build/install.

IMPORTANT:
- Return ONLY JSON (no markdown, no explanation).
- For files, you MUST use content_b64 = base64(UTF-8). Do NOT include raw file contents anywhere.
- Keep plan/notes/run_instructions SHORT to avoid malformed JSON.
- Runtime tests/smoke tests are DISABLED. Do not include them.

{context}
{AGENT_JSON_SCHEMA}

Task:
{task}

Workspace tree:
{tree}
{file_section}
Last execution logs (if any):
{last_logs}
""".strip()


# -------------------------
# Workspace rollback
# -------------------------
def _rollback_workspace(ws: Path, is_existing: bool) -> None:
    if is_existing:
        print("[Agent] Existing workspace preserved (no rollback for existing projects).")
        return
    if not ws.exists():
        return
    file_count = _workspace_file_count(ws)
    if file_count == 0:
        try:
            shutil.rmtree(ws)
            print(f"[Agent] 🗑  Rolled back empty workspace: {ws}")
        except Exception as e:
            print(f"[Agent] Warning: could not remove workspace: {e}")
    else:
        print(f"[Agent] Workspace has {file_count} file(s), keeping: {ws}")


# ===================================================================
# Agent run loop
# ===================================================================
def _run_command_list(
    ws: Path,
    commands: List[List[str]],
    label: str,
    approve: bool,
    wide: bool,
    timeout_s: int,
) -> Tuple[bool, str]:
    logs_parts: List[str] = []
    all_ok = True

    for idx, c in enumerate(commands, start=1):
        if not isinstance(c, list) or not all(isinstance(x, str) for x in c):
            all_ok = False
            logs_parts.append(f"[{label} {idx}] INVALID COMMAND: {c}")
            break

        print(f"  -> {_shlex_join(c)}")

        run_spin = Spinner("Executing")
        run_spin.start()
        try:
            ok, rc, out, err = agent_run_cmd(ws, c, timeout_s=timeout_s, approve=approve, wide=wide)
        finally:
            run_spin.stop()

        status = "OK" if ok else "FAIL"
        print(f"     [{status}] exit={rc}")
        if not ok and err:
            err_preview = err.strip()[-700:]
            print(f"     Error: {err_preview}")

        logs_parts.append("\n".join([
            f"[{label} {idx}] {_shlex_join(c)}",
            f"exit={rc}",
            "stdout:", out[-4000:] if out else "",
            "stderr:", err[-4000:] if err else "",
            "-" * 40,
        ]))

        if not ok:
            all_ok = False
            break

    return all_ok, "\n".join(logs_parts)


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
    explicit_workspace: Optional[str] = None,
) -> int:
    settings = get_all_settings(conn)
    provs = enabled_providers(settings)
    if not provs:
        print("No providers enabled. Run: python3 orqestra.py --onboard")
        return 2

    chosen = provider_alias(provider) if provider else provs[0]
    if chosen not in provs:
        print(f"Provider '{chosen}' not enabled. Enabled: {', '.join(provs)}")
        return 2

    master_key = load_or_create_master_key()
    ws, is_existing = resolve_workspace(explicit_workspace, task)

    print(f"\n[Agent] Workspace: {ws}" + (" (existing)" if is_existing else " (new, deferred creation)"))

    last_logs = ""
    final_instructions = ""
    ever_wrote_files = False
    last_success = False

    valid_rounds = 0
    total_attempts = 0
    max_total_attempts = max(30, iters * 8)

    while valid_rounds < iters and total_attempts < max_total_attempts:
        total_attempts += 1
        print(f"\n[Agent] Iteration {valid_rounds + 1}/{iters}  (attempt {total_attempts}/{max_total_attempts})")

        # --- Call LLM ---
        spin = Spinner("Waiting for the agent")
        spin.start()
        raw = ""
        try:
            prompt = build_agent_prompt(task, ws, last_logs, is_existing_ws=is_existing)
            raw = call_provider_with_retries(
                settings, master_key, chosen, prompt, model_override, timeout, force_json=True
            )
            if not (raw or "").strip():
                retry_prompt = prompt + "\n\nCRITICAL: Output must be ONLY JSON."
                raw = call_provider_with_retries(
                    settings, master_key, chosen, retry_prompt, model_override, timeout, force_json=True
                )
        except Exception as e:
            spin.stop()
            print(f"\n[Agent] ❌ Provider error: {type(e).__name__}: {e}")
            print("[Agent] All retries exhausted. Aborting.")
            if not ever_wrote_files:
                _rollback_workspace(ws, is_existing)
            return 3
        finally:
            spin.stop()

        # Save raw response
        try:
            if not ws.exists():
                ws.mkdir(parents=True, exist_ok=True)
            safe_join(ws, "raw_agent_response.txt").write_text(raw or "", encoding="utf-8")
        except Exception:
            pass

        # --- Parse JSON ---
        try:
            data = agent_parse_json(raw)
        except ValueError as e:
            print("\n[Agent] ❌ JSON parse failed (will retry; does NOT consume iteration).")
            last_logs = "\n".join([
                "JSON_PARSE_ERROR:", str(e),
                "RAW_OUTPUT_PREVIEW:", (raw or "")[:1400].replace("\n", "\\n"),
                "INSTRUCTION: Return ONLY valid JSON object. No markdown, no extra text.",
            ])
            continue

        plan = data.get("plan") or []
        files = data.get("files") or []
        commands = data.get("commands") or []
        final_instructions = (data.get("run_instructions") or "").strip()

        if plan:
            print("\n[Plan]")
            for p in plan:
                print(f"  • {p}")

        # --- Write files ---
        if files:
            if not ws.exists():
                ws.mkdir(parents=True, exist_ok=True)
                print(f"[Agent] Created workspace: {ws}")

            print(f"\n[Agent] Writing {len(files)} file(s)...")
            try:
                agent_write_files(ws, files)
                ever_wrote_files = True
            except Exception as e:
                print(f"\n[Agent] ❌ Writing files failed: {e}")
                print("[Agent] Will retry (does NOT consume iteration).")
                last_logs = "\n".join([
                    "FILE_WRITE_ERROR:", str(e),
                    "INSTRUCTION: Fix your base64. content_b64 MUST be valid base64 of UTF-8 file content.",
                    "Return ONLY JSON.",
                ])
                continue

        if not ws.exists():
            ws.mkdir(parents=True, exist_ok=True)

        # --- Run commands (install/build) ---
        if commands:
            print(f"\n[Agent] Running {len(commands)} command(s)...")
            cmds_ok, cmds_logs = _run_command_list(
                ws, commands, "cmd", approve, wide, timeout_s=cmd_timeout
            )
            if not cmds_ok:
                print("\n[Agent] ❌ Command failed. Sending error to AI for fix...")
                last_logs = "\n".join([
                    cmds_logs,
                    "INSTRUCTION: The command above failed OR timed out. Read the error output carefully.",
                    "Fix the code/config and return corrected file(s). Return ONLY JSON.",
                ])
                last_success = False
                continue

        # No tests at all: if we got here, the iteration is considered valid.
        valid_rounds += 1
        last_success = True
        print("\n[Agent] ✅ Iteration complete — commands (if any) succeeded; no tests executed.")
        break

    # ========================
    # Final summary
    # ========================
    print("\n=========================")
    print("AGENT DONE")
    print("=========================")

    if last_success and ever_wrote_files:
        print(f"Workspace: {ws}")
        if final_instructions:
            print("\nRun instructions:\n")
            print(final_instructions)
        else:
            print("\nNo run instructions returned. Check workspace files.")
        return 0

    if not ever_wrote_files:
        _rollback_workspace(ws, is_existing)

    print(f"Workspace: {ws}")
    if total_attempts >= max_total_attempts:
        print("\n[Agent] Reached max attempts safety cap.")
    return 1


# -------------------------
# Onboarding
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
                checked=(
                    existing.get("provider.gemini.enabled", "false") == "true"
                    or existing.get("provider.google.enabled", "false") == "true"
                ),
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
        openai_model = questionary.select(
            "OpenAI model (arrow keys):", choices=OPENAI_MODEL_CHOICES, default=openai_model_existing
        ).ask()
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
        anthropic_model = questionary.select(
            "Anthropic model (arrow keys):", choices=ANTHROPIC_MODEL_CHOICES, default=anthropic_model_existing
        ).ask()
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
        gemini_model = questionary.select(
            "Gemini model (arrow keys):", choices=GEMINI_MODEL_CHOICES, default=gemini_model_existing
        ).ask()
        if gemini_model is None:
            raise SystemExit(130)
        gemini_key = questionary.password("Gemini API key (hidden):").ask() or ""
        if not gemini_key.strip():
            questionary.print("Gemini API key is required when Gemini is enabled.", style="bold fg:red")
            raise SystemExit(2)
    else:
        gemini_model = gemini_model_existing

    # Ollama
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
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default: 60, agent default: 300)")

    ap.add_argument("--show", action="store_true", help="Show all saved settings (secrets hidden)")
    ap.add_argument("--show-secrets", action="store_true", help="Show decrypted secrets (requires master key)")
    ap.add_argument("--ollama-models", action="store_true", help="List Ollama models from configured base URL")

    # ---- AGENT MODE ----
    ap.add_argument("--agent", type=str, help='Run coding agent, e.g. --agent "build a web UI with login"')
    ap.add_argument("--agent-iters", type=int, default=10, help="Max valid fix rounds")
    ap.add_argument("--agent-approve", action="store_true", help="Ask before each command")
    ap.add_argument("--agent-wide", action="store_true", help="Disable allowlist (still blocks sudo + destructive patterns)")
    ap.add_argument("--agent-cmd-timeout", type=int, default=900, help="Per-command timeout seconds")
    ap.add_argument("--agent-workspace", type=str, default=None,
                    help='Use existing workspace dir (e.g. --agent-workspace ./workspace/my-project or "." for CWD)')

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
            agent_timeout = args.timeout if args.timeout != 60 else DEFAULT_AGENT_TIMEOUT
            return run_agent(
                conn=conn,
                task=args.agent,
                provider=args.provider,
                model_override=args.model,
                timeout=agent_timeout,
                iters=args.agent_iters,
                approve=args.agent_approve,
                wide=args.agent_wide,
                cmd_timeout=args.agent_cmd_timeout,
                explicit_workspace=args.agent_workspace,
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