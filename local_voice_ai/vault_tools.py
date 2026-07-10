"""Vault tools for the LiveKit agent — ported from ~/second-brain-agent/tools.py.

Read-only tools execute immediately. Writes (store_memory) and shell commands
(run_command) are two-step: the tool stashes a pending action and returns a
confirmation prompt for the agent to speak; confirm_pending() executes it,
cancel_pending() drops it. ask_claude() escalates one question to Claude Code
headless (existing CLI auth — no API key, no per-token billing).
"""
import datetime
import os
import shlex
import subprocess

VAULT_PATH = "/Users/alberto/Library/Mobile Documents/iCloud~md~obsidian/Documents/ACH-2B"
BRAIN = os.path.join(VAULT_PATH, "99 System", "bin", "brain")
LOG_FILE = "/Volumes/CH-DataOne/AlbertoDBP/Second-Brain-Master/local-voice-ai/vault-agent.log"
CLAUDE_CMD = "claude"
CLAUDE_EMPTY_MCP = os.path.expanduser("~/second-brain-agent/empty-mcp.json")

COMMAND_ALLOWLIST = (
    "brain ", "graphify update", "graphify query", "graphify path",
    "graphify explain", "git status", "git log", "git diff", "git pull",
    "ls", "find",
)
COMMAND_BLOCKLIST = (
    "sudo", "rm", "dd", "mkfs", "chmod", "chown", "curl", "wget", "kill",
    "shutdown", "reboot", "mv", "cp", ">", ">>", "|", ";", "&", "`", "$(",
)
ALLOWED_DIRS = (
    VAULT_PATH,
    "/Volumes/CH-DataOne/AlbertoDBP",
    os.path.expanduser("~/second-brain-agent"),
    "/Volumes/CH-DataOne/AlbertoDBP/Second-Brain-Master/local-voice-ai",
)
COMMAND_TIMEOUT_SEC = 60
DEFAULT_CWD = VAULT_PATH

# One pending confirmable action at a time: (description, thunk)
_pending = None


def _log(kind, detail):
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fh.write(f"[{ts}] {kind}: {detail}\n")


def _in_vault(path):
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(VAULT_PATH, path))
    root = os.path.realpath(VAULT_PATH)
    return full if (full == root or full.startswith(root + os.sep)) else None


# --- read-only -------------------------------------------------------------

def brain_query(question: str) -> str:
    try:
        out = subprocess.run([BRAIN, "query", question], capture_output=True,
                             text=True, timeout=30)
        _log("brain_query", question)
        return out.stdout.strip() or "(no evidence returned)"
    except Exception as e:
        return f"(brain_query error: {e})"


def read_note(path: str) -> str:
    full = _in_vault(path)
    if not full:
        return f"(refused: '{path}' is outside the vault)"
    if not os.path.isfile(full):
        return f"(not found: {path})"
    _log("read_note", path)
    return open(full, encoding="utf-8").read()[:6000]


def list_notes(subdir: str = "") -> str:
    base = _in_vault(subdir) if subdir else VAULT_PATH
    if not base:
        return f"(refused: '{subdir}' is outside the vault)"
    hits = []
    for dp, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".md"):
                hits.append(os.path.relpath(os.path.join(dp, f), VAULT_PATH))
    _log("list_notes", subdir or "(root)")
    return "\n".join(sorted(hits)[:200]) or "(no markdown files)"


# --- confirmation-gated ------------------------------------------------------

def store_memory(text: str) -> str:
    global _pending

    def _do():
        out = subprocess.run([BRAIN, "store", text], capture_output=True,
                             text=True, timeout=30)
        _log("store_memory", text[:120])
        return out.stdout.strip() or "stored."

    _pending = (f"save to the vault: {text[:200]}", _do)
    return (f"PENDING CONFIRMATION — ask the user out loud to confirm saving this "
            f"memory, then call confirm_pending only if they say yes: {text[:200]}")


def _command_ok(cmd: str):
    stripped = cmd.strip()
    low = stripped.lower()
    for bad in COMMAND_BLOCKLIST:
        if bad in low:
            return False, f"blocked token '{bad}'"
    if not any(stripped.startswith(p) for p in COMMAND_ALLOWLIST):
        return False, "not on the allowlist"
    try:
        shlex.split(stripped)
    except ValueError as e:
        return False, f"unparseable ({e})"
    return True, "ok"


def run_command(cmd: str, cwd: str = "") -> str:
    global _pending
    cwd = cwd or DEFAULT_CWD
    ok, reason = _command_ok(cmd)
    real_cwd = os.path.realpath(cwd)
    if not any(real_cwd == os.path.realpath(d) or real_cwd.startswith(os.path.realpath(d) + os.sep)
               for d in ALLOWED_DIRS):
        ok, reason = False, f"cwd '{cwd}' not allowed"
    if not ok:
        _log("run_command REFUSED", f"{cmd}  [{reason}]")
        return f"REFUSED — {reason}"

    def _do():
        _log("run_command", f"{cmd}  (cwd={cwd})")
        try:
            r = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True,
                               text=True, timeout=COMMAND_TIMEOUT_SEC)
            out = (r.stdout or r.stderr).strip()
            _log("run_command result", f"rc={r.returncode} {out[:200]}")
            return out[:2000] or f"(done, exit {r.returncode})"
        except subprocess.TimeoutExpired:
            return "(command timed out)"
        except Exception as e:
            return f"(command error: {e})"

    _pending = (f"run: {cmd} in {os.path.basename(cwd)}", _do)
    return (f"PENDING CONFIRMATION — ask the user out loud to confirm, then call "
            f"confirm_pending only if they say yes: {cmd}")


def confirm_pending() -> str:
    global _pending
    if not _pending:
        return "(nothing pending to confirm)"
    desc, thunk = _pending
    _pending = None
    return thunk()


def cancel_pending() -> str:
    global _pending
    if not _pending:
        return "(nothing pending)"
    desc, _ = _pending
    _pending = None
    _log("cancelled", desc)
    return f"(cancelled: {desc})"


# --- Claude escalation -------------------------------------------------------

def ask_claude(question: str) -> str:
    """One-shot question to Claude Code headless. Uses existing CLI auth."""
    _log("ask_claude", question[:200])
    try:
        cmd = [CLAUDE_CMD, "-p", question]
        if os.path.isfile(CLAUDE_EMPTY_MCP):
            cmd += ["--mcp-config", CLAUDE_EMPTY_MCP, "--strict-mcp-config"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           cwd=os.path.expanduser("~"))
        out = (r.stdout or r.stderr).strip()
        return out[:3000] or "(Claude returned nothing)"
    except subprocess.TimeoutExpired:
        return "(Claude timed out after 120 seconds)"
    except FileNotFoundError:
        return "(Claude CLI not found on PATH)"
    except Exception as e:
        return f"(ask_claude error: {e})"
