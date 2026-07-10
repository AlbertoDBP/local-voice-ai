"""Vault-scoped MCP server for headless Victoria (stdio transport).

Exposes exactly four tools to the LiveKit-Victoria bridge's headless Claude
process: brain_query / read_note / list_notes (read-only) and brain_store
(write, vault-only). This is the enforcement boundary for "vault-scoped
tools only" — deliberately NOT exposing run_command or any shell access.
Claude's own Bash tool is separately disabled at the CLI-flag level
(--disallowedTools "Bash") because --allowedTools alone does not restrict
headless (-p) sessions to a narrow command prefix (verified live
2026-07-10: an allowed "Bash(<brain>:*)" pattern did not stop an
unrelated `whoami` from running — only a full Bash disallow does).

brain_store writes immediately, with no CONFIRM/pending staging — unlike
the qwen2.5 vault agent (which needs a mechanical gate because it can't
reason), Victoria is Claude: the gate is her own persona instructions
(draft-first, fast-lane test), enforced by the system prompt the bridge
injects, not by this server.

Run standalone: python -m local_voice_ai.vault_mcp_server
"""
import datetime
import subprocess

from mcp.server.fastmcp import FastMCP

from . import vault_tools

mcp = FastMCP("vault")


@mcp.tool()
def brain_query(question: str) -> str:
    """Retrieve evidence from Alberto's second-brain vault for a question.
    Use FIRST for any question about Alberto, his businesses, finances,
    people, or plans."""
    return vault_tools.brain_query(question)


@mcp.tool()
def read_note(path: str) -> str:
    """Read a markdown note from the vault by its vault-relative path."""
    return vault_tools.read_note(path)


@mcp.tool()
def list_notes(subdir: str = "") -> str:
    """List markdown notes in the vault, optionally under a subdirectory."""
    return vault_tools.list_notes(subdir)


@mcp.tool()
def brain_store(text: str) -> str:
    """Save a new memory to the vault (writes a memory file + catalogue
    line + log entry, in one step). Only call this after you have actually
    decided — per Victoria's own protocols — that this is worth keeping."""
    out = subprocess.run([vault_tools.BRAIN, "store", text], capture_output=True,
                         text=True, timeout=30)
    with open(vault_tools.LOG_FILE, "a", encoding="utf-8") as fh:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fh.write(f"[{ts}] brain_store (victoria bridge): {text[:120]}\n")
    return out.stdout.strip() or "stored."


if __name__ == "__main__":
    mcp.run(transport="stdio")
