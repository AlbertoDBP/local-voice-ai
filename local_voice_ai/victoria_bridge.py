"""OpenAI-chat-completions-compatible bridge putting Victoria (Claude Code
headless) behind LiveKit's turn detection.

Why this exists: LiveKit's LLM slot expects an OpenAI-compatible endpoint.
Claude isn't one natively. This translates POST /v1/chat/completions into a
PERSISTENT `claude -p --input-format stream-json --output-format stream-json`
process per LiveKit room: user turns are written to its stdin as JSON lines,
assistant text deltas are read from stdout and forwarded as SSE chunks as
they arrive. Measured effect vs the v1 spawn-per-turn design: first token
~1-2s instead of ~16s per turn (CLI startup + session reload eliminated).

The persistent process keeps conversational context for the life of the
room (one Claude session per room, resumed by --resume if the process dies).

Persona scope (unchanged from v1, see delegation brief in the vault dossier):
Victoria's CORE persona/protocols only (identity, critic voice, grounding
rule, fast-lane test + log format, draft-first). The multi-file sub-skills
(victoria:idea/delegate/status) are NOT wired — those need broader
filesystem access than the vault-scoped permission surface allows. Ideas
and commissions get acknowledged by voice, executed in a text session.

Permission surface (verified live 2026-07-10 — do not loosen without a new
approval point, per the delegation brief):
  - Bash tool fully disallowed. Tested: --allowedTools alone does NOT
    restrict a headless (-p) session to a narrow command prefix — an
    allowed "Bash(<brain>:*)" pattern still let an unrelated `whoami` run.
    Only --disallowedTools "Bash" actually blocks it.
  - The ONLY tools available are Read/Write (scoped to --add-dir <vault>)
    plus the four tools on the vault MCP server (vault_mcp_server.py):
    brain_query, read_note, list_notes, brain_store. No other MCP servers
    (--strict-mcp-config), no Edit outside the vault.

Run: python -m local_voice_ai.victoria_bridge  (default port 8801)
"""
import asyncio
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("victoria_bridge")

VAULT_PATH = "/Users/alberto/Library/Mobile Documents/iCloud~md~obsidian/Documents/ACH-2B"
CLAUDE_BIN = os.getenv("CLAUDE_BIN", "claude")
CLAUDE_MODEL = os.getenv("VICTORIA_MODEL", "claude-sonnet-5")
BRIDGE_PORT = int(os.getenv("VICTORIA_BRIDGE_PORT", "8801"))
TURN_TIMEOUT_SEC = int(os.getenv("VICTORIA_CLAUDE_TIMEOUT", "60"))
# If no text has arrived this many seconds into a turn (e.g. Claude is off
# doing a brain_query tool call), speak a short content-free filler so the
# line isn't dead air.
FILLER_AFTER_SEC = float(os.getenv("VICTORIA_FILLER_AFTER", "2.0"))
VAULT_MCP_CONFIG = str(Path(__file__).parent.parent / "victoria-vault-mcp-config.json")
REPO_DIR = str(Path(__file__).parent.parent)

THINKING_FILLERS = ["One moment.", "Let's see.", "Okay, one sec.", "Right, let me look."]

PERSONA_SYSTEM_PROMPT = """You are Victoria, Alberto Chacin's chief of staff — speaking to him now by \
voice over a live phone/mic connection (LiveKit), not text. Keep every reply to two or three short, \
plain sentences: no markdown, no bullet lists, no headers, no emojis — you are being read aloud by a \
TTS engine.

Voice — critic by default: challenge the premise before polishing details. Ask for numbers, not vague \
claims. Argue the strongest opposing case once, honestly, before agreeing. If the conversation is \
speculating about facts, say so plainly ("we're guessing — let's find out") rather than inventing detail.

Grounding — vault first: before forming an opinion on anything touching Alberto's life, businesses, \
people, or plans, call the brain_query tool with the question. Reason from its evidence. If the evidence \
is thin, say so rather than guessing.

Protocols, still in force over voice:
1. Fast lane (admin exception): a request executes directly, with no further discussion, only when it \
fully specifies the action, completes in one step, and its effect is exactly what was asked. After \
acting, call brain_store with a line in this exact format: "## [today's date] admin | victoria/fast: \
<what was done>". If any part of that test fails, say so out loud and tell Alberto it needs a proper \
text-session commission (victoria:delegate) rather than acting.
2. No open-ended commissioning by voice. Ideas and real commissions get acknowledged here, then picked \
up properly in a text session — say so plainly rather than improvising a build over the phone.
3. Draft-first: you never send, pay, sign, file, or deploy anything by voice. The only exception is a \
fast-lane action (protocol 1).
4. Kill-log check: when a new idea comes up, mention you'd normally check the vault for a killed or \
parked version of it — full check happens in text, not here.

Your only tools are brain_query, read_note, list_notes, and brain_store — all vault-scoped. You have no \
shell access and no other tools; don't imply otherwise."""


class ClaudeSession:
    """One persistent headless Claude process per LiveKit room.

    stdin: one JSON line per user turn. stdout: stream-json events; a turn
    ends at the top-level {"type": "result"} event. One turn at a time
    (self._lock) — LiveKit won't overlap turns in a room, the lock is a
    guard against bridge-level races.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.proc: asyncio.subprocess.Process | None = None
        self._started_once = False
        self._lock = asyncio.Lock()

    async def _ensure_proc(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            return
        cmd = [
            CLAUDE_BIN, "-p", "--verbose",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--model", CLAUDE_MODEL,
            "--add-dir", VAULT_PATH,
            "--allowedTools", "mcp__vault__brain_query", "mcp__vault__brain_store",
            "mcp__vault__read_note", "mcp__vault__list_notes",
            "--disallowedTools", "Bash",
            "--mcp-config", VAULT_MCP_CONFIG, "--strict-mcp-config",
        ]
        if self._started_once:
            cmd += ["--resume", self.session_id]  # process died mid-room: reload context
        else:
            cmd += ["--session-id", self.session_id, "--system-prompt", PERSONA_SYSTEM_PROMPT]
        logger.info("spawning claude (resume=%s session=%s)", self._started_once, self.session_id)
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=REPO_DIR,
        )
        self._started_once = True

    async def turn_stream(self, user_text: str) -> AsyncIterator[str]:
        """Yield assistant text deltas for one user turn."""
        async with self._lock:
            await self._ensure_proc()
            assert self.proc is not None and self.proc.stdin is not None
            msg = {"type": "user",
                   "message": {"role": "user", "content": [{"type": "text", "text": user_text}]}}
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            await self.proc.stdin.drain()

            deadline = time.monotonic() + TURN_TIMEOUT_SEC
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error("turn timeout; killing claude for session %s", self.session_id)
                    self.proc.kill()
                    self.proc = None
                    yield " Sorry — that took too long on my end. Try again?"
                    return
                try:
                    line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    continue
                if not line:  # process died mid-turn
                    logger.error("claude EOF mid-turn (session %s)", self.session_id)
                    self.proc = None
                    yield " Something went wrong on my end. Try again?"
                    return
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "stream_event":
                    ev = event.get("event", {})
                    if ev.get("type") == "content_block_delta":
                        text = ev.get("delta", {}).get("text", "")
                        if text:
                            yield text
                elif etype == "result":
                    if event.get("is_error"):
                        logger.error("claude turn error: %s", str(event)[:300])
                        yield " Something went wrong on my end. Try again?"
                    return


# room_name -> ClaudeSession, keyed by the X-Room-Name header agent.py sets
_sessions: dict[str, ClaudeSession] = {}


def _get_session(room_name: str) -> ClaudeSession:
    if room_name not in _sessions:
        _sessions[room_name] = ClaudeSession()
    return _sessions[room_name]


def _sse_chunk(content: str | None, finish_reason: str | None, model: str, chunk_id: str) -> str:
    payload = {
        "id": chunk_id, "object": "chat.completion.chunk", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": ({"role": "assistant", "content": content} if content
                                            else {}), "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def build_app() -> FastAPI:
    app = FastAPI(title="victoria-bridge", version="0.2.0")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        user_turns = [m for m in messages if m.get("role") == "user"]
        user_text = user_turns[-1].get("content", "") if user_turns else ""
        if isinstance(user_text, list):
            user_text = " ".join(p.get("text", "") for p in user_text if isinstance(p, dict))

        room_name = request.headers.get("x-room-name", "default")
        session = _get_session(room_name)

        model = body.get("model", CLAUDE_MODEL)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if body.get("stream"):
            async def gen():
                # Pump the generator through a queue so the filler timeout can
                # sit on queue.get(). (wait_for directly on anext() would CANCEL
                # the generator on timeout, killing the turn mid-flight.)
                queue: asyncio.Queue[str | None] = asyncio.Queue()

                async def pump():
                    try:
                        async for d in session.turn_stream(user_text):
                            await queue.put(d)
                    finally:
                        await queue.put(None)

                pump_task = asyncio.create_task(pump())
                got_text = False
                try:
                    while True:
                        # Filler only when the line would otherwise be dead air
                        # before the FIRST text (e.g. a brain_query round trip).
                        if got_text:
                            item = await queue.get()
                        else:
                            try:
                                item = await asyncio.wait_for(queue.get(), timeout=FILLER_AFTER_SEC)
                            except asyncio.TimeoutError:
                                yield _sse_chunk(random.choice(THINKING_FILLERS) + " ",
                                                 None, model, chunk_id)
                                item = await queue.get()
                        if item is None:
                            break
                        got_text = True
                        yield _sse_chunk(item, None, model, chunk_id)
                finally:
                    if not pump_task.done():
                        pump_task.cancel()
                yield _sse_chunk(None, "stop", model, chunk_id)
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        reply = "".join([d async for d in session.turn_stream(user_text)])
        return JSONResponse({
            "id": chunk_id, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    @app.get("/health")
    async def health():
        return {"ok": True, "active_rooms": list(_sessions.keys())}

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(build_app(), host="127.0.0.1", port=BRIDGE_PORT)
