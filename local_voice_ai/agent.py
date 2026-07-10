"""LiveKit Agents worker.

Moved verbatim from ``livekit_agent/src/agent.py``. The only change is that the
default base URLs are loopback (``127.0.0.1``) instead of Docker service names —
the supervisor spawns the inference children on loopback ports, so this is
correct for both single-image deployment and bare-metal local runs.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


from . import vault_tools


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the voice interface to Alberto's second brain — an Obsidian "
                "vault holding his businesses (DBP, IMC, Student First, 305 Music Lab, "
                "Valencia 218), finances, people, and plans. You are speaking out loud: "
                "keep answers to two or three plain sentences, no formatting, no emojis. "
                "For ANY question about Alberto, his businesses, finances, people, or "
                "plans, call brain_query FIRST and answer only from its evidence; say so "
                "when the evidence is insufficient rather than guessing. "
                "store_memory and run_command stage a pending action: speak the "
                "confirmation question they return, and call confirm_pending only after "
                "the user clearly says yes (cancel_pending on no or on a topic change). "
                "Call ask_claude only when the user explicitly asks to bring Claude in "
                "(for example 'ask Claude'); warn them it takes a minute, and summarize "
                "its reply out loud in your own words."
            ),
        )

    @function_tool()
    async def brain_query(self, context: RunContext, question: str) -> str:
        """Retrieve evidence from Alberto's second-brain vault. Use FIRST for any
        question about Alberto, his businesses, finances, people, or plans.

        Args:
            question: The question to retrieve vault evidence for.
        """
        return vault_tools.brain_query(question)

    @function_tool()
    async def read_note(self, context: RunContext, path: str) -> str:
        """Read a markdown note from the vault by its vault-relative path.

        Args:
            path: Vault-relative path of the note.
        """
        return vault_tools.read_note(path)

    @function_tool()
    async def list_notes(self, context: RunContext, subdir: str = "") -> str:
        """List markdown notes in the vault, optionally under a subdirectory.

        Args:
            subdir: Optional vault subdirectory to list.
        """
        return vault_tools.list_notes(subdir)

    @function_tool()
    async def store_memory(self, context: RunContext, text: str) -> str:
        """Stage saving a new memory to the vault. Returns a confirmation prompt
        to speak; execute with confirm_pending only after the user says yes.

        Args:
            text: The memory text to save.
        """
        return vault_tools.store_memory(text)

    @function_tool()
    async def run_command(self, context: RunContext, cmd: str, cwd: str = "") -> str:
        """Stage an allowlisted shell command (brain, graphify, read-only git, ls,
        find). Returns a confirmation prompt to speak; execute with
        confirm_pending only after the user says yes.

        Args:
            cmd: The command line to run.
            cwd: Optional working directory.
        """
        return vault_tools.run_command(cmd, cwd)

    @function_tool()
    async def confirm_pending(self, context: RunContext) -> str:
        """Execute the pending staged action. Call ONLY after the user clearly
        said yes to the spoken confirmation question."""
        return vault_tools.confirm_pending()

    @function_tool()
    async def cancel_pending(self, context: RunContext) -> str:
        """Cancel the pending staged action (user said no or moved on)."""
        return vault_tools.cancel_pending()

    @function_tool()
    async def ask_claude(self, context: RunContext, question: str) -> str:
        """Escalate one question to Claude (slower, smarter, cloud). Call ONLY
        when the user explicitly asks to involve Claude.

        Args:
            question: The question to send to Claude.
        """
        return vault_tools.ask_claude(question)


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    llama_model = os.getenv("LLAMA_MODEL", "gemma-4-e2b")
    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    llama_api_key = os.getenv("LLAMA_API_KEY", "no-key-needed")

    stt_provider = os.getenv("STT_PROVIDER", "nemotron").lower()
    if stt_provider == "whisper":
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "Systran/faster-whisper-small"
    else:
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "nemotron-speech-streaming"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model = os.getenv("STT_MODEL", default_stt_model)
    stt_api_key = os.getenv("STT_API_KEY", "no-key-needed")

    tts_base_url = os.getenv("TTS_BASE_URL", "http://127.0.0.1:8880/v1")
    tts_voice = os.getenv("TTS_VOICE", "af_nova")
    tts_api_key = os.getenv("TTS_API_KEY", "no-key-needed")

    # AGENT_MODE=victoria points the LLM slot at the local Victoria bridge
    # (headless Claude, vault-scoped tools, real turn detection via LiveKit)
    # instead of Ollama/llama.cpp. Additive, not a replacement: default mode
    # is unchanged (the qwen2.5 vault agent), switch via .env.local + restart,
    # same pattern as every other config toggle in this repo.
    agent_mode = os.getenv("AGENT_MODE", "vault").lower()
    if agent_mode == "victoria":
        llama_base_url = os.getenv("VICTORIA_BRIDGE_URL", "http://127.0.0.1:8801/v1")
        llama_model = os.getenv("VICTORIA_MODEL", "claude-sonnet-5")
        llama_api_key = "no-key-needed"

    logger.info(
        "agent session: mode=%s stt=%s/%s llm=%s/%s tts=%s",
        agent_mode, stt_provider, stt_model, llama_base_url, llama_model, tts_base_url,
    )

    llm_extra_kwargs = {"extra_headers": {"X-Room-Name": ctx.room.name}} if agent_mode == "victoria" else {}

    session = AgentSession(
        stt=openai.STT(base_url=stt_base_url, model=stt_model, api_key=stt_api_key),
        llm=openai.LLM(base_url=llama_base_url, model=llama_model, api_key=llama_api_key,
                       **llm_extra_kwargs),
        # The model name selects the wire protocol the openai TTS plugin uses:
        # only {"tts-1", "tts-1-hd"} use the raw-audio-bytes stream that the
        # Kokoro server speaks. Any other name (e.g. "kokoro") routes the plugin
        # into the gpt-4o-mini-tts SSE reader, which parses Kokoro's binary audio
        # body as text, pushes zero frames, and raises "no audio frames were
        # pushed". Kokoro ignores the model field, so "tts-1" is purely a
        # protocol selector here.
        tts=openai.TTS(base_url=tts_base_url, model="tts-1", voice=tts_voice, api_key=tts_api_key,
                       speed=float(os.getenv("TTS_SPEED", "1.0"))),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        # Fast turn-taking (env-tunable): since interruption is supported, an
        # occasional early jump-in is cheap — the user just keeps talking and
        # the agent yields — so bias toward responding sooner. Defaults 0.5/6.0.
        min_endpointing_delay=float(os.getenv("MIN_ENDPOINTING_DELAY", "0.2")),
        max_endpointing_delay=float(os.getenv("MAX_ENDPOINTING_DELAY", "3.0")),
    )

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()

    if agent_mode == "victoria":
        # say() speaks directly (no LLM round trip) so this is instant —
        # unlike a real reply, which pays the full claude -p latency. Signals
        # the room is actually live before Alberto starts talking to silence.
        session.say("Victoria here — go ahead.")


if __name__ == "__main__":
    cli.run_app(server)
