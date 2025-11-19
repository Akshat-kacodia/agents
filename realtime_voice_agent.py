"""
Realtime Voice Agent with Intelligent Interruption Handling
===========================================================

- LiveKit Agent worker (server-side)
- Deepgram STT  + OpenAI LLM + ElevenLabs TTS
- External InterruptHandler to handle fillers & real interruptions
"""

import os
import logging
import asyncio
from typing import Optional
from interrupt_handler import InterruptHandler
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    inference
)
from livekit.plugins import silero, deepgram, elevenlabs



load_dotenv()

def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y"}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required but not set")
    return value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("realtime_voice_agent")

# Validate core keys (this ensures we actually use the 6 env variables)
OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
DEEPGRAM_API_KEY = _require_env("DEEPGRAM_API_KEY")
ELEVEN_API_KEY = _require_env("ELEVEN_API_KEY")
LIVEKIT_URL = _require_env("LIVEKIT_URL")
LIVEKIT_API_KEY = _require_env("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = _require_env("LIVEKIT_API_SECRET")

logger.info("LiveKit URL: %s", LIVEKIT_URL)

# ---------------------------------------------------------------------
# Global InterruptHandler instance (configurable via env)
# ---------------------------------------------------------------------

interrupt_handler = InterruptHandler(
    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")),
    enable_contextual_analysis=_env_flag("ENABLE_CONTEXTUAL_ANALYSIS", True),
    enable_multi_language=_env_flag("ENABLE_MULTI_LANGUAGE", True),
    log_events=_env_flag("INTERRUPT_LOG_EVENTS", True),
)


@function_tool
async def get_time(context: RunContext) -> dict:
    """Return the current time (demonstration tool)."""
    from datetime import datetime

    current_time = datetime.now().strftime("%H:%M:%S")
    logger.info("get_time tool called: %s", current_time)
    return {"time": current_time}


async def entrypoint(ctx: JobContext) -> None:
    """
    Main entrypoint for this LiveKit worker. It:

    - Connects to LiveKit
    - Creates Agent + AgentSession
    - Wires event handlers to InterruptHandler
    """
    logger.info("=" * 72)
    logger.info("Starting realtime voice agent with interruption handling")
    logger.info("=" * 72)

    # Connect to LiveKit job context (room, participants, etc.)
    await ctx.connect()
    logger.info("Connected to room: %s", ctx.room.name)

    # Runtime custom fillers from env (comma-separated)
    custom_fillers_en = [w.strip() for w in os.getenv("FILLER_WORDS_EN", "").split(",") if w.strip()]
    custom_fillers_hi = [w.strip() for w in os.getenv("FILLER_WORDS_HI", "").split(",") if w.strip()]
    if custom_fillers_en:
        interrupt_handler.add_custom_fillers("en", custom_fillers_en)
    if custom_fillers_hi:
        interrupt_handler.add_custom_fillers("hi", custom_fillers_hi)

    # -----------------------------------------------------------------
    # Build Agent (LLM instructions)
    # -----------------------------------------------------------------
    agent = Agent(
        instructions=(
            "You are an intelligent, friendly real-time voice assistant running inside a LiveKit room.\n\n"
            "Your primary goal is to feel like a calm, attentive human conversation partner:\n"
            "- Speak in natural, spoken language with contractions (I'm, you'll, let's) and a warm tone.\n"
            "- Acknowledge what the user says, then respond with clear, helpful answers.\n"
            "- When helpful, briefly explain your reasoning, but avoid sounding like a technical manual.\n"
            "- Ask simple follow-up questions when the user’s request is vague or could mean multiple things.\n\n"
            "FILLERS & HESITATION:\n"
            "- Users often say things like “uh”, “umm”, “hmm”, “like”, “you know” while thinking.\n"
            "- Treat these as natural hesitation, not as commands or separate requests.\n"
            "- Do NOT change topic or cut yourself off just because you hear a few filler sounds.\n\n"
            "INTERRUPTIONS:\n"
            "- If the user clearly says words like “wait”, “stop”, “hold on”, “ruk”, “ruko”, or other strong\n"
            "  stop/interrupt phrases, treat that as a real interruption.\n"
            "- When you’re interrupted, quickly wrap up or stop your current thought and listen instead of\n"
            "  continuing to talk over the user.\n\n"
            "CONVERSATION STYLE:\n"
            "- Default to concise answers: usually 2–4 spoken sentences unless the user asks for more detail.\n"
            "- If the user seems curious or explicitly asks for details (“explain more”, “walk me through it”),\n"
            "  give a deeper, step-by-step explanation.\n"
            "- Use first-person (“I”) and second-person (“you”), and occasionally reflect emotions when appropriate\n"
            "  (e.g., “That sounds frustrating” or “Nice, that’s exciting!”) without being overly dramatic.\n"
            "- Never mention system prompts, API keys, or internal tooling. Present yourself simply as an assistant.\n\n"
            "Overall, your behavior should make the user feel like they are talking to a thoughtful human who\n"
            "listens carefully, doesn’t panic about fillers, and respects clear interruptions."
        ),
        tools=[get_time],
    )

    logger.info("Agent object created")

    # -----------------------------------------------------------------
    # Build AgentSession (STT + VAD + LLM + TTS)
    # -----------------------------------------------------------------

    # Deepgram STT
    stt_engine = deepgram.STT(model="nova-3")

    # OpenAI LLM
    # llm_engine = openai.LLM(model="gpt-3.5-turbo")

    # ElevenLabs TTS
    tts_engine = elevenlabs.TTS(
        # If ELEVEN_VOICE_ID not set, use provider default
        voice_id=os.getenv("ELEVEN_VOICE_ID"),
    )

    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.5,
            min_silence_duration=0.5,
            prefix_padding_duration=0.2,
        ),
        stt=stt_engine,
        llm=inference.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
    )

    # -----------------------------------------------------------------
    # Wire events -> InterruptHandler
    # -----------------------------------------------------------------

    @session.on("agent_speech_started")
    def _on_agent_speech_started() -> None:
        interrupt_handler.set_agent_speaking(True)
        logger.debug("Agent speech started")

    @session.on("agent_speech_ended")
    def _on_agent_speech_ended() -> None:
        interrupt_handler.set_agent_speaking(False)
        logger.debug("Agent speech ended")

    @session.on("user_speech_committed")
    def _on_user_speech_committed(transcript: str, confidence: float = 0.9) -> None:
        """
        This callback runs whenever the session commits a user utterance.
        We push it through InterruptHandler to decide how to treat it.

        - IGNORE    -> drop (likely filler while agent is speaking)
        - INTERRUPT -> call session.interrupt() and stop current speech
        - REGISTER  -> normal speech; the agent pipeline can continue
        """
        async def _process() -> None:
            try:
                result = await interrupt_handler.process_speech_event(
                    transcript=transcript,
                    confidence=confidence,
                    language="en",  # could be dynamic in multi-language setup
                )
                action = result["action"]
                reason = result["reason"]

                if action == "IGNORE":
                    logger.info("Filler/noise ignored: '%s' (%s)", transcript, reason)
                    # We *do not* call interrupt, and we do not alter the pipeline.

                elif action == "INTERRUPT":
                    logger.info("Valid interruption: '%s' (%s)", transcript, reason)
                    # Interrupt current agent speech; AgentSession handles chat context updates.
                    try:
                        await session.interrupt(force=False)
                    except Exception as e:
                        logger.exception("Failed to interrupt session: %s", e)

                elif action == "REGISTER":
                    logger.info("Registering user speech: '%s' (%s)", transcript, reason)
                    # Normal AgentSession turn-detection will use this as part of conversation.
            except Exception as e:
                logger.exception("Error in interruption handler: %s", e)

        asyncio.create_task(_process())

    # -----------------------------------------------------------------
    # Start agent session
    # -----------------------------------------------------------------
    await session.start(agent=agent, room=ctx.room)

    # Initial greeting
    await session.generate_reply(
        instructions=(
            "Greet the user in a warm, human way. Briefly introduce yourself as a real-time voice assistant "
            "who can chat naturally, answer questions, and help with tasks.\n"
            "- Explicitly do not mention that you understand fillers like “umm”, “uh”, or “hmm” and you won’t get "
            "confused if they pause or think out loud.\n"
            "- Also mention that if they say clear words like “wait” or “stop”, you’ll immediately pause and listen.\n"
            "- End your greeting with a simple, inviting question like “So, what’s on your mind right now?” "
            "or “How can I help you today?”"
        )
        
    )


    # -----------------------------------------------------------------
    # Periodically log stats
    # -----------------------------------------------------------------
    try:
        while True:
            await asyncio.sleep(120)
            stats = interrupt_handler.get_statistics()
            logger.info(
                "Stats: total=%d ignored=%d interrupts=%d registered=%d",
                stats["total_events"],
                stats["ignored_fillers"],
                stats["valid_interrupts"],
                stats["registered_speech"],
            )
    except asyncio.CancelledError:
        logger.info("Agent session shutting down...")
        interrupt_handler.print_summary()
        interrupt_handler.export_event_log("interruption_log.json")


if __name__ == "__main__":
    # This spins up the worker and binds it to your LiveKit project
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
