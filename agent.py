import asyncio
import logging
import os

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, llm
from livekit.plugins import groq, deepgram

from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION, SESSION_GREETING
from rag.qdrant_service import check_qdrant_connection
from rag.retriever import format_context, retrieve_context
load_dotenv(override=True)

logger = logging.getLogger(__name__)


# -----------------------------
# Assistant Definition
# -----------------------------
class Assistant(Agent):
    def __init__(self, *, rag_enabled: bool):
        self._rag_enabled = rag_enabled
        super().__init__(
            instructions=AGENT_INSTRUCTION,
            llm=groq.LLM(
                model="openai/gpt-oss-120b",
                temperature=0.7,
            ),
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        if not self._rag_enabled:
            return

        query = new_message.text_content or ""
        if not query.strip():
            return
        if not _should_use_rag(query):
            return

        chunks = await retrieve_context(query)
        logger.info("RAG retrieved %d documents for query", len(chunks))
        if not chunks:
            return

        retrieved_context = format_context(chunks)

        system_prompt = (
            "You are a helpful voice assistant.\n\n"
            "Use the following context to answer the user's question.\n\n"
            "Context:\n"
            f"{retrieved_context}\n\n"
            "User Question:\n"
            f"{query}\n\n"
            "Answer conversationally since the response will be spoken.\n"
            "Keep responses concise and natural for spoken conversation. Avoid long paragraphs."
        )

        turn_ctx.add_message(
            role="system",
            content=system_prompt,
            created_at=new_message.created_at - 0.001,
        )


def _rag_config_ready() -> bool:
    """Return True if required RAG configuration is present."""
    missing: list[str] = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("QDRANT_URL"):
        missing.append("QDRANT_URL")
    if missing:
        logger.warning("RAG disabled: missing configuration: %s", ", ".join(missing))
        return False
    return True


def _should_use_rag(query: str) -> bool:
    """Heuristic to decide whether a query needs RAG retrieval."""
    text = " ".join(query.lower().strip().split())
    if not text:
        return False
    if "?" in text:
        return True

    interrogatives = (
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "which",
        "whose",
        "whom",
    )
    if text.startswith(interrogatives):
        return True

    keywords = (
        "tell me",
        "explain",
        "details",
        "information",
        "policy",
        "guide",
        "documentation",
        "doc",
        "docs",
        "help",
        "knowledge",
        "reference",
    )
    if any(key in text for key in keywords):
        return True

    return False


async def _initialize_rag() -> bool:
    """Initialize RAG dependencies and validate connectivity."""
    if not _rag_config_ready():
        return False
    try:
        await asyncio.to_thread(check_qdrant_connection)
    except Exception as exc:
        logger.warning("RAG disabled: Qdrant connection failed: %s", exc)
        return False
    return True


# -----------------------------
# Job Entrypoint
# -----------------------------
async def entrypoint(ctx: agents.JobContext):

    await ctx.connect()
    rag_enabled = await _initialize_rag()

    session = AgentSession(
        llm=groq.LLM(
            model="openai/gpt-oss-120b",
            temperature=0.7,
        ),

        stt=deepgram.STT(
            language="en-US",
            model="nova-3",
            detect_language=False,
            interim_results=True,
            punctuate=True,
        ),

        tts=deepgram.TTS(model="aura-2-andromeda-en"),
        turn_detection="stt",
        min_endpointing_delay=0.07,
    )

    assistant = Assistant(rag_enabled=rag_enabled)

    await session.start(
        room=ctx.room,
        agent=assistant,
    )

    await session.generate_reply(
        instructions=(
            f"{SESSION_INSTRUCTION}\n\n"
            "Use this exact opening and nothing else:\n"
            f"{SESSION_GREETING}"
        )
    )


# -----------------------------
# Run Worker
# -----------------------------
if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
