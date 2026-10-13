"""Chainlit Agentic Chat UI for AstroPilote (Google ADK + Gemma 4 + Hybrid RAG).

Usage:
    uv run chainlit run src/astropilote/app/chainlit_app.py -w
"""

from __future__ import annotations

import chainlit as cl

from astropilote.agent.graph import run_agent
from astropilote.config import settings


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    """Provide one-click starter prompts in the Chainlit interface."""
    return [
        cl.Starter(
            label="Fontainebleau Session Plan (Oct 17 · 80mm APO)",
            message=(
                "Saturday night, October 17, 2026, I will be in Fontainebleau with an 80 mm refractor "
                "(480 mm focal length). What targets should I observe or image?"
            ),
        ),
        cl.Starter(
            label="Chamonix Narrowband Cygnus (Ha/OIII)",
            message=(
                "I am in Chamonix on August 15, 2026 with a wide-field astrophotography rig (250 mm focal length) "
                "and a dual-band Ha/OIII filter. Which emission nebulae or supernova remnants in Cygnus can I shoot tonight?"
            ),
        ),
        cl.Starter(
            label="Andromeda Satellite Galaxies (M31 / M32 / M110)",
            message=(
                "What are the two brightest satellite galaxies of the Andromeda Galaxy (M31) that can be seen "
                "in the same wide field of view?"
            ),
        ),
        cl.Starter(
            label="Out-of-Corpus Trap (IC 1101 -> Abstain)",
            message=(
                "What can you observe of the supergiant elliptical galaxy IC 1101 in Abell 2029 with an amateur telescope?"
            ),
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    """Send welcome banner with active Google ADK + Gemma 4 configuration."""
    await cl.Message(
        content=(
            f"### 🔭 AstroPilote — Google ADK & Hybrid RAG Observatory Copilot\n"
            f"- **Agent Runtime:** `google-adk` (`LlmAgent` + Pure-Python Verifier)\n"
            f"- **LLM Model:** `{settings.llm_model}` (`{settings.llm_provider}`)\n"
            f"- **Hybrid Retrieval:** `bm25s` + `Qdrant` (`{settings.embedding_model}`) + Cross-Encoder (`{settings.reranker_model}`)"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Execute the Google ADK pipeline and stream tool steps + retrieved sources in Chainlit."""
    answer, state, eval_row = run_agent(message.content, question_id="chainlit_turn")

    # Render each tool call as an expandable Chainlit Step
    for tc in eval_row.tool_calls:
        async with cl.Step(name=f"Tool: {tc.name} ({tc.duration_s * 1000:.1f} ms)", type="tool") as step:
            step.input = str(tc.arguments)
            step.output = f"Completed in {tc.duration_s:.4f}s"

    # Build side-panel source elements for retrieved chunks
    elements: list[cl.Text] = []
    for ch in state.get("retrieved", [])[:6]:
        elements.append(
            cl.Text(
                name=f"[{ch.doc_id}] {ch.title}",
                content=f"Section: {ch.section} (Score: {ch.score})\n\n{ch.text}",
                display="side",
            )
        )

    plan_md = ""
    if answer.plan:
        rows = [
            "| Window | Target | Alt / Dir | Difficulty | Notes | Source |",
            "|---|---|---|---|---|---|",
        ]
        for item in answer.plan:
            w_str = f"{item.start.strftime('%H:%M')}–{item.end.strftime('%H:%M')}"
            rows.append(
                f"| `{w_str}` | **{item.name}** | `{item.altitude_deg:.1f}° {item.direction}` | "
                f"`{item.difficulty}` | {item.imaging_notes or ''} | `{', '.join(item.sources)}` |"
            )
        plan_md = "\n\n#### Scheduled Night Plan\n" + "\n".join(rows) + "\n\n"

    footer = (
        f"\n\n---\n*Outcome: `{eval_row.stop_reason.upper()}` · Verified: `{answer.verified}` · "
        f"Latency: `{eval_row.latency_s:.2f}s` · Tokens: `{eval_row.tokens_in}/{eval_row.tokens_out}`*"
    )

    await cl.Message(
        content=f"{plan_md}{answer.answer}{footer}",
        elements=elements,
    ).send()
