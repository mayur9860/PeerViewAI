"""
LangGraph guardrail agent — Socratic critique pipeline.

Graph: critic_node → coach_node → reviewer_node
  - reviewer → coach (if unsafe & retry_count < 2)
  - reviewer → END  (if safe OR retry_count >= 2)

The Coach node is STRUCTURALLY FORBIDDEN from emitting replacement prose.
This is enforced by the Reviewer node's deterministic + LLM hybrid check.
"""

import re
from typing import Literal, TypedDict

from config import settings
from persona_prompts import get_persona_prompt


# ---------------------------------------------------------------------------
# State schema (spec §6.1)
# ---------------------------------------------------------------------------

class CritiqueState(TypedDict):
    section_text: str
    section_type: str
    archetype_label: str
    critic_notes: str
    draft_response: str
    review_verdict: Literal["safe", "unsafe"]
    review_reason: str
    retry_count: int
    transcript: list  # accumulated transcript turns


# ---------------------------------------------------------------------------
# Hardcoded fallback — used when reviewer fails after max retries
# ---------------------------------------------------------------------------

FALLBACK_RESPONSE = (
    "Let's take a step back and look at this section from a reviewer's perspective. "
    "What would a skeptical reader find most unconvincing here? "
    "Which specific claim feels most in need of stronger evidence or clearer reasoning? "
    "Consider re-reading your section with fresh eyes and identifying the single weakest argument."
)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def critic_node(state: CritiqueState) -> dict:
    """Critic node: analyze section text and produce structured craft gaps.

    Input: section_text + section_type + RAG results
    Output: critic_notes — concrete craft gaps
    """
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)

    # Fetch relevant RAG results for context
    rag_context = ""
    try:
        from rag_engine import search as rag_search
        results = await rag_search(
            query=f"common issues in {state['section_type']} section of academic paper",
            section_type=state["section_type"],
        )
        if results:
            rag_context = "\n\n---\nRelevant craft guidance:\n"
            for i, r in enumerate(results, 1):
                rag_context += f"\n{i}. {r.content}\n"
    except Exception:
        pass  # RAG failure is non-fatal

    system_prompt = """You are an expert academic writing critic. Analyze the given section of an academic paper and produce a STRUCTURED LIST of concrete craft gaps.

For each gap, provide:
- The specific location (paragraph/sentence number)
- The type of issue (overclaiming, unsupported assertion, missing context, structural problem, unclear contribution, etc.)
- A brief description of what's wrong

Do NOT provide suggestions or rewrites. Only identify problems.
Output a numbered list of craft gaps."""

    user_prompt = f"""Section type: {state['section_type']}
Writer archetype: {state['archetype_label']}

Section text:
{state['section_text']}
{rag_context}

Identify all craft gaps in this section:"""

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )

    critic_notes = response.text.strip()

    transcript = state.get("transcript", [])
    transcript.append({"role": "critic", "content": critic_notes})

    return {"critic_notes": critic_notes, "transcript": transcript}


async def coach_node(state: CritiqueState) -> dict:
    """Coach node: generate Socratic questions based on critic notes.

    Persona-styled using archetype_label. NEVER writes replacement sentences.
    """
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)

    persona = get_persona_prompt(state["archetype_label"])

    system_prompt = f"""{persona}

CRITICAL RULES — you MUST follow ALL of these:
1. Ask ONLY Socratic questions that guide the writer to fix issues themselves.
2. NEVER write or suggest replacement sentences.
3. NEVER provide a rewritten version of any part of the user's text.
4. NEVER put suggested text in quotation marks.
5. Reference specific parts of the writer's text by describing them, not by rewriting them.
6. Each question should help the writer see a specific weakness and think about how to address it.
7. If referencing statistical concepts or methodology, you may use LaTeX notation like $p < 0.05$ or $r^2$.

You will receive critic notes identifying craft gaps. Transform each gap into a probing Socratic question."""

    user_prompt = f"""Critic notes for this {state['section_type']} section:

{state['critic_notes']}

Original section text (for reference, do NOT rewrite any part):
{state['section_text'][:2000]}

Generate Socratic coaching questions for each craft gap identified above:"""

    # Add retry context if this is a retry
    if state["retry_count"] > 0:
        user_prompt += f"\n\nIMPORTANT: Your previous response was flagged as containing replacement text. Reason: {state.get('review_reason', 'unknown')}. You MUST ask ONLY questions this time. Do NOT include any suggested rewrites."

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        ),
    )

    draft_response = response.text.strip()

    transcript = state.get("transcript", [])
    transcript.append({"role": "coach", "content": draft_response})

    return {"draft_response": draft_response, "transcript": transcript}


async def reviewer_node(state: CritiqueState) -> dict:
    """Reviewer node: dual-check safety gate on coach's response.

    Check 1 (deterministic): Regex/heuristic — reject if response contains
    quotation-marked text longer than 8 words not attributed as user's quote.

    Check 2 (LLM): Gemini classifier — SAFE or UNSAFE with reason.

    If EITHER check flags it, verdict is "unsafe".
    """
    draft_response = state["draft_response"]
    review_verdict = "safe"
    review_reason = ""

    # ---- Check 1: Deterministic heuristic ----
    # Find all quoted strings longer than 8 words
    quoted_strings = re.findall(r'["\u201c\u201d]((?:[^"\u201c\u201d]|\n)+?)["\u201c\u201d]', draft_response)

    for quoted in quoted_strings:
        word_count = len(quoted.split())
        if word_count > 8:
            # Check if it's attributed as a quote from the user's own draft
            # Look for phrases like "you wrote", "your text says", "in your draft"
            context_before_quote = draft_response[:draft_response.find(quoted)][-100:]
            attribution_patterns = [
                r'you\s+wrote', r'your\s+(?:text|draft|section|paper)',
                r'the\s+(?:original|user)',  r'from\s+(?:the|your)',
                r'currently\s+(?:reads|says|states)',
            ]
            is_attributed = any(
                re.search(p, context_before_quote, re.IGNORECASE)
                for p in attribution_patterns
            )
            if not is_attributed:
                review_verdict = "unsafe"
                review_reason = f"Heuristic: Found unattributed quoted text longer than 8 words: '{quoted[:50]}...'"
                break

    # ---- Check 2: LLM classifier (only if heuristic passed) ----
    if review_verdict == "safe":
        try:
            from google import genai

            client = genai.Client(api_key=settings.gemini_api_key)

            classifier_prompt = f"""Examine this coaching response and determine if it contains any DRAFTED REPLACEMENT TEXT for the user's academic paper.

A response is UNSAFE if it:
- Contains suggested rewrites or replacement sentences
- Provides "improved" versions of the user's text
- Includes example sentences meant to replace the user's writing
- Drafts new prose that the user could copy-paste into their paper

A response is SAFE if it:
- Contains only questions
- References the user's text descriptively without rewriting it
- Provides general guidance without specific replacement text
- Uses short quoted phrases (≤8 words) from the user's own text for reference

Response to evaluate:
{draft_response}

Answer with exactly one word on the first line: SAFE or UNSAFE
Then provide one sentence explaining your reasoning."""

            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=classifier_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )

            verdict_text = response.text.strip()
            first_line = verdict_text.split('\n')[0].strip().upper()

            if "UNSAFE" in first_line:
                review_verdict = "unsafe"
                # Extract reason from second line
                lines = verdict_text.split('\n')
                review_reason = f"LLM classifier: {lines[1].strip() if len(lines) > 1 else 'Flagged as containing replacement text'}"

        except Exception as e:
            # If LLM check fails, let the heuristic result stand
            review_reason = f"LLM classifier unavailable: {e}"

    transcript = state.get("transcript", [])
    transcript.append({
        "role": "reviewer",
        "content": f"Verdict: {review_verdict}. {review_reason}" if review_reason else f"Verdict: {review_verdict}",
    })

    return {
        "review_verdict": review_verdict,
        "review_reason": review_reason,
        "transcript": transcript,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph():
    """Build the LangGraph StateGraph."""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(CritiqueState)

    # Add nodes
    graph.add_node("critic_node", critic_node)
    graph.add_node("coach_node", coach_node)
    graph.add_node("reviewer_node", reviewer_node)

    # Add edges
    graph.add_edge("critic_node", "coach_node")
    graph.add_edge("coach_node", "reviewer_node")

    # Conditional edge from reviewer
    def reviewer_router(state: CritiqueState) -> str:
        if state["review_verdict"] == "safe":
            return "end"
        if state["retry_count"] >= 2:
            return "end"
        return "retry"

    graph.add_conditional_edges(
        "reviewer_node",
        reviewer_router,
        {
            "end": END,
            "retry": "coach_node",
        },
    )

    # Entry point
    graph.set_entry_point("critic_node")

    return graph.compile()


# Lazy singleton
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Retry count incrementer — runs as part of the coach re-entry
# We patch the state before coach re-runs
# ---------------------------------------------------------------------------

# Override coach_node to increment retry on re-entry
_original_coach = coach_node


async def _coach_with_retry_increment(state: CritiqueState) -> dict:
    """Wrapper that increments retry_count when coach is re-entered from reviewer."""
    if state.get("review_verdict") == "unsafe":
        state["retry_count"] = state.get("retry_count", 0) + 1
    return await _original_coach(state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_critique_agent(
    section_text: str,
    section_type: str,
    archetype_label: str,
) -> dict:
    """Run the full critique agent pipeline.

    Returns:
        {
            "transcript": list of {"role": str, "content": str},
            "outcome": "safe" | "flagged_and_retried" | "abandoned",
            "final_message": str
        }
    """
    graph = _get_graph()

    initial_state: CritiqueState = {
        "section_text": section_text,
        "section_type": section_type,
        "archetype_label": archetype_label,
        "critic_notes": "",
        "draft_response": "",
        "review_verdict": "safe",
        "review_reason": "",
        "retry_count": 0,
        "transcript": [],
    }

    # Run the graph
    final_state = await graph.ainvoke(initial_state)

    # Determine outcome
    if final_state["review_verdict"] == "safe":
        if final_state["retry_count"] > 0:
            outcome = "flagged_and_retried"
        else:
            outcome = "safe"
        final_message = final_state["draft_response"]
    else:
        # Max retries exhausted with unsafe verdict — use fallback
        outcome = "abandoned"
        final_message = FALLBACK_RESPONSE
        final_state["transcript"].append({
            "role": "system",
            "content": f"Safety fallback activated after {final_state['retry_count']} retries. Using templated Socratic questions.",
        })

    return {
        "transcript": final_state["transcript"],
        "outcome": outcome,
        "final_message": final_message,
    }
