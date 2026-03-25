"""
Multi-Agent AI Pipeline
=======================
A planner-driven pipeline that routes queries through specialist agents
in sequence, each building on the previous one's output.

Supports Claude, Gemini, and Mistral via LiteLLM.

Usage:
    python main.py "your question"
    python main.py --gemini "your question"
    python main.py --mistral "your question"
"""

import io
import json
import os
import sys

import litellm
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

PROVIDERS = {
    "--claude":  "anthropic/claude-sonnet-4-20250514",
    "--gemini":  "gemini/gemini-2.0-flash",
    "--mistral": "mistral/mistral-small-latest",
}

AVAILABLE_AGENTS = [
    "game_theory",
    "first_principles",
    "assumption_questioner",
    "math",
    "report_writer",
]

MAX_RETRIES = 3


def select_model():
    for flag in PROVIDERS:
        if flag in sys.argv:
            sys.argv.remove(flag)
            return PROVIDERS[flag]
    return PROVIDERS["--claude"]


MODEL = select_model()


# ── Core LLM call ──────────────────────────────────────────────────────────────

def call_llm(system_prompt, user_message):
    try:
        response = litellm.completion(
            model=MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {e}"


# ── Shared message builder ─────────────────────────────────────────────────────

def _build_message(query, prior_context, follow_up):
    """Combines the original query with accumulated context when available."""
    if prior_context:
        return (
            f"Original question: {query}\n\n"
            f"Prior analysis:\n{prior_context}\n\n"
            f"{follow_up}"
        )
    return query


# ── Planner agent ──────────────────────────────────────────────────────────────

def planner_agent(query):
    """
    Analyzes the query and returns an ordered list of agent names.
    The list forms the execution plan for the pipeline.
    """
    system_prompt = f"""
    You are a query planner. Analyze the user's question and decide which
    agents to run and in what order.

    Available agents:
    - game_theory         : strategic/competitive analysis, incentives, equilibria
    - first_principles    : strip assumptions, reason from base facts only
    - assumption_questioner: challenge hidden assumptions, find blind spots
    - math                : generate + execute Python code (only for calculations)
    - report_writer       : synthesize all prior analysis into a readable summary

    Rules:
    1. Return ONLY a JSON array, e.g. ["first_principles", "report_writer"]
    2. Use 1–4 agents. Simple math → ["math"]. Complex → chain 2–4.
    3. report_writer should always be last when included.

    Examples:
    "What is 25 * 48?"                          → ["math"]
    "Should a startup lower prices to compete?" → ["first_principles", "game_theory", "report_writer"]
    "Why do people think AI will take all jobs?"→ ["assumption_questioner", "first_principles", "report_writer"]
    """

    result = call_llm(system_prompt, query)

    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        plan = json.loads(cleaned)
        plan = [a for a in plan if a in AVAILABLE_AGENTS]
        return plan if plan else ["first_principles", "report_writer"]
    except (json.JSONDecodeError, TypeError):
        return ["first_principles", "report_writer"]


# ── Specialist agents ──────────────────────────────────────────────────────────

def game_theory_agent(query, prior_context=None):
    system = """
    You are a game theory strategist.
    Identify players, strategies, incentives, payoffs, and equilibria.
    Be specific and concise — under 200 words.
    """
    return call_llm(system, _build_message(
        query, prior_context,
        "Analyze from a game theory perspective, building on the prior analysis."
    ))


def first_principles_agent(query, prior_context=None):
    system = """
    You are a first principles thinker.
    Identify base facts, flag assumptions, rebuild reasoning from the ground up.
    Under 200 words.
    """
    return call_llm(system, _build_message(
        query, prior_context,
        "Apply first principles thinking. Build on or challenge the prior analysis."
    ))


def assumption_questioner_agent(query, prior_context=None):
    system = """
    You are a critical thinker.
    Identify hidden assumptions, challenge each one, highlight blind spots.
    Under 200 words.
    """
    return call_llm(system, _build_message(
        query, prior_context,
        "Challenge the assumptions in this analysis. What is being taken for granted?"
    ))


def report_writer_agent(query, prior_context=None):
    system = """
    You are a report writer. Synthesize prior analysis into a clear, conversational
    summary — bottom line first, then supporting reasoning. No jargon.
    """
    return call_llm(system, _build_message(
        query, prior_context,
        "Write a clean, conversational report summarizing all of this."
    ))


# Math agent with self-correction loop ─────────────────────────────────────────

def _execute_code(code):
    """Runs Python code in an isolated namespace, capturing printed output."""
    try:
        old_stdout, sys.stdout = sys.stdout, io.StringIO()
        exec(code, {})
        output = sys.stdout.getvalue().strip()
        sys.stdout = old_stdout
        return True, output
    except Exception as e:
        sys.stdout = old_stdout
        return False, str(e)


def math_agent(query, prior_context=None):
    """
    Generates Python code for the calculation, executes it, and returns the result.
    If execution fails, asks the LLM to fix the code and retries (up to MAX_RETRIES).
    """
    system = """
    You are a Python math solver.
    Write code that prints the final answer with print().
    Return ONLY raw Python code — no markdown, no explanations.
    """
    code = call_llm(system, _build_message(
        query, prior_context,
        "Write Python code to calculate whatever numbers are needed here."
    ))

    for attempt in range(MAX_RETRIES):
        success, result = _execute_code(code)
        if success:
            return f"Calculation result: {result}"

        print(f"  Math error (attempt {attempt + 1}): {result}")
        if attempt < MAX_RETRIES - 1:
            code = call_llm(
                "You are a Python debugger. Fix the code and return ONLY corrected code, no markdown.",
                f"Problem: {query}\n\nBroken code:\n{code}\n\nError: {result}\n\nFix it.",
            )

    return f"Math failed after {MAX_RETRIES} attempts. Last error: {result}"


# ── Agent registry ─────────────────────────────────────────────────────────────

AGENT_REGISTRY = {
    "game_theory":          game_theory_agent,
    "first_principles":     first_principles_agent,
    "assumption_questioner": assumption_questioner_agent,
    "math":                 math_agent,
    "report_writer":        report_writer_agent,
}


# ── Pipeline executor ──────────────────────────────────────────────────────────

def run_pipeline(query, plan):
    """
    Executes agents in the order defined by the plan.

    Each agent receives:
      - The original query (so context of the question is never lost)
      - All accumulated output from prior agents (so each step builds on the last)

    Errors in individual agents are logged but do not stop the pipeline.
    """
    context = ""
    completed = []

    for i, agent_name in enumerate(plan, 1):
        inputs = "query" if i == 1 else f"query + {' + '.join(completed)}"
        print(f"[{i}/{len(plan)}] {agent_name}  ({inputs})")

        fn = AGENT_REGISTRY.get(agent_name)
        if not fn:
            print(f"  Unknown agent '{agent_name}', skipping.")
            continue

        try:
            result = fn(query, prior_context=context if i > 1 else None)

            if result is None:
                print(f"  {agent_name} returned no output, skipping.")
                context += f"\n\n--- {agent_name} ---\n(no output returned)"
                continue

            if result.startswith("[ERROR]"):
                print(f"  Error: {result}")
                context += f"\n\n--- {agent_name} ---\n(error: {result})"
            else:
                print(f"  -> {result[:80].replace(chr(10), ' ')}...")
                context += f"\n\n--- {agent_name} ---\n{result}"
                completed.append(agent_name)

        except Exception as e:
            print(f"  {agent_name} crashed: {e}")
            context += f"\n\n--- {agent_name} ---\n(crashed: {e})"

    return context


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Ask me anything: ")

    print(f"\nQuery : {query}")
    print(f"Model : {MODEL}")

    # Step 1 — planner decides which agents to run and in what order
    plan = planner_agent(query)
    print(f"Plan  : {' -> '.join(plan)}\n")

    # Step 2 — run the pipeline, agents chain their outputs
    output = run_pipeline(query, plan)

    # Step 3 — print the last agent's section as the final answer
    last = plan[-1]
    parts = output.split(f"--- {last} ---")
    print(f"\nANSWER:\n{parts[-1].strip() if len(parts) > 1 else output.strip()}")


if __name__ == "__main__":
    main()
