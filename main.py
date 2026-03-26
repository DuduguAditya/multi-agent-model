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

import argparse
import json
import os
import subprocess
import sys
from typing import List, Literal

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

PROVIDERS = {
    "claude":   "anthropic/claude-sonnet-4-20250514",
    "gemini":   "gemini/gemini-2.0-flash",
    "mistral":  "mistral/mistral-small-latest",
}

AVAILABLE_AGENTS = [
    "game_theory",
    "first_principles",
    "assumption_questioner",
    "math",
    "report_writer",
]

MAX_RETRIES = 3

# Pydantic model for type-safe planner output validation
AgentName = Literal["game_theory", "first_principles", "assumption_questioner", "math", "report_writer"]

class ExecutionPlan(BaseModel):
    agents: List[AgentName]


# ── Core LLM call ──────────────────────────────────────────────────────────────

def call_llm(system_prompt, user_message, model):
    try:
        response = litellm.completion(
            model=model,
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

def planner_agent(query, model):
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

    result = call_llm(system_prompt, query, model)

    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        # Wrap the array in a dict so Pydantic can validate it
        plan = ExecutionPlan(agents=json.loads(cleaned))
        return plan.agents
    except (json.JSONDecodeError, TypeError, ValidationError):
        return ["first_principles", "report_writer"]


# ── Specialist agents ──────────────────────────────────────────────────────────

def game_theory_agent(query, model, prior_context=None):
    system = """
    You are a game theory strategist.
    Identify players, strategies, incentives, payoffs, and equilibria.
    Be specific and concise — under 200 words.
    """
    return call_llm(system, _build_message(
        query, prior_context,
        "Analyze from a game theory perspective, building on the prior analysis."
    ), model)


def first_principles_agent(query, model, prior_context=None):
    system = """
    You are a first principles thinker.
    Identify base facts, flag assumptions, rebuild reasoning from the ground up.
    Under 200 words.
    """
    return call_llm(system, _build_message(
        query, prior_context,
        "Apply first principles thinking. Build on or challenge the prior analysis."
    ), model)


def assumption_questioner_agent(query, model, prior_context=None):
    system = """
    You are a critical thinker.
    Identify hidden assumptions, challenge each one, highlight blind spots.
    Under 200 words.
    """
    return call_llm(system, _build_message(
        query, prior_context,
        "Challenge the assumptions in this analysis. What is being taken for granted?"
    ), model)


def report_writer_agent(query, model, prior_context=None):
    system = """
    You are a report writer. Synthesize prior analysis into a clear, conversational
    summary — bottom line first, then supporting reasoning. No jargon.
    """
    return call_llm(system, _build_message(
        query, prior_context,
        "Write a clean, conversational report summarizing all of this."
    ), model)


# Math agent with self-correction loop ─────────────────────────────────────────

def _execute_code(code):
    """Runs generated code in a sandboxed subprocess, isolated from the main process."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return True, proc.stdout.strip()
        return False, proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Code execution timed out after 10 seconds."
    except Exception as e:
        return False, str(e)


def math_agent(query, model, prior_context=None):
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
    ), model)

    for attempt in range(MAX_RETRIES):
        success, result = _execute_code(code)
        if success:
            return f"Calculation result: {result}"

        print(f"  Math error (attempt {attempt + 1}): {result}")
        if attempt < MAX_RETRIES - 1:
            code = call_llm(
                "You are a Python debugger. Fix the code and return ONLY corrected code, no markdown.",
                f"Problem: {query}\n\nBroken code:\n{code}\n\nError: {result}\n\nFix it.",
                model,
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

def run_pipeline(query, plan, model):
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
            result = fn(query, model, prior_context=context if i > 1 else None)

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
    parser = argparse.ArgumentParser(description="Multi-Agent AI Pipeline")
    parser.add_argument("query", nargs="?", default=None, help="Your question")
    parser.add_argument("--model", choices=PROVIDERS.keys(), default="claude", help="Model to use (default: claude)")
    args = parser.parse_args()

    model = PROVIDERS[args.model]
    print(f"Model : {model}  |  type 'quit' to exit\n")

    # Conversation history — stores (question, answer) pairs across turns
    history = []

    # Keep the session alive for follow-up questions
    while True:
        query = args.query or input("Ask me anything: ")
        args.query = None  # clear after first use so loop prompts on next turn

        if query.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # /btw — quick side question, bypasses the pipeline and history
        if query.strip().lower().startswith("/btw"):
            side_question = query.strip()[4:].strip()
            print("\n[btw]")
            answer = call_llm("You are a helpful assistant. Answer concisely.", side_question, model)
            print(f"{answer}\n")
            continue

        # Prepend prior conversation so agents have memory of past exchanges
        if history:
            prior = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
            full_query = f"Previous conversation:\n{prior}\n\nNew question: {query}"
        else:
            full_query = query

        print(f"\nQuery : {query}")

        # Step 1 — planner decides which agents to run and in what order
        plan = planner_agent(full_query, model)
        print(f"Plan  : {' -> '.join(plan)}\n")

        # Step 2 — run the pipeline, agents chain their outputs
        output = run_pipeline(full_query, plan, model)

        # Step 3 — extract and print the last agent's section as the final answer
        last = plan[-1]
        parts = output.split(f"--- {last} ---")
        answer = parts[-1].strip() if len(parts) > 1 else output.strip()
        print(f"\nANSWER:\n{answer}\n")

        # Store this exchange for future turns
        history.append((query, answer))


if __name__ == "__main__":
    main()
