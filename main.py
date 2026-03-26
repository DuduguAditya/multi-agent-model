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
from pydantic import BaseModel, ValidationError, Field

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
    agents: List[AgentName] = Field(description="list of agents names")


# ── Core LLM call ──────────────────────────────────────────────────────────────

def call_llm(system_prompt, user_message, model, chat_history=[], structured_class=None):
    
    messages = [{"role": "system", "content": system_prompt}]

    # Adding conversation chat_history to messages var
    for user_msg, ai_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ai_msg})

    # Adding user query
    messages.append({"role": "user",   "content": user_message})

    try:
        response = litellm.completion(
            model=model,
            max_tokens=2048,
            messages=messages,
            response_format=structured_class
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"[ERROR] {e}"


# ── Shared message builder ─────────────────────────────────────────────────────

def _build_message(query, agent_ctx, follow_up):
    """Combines the original query with accumulated context when available."""
    if agent_ctx:
        return (
            f"Prior analysis:\n{agent_ctx}\n\n"
            f"Original question: {query}\n\n"
            f"{follow_up}"
        )
    return query


# ── Planner agent ──────────────────────────────────────────────────────────────

def planner_agent(query, model, chat_history):
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

    result = call_llm(system_prompt, query, model, chat_history, structured_class=ExecutionPlan)

    try:
        plan: ExecutionPlan = ExecutionPlan.model_validate_json(result)
        return plan.agents
    except ValidationError:
        return ["first_principles", "report_writer"]


# ── Specialist agents ──────────────────────────────────────────────────────────

def game_theory_agent(query, model, agent_ctx=None, chat_history=[]):
    system = """
    You are a game theory strategist.
    Identify players, strategies, incentives, payoffs, and equilibria.
    Be specific and concise — under 200 words.
    """
    return call_llm(system, _build_message(
        query, agent_ctx,
        "Analyze from a game theory perspective, building on the prior analysis."
    ), model, chat_history)


def first_principles_agent(query, model, agent_ctx=None, chat_history=[]):
    system = """
    You are a first principles thinker.
    Identify base facts, flag assumptions, rebuild reasoning from the ground up.
    Under 200 words.
    """
    return call_llm(system, _build_message(
        query, agent_ctx,
        "Apply first principles thinking. Build on or challenge the prior analysis."
    ), model, chat_history)


def assumption_questioner_agent(query, model, agent_ctx=None, chat_history=[]):
    system = """
    You are a critical thinker.
    Identify hidden assumptions, challenge each one, highlight blind spots.
    Under 200 words.
    """
    return call_llm(system, _build_message(
        query, agent_ctx,
        "Challenge the assumptions in this analysis. What is being taken for granted?"
    ), model, chat_history)


def report_writer_agent(query, model, agent_ctx=None, chat_history=[]):
    system = """
    You are a report writer. Synthesize prior analysis into a clear, conversational
    summary — bottom line first, then supporting reasoning. No jargon.
    """
    return call_llm(system, _build_message(
        query, agent_ctx,
        "Write a clean, conversational report summarizing all of this."
    ), model, chat_history)


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


def math_agent(query, model, agent_ctx=None, chat_history=[]):
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
        query, agent_ctx,
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
                model, chat_history
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

def run_pipeline(query, agent_name_list, model, chat_history):
    """
    Executes agents in the order defined by the agent_name_list.

    Each agent receives:
      - The original query (so context of the question is never lost)
      - All accumulated output from prior agents (so each step builds on the last)

    Errors in individual agents are logged but do not stop the pipeline.
    """

    agent_results = []

    for i, agent_name in enumerate(agent_name_list, 1):
        print(f"[{i}/{len(agent_name_list)}] ---> {agent_name}")

        fn = AGENT_REGISTRY.get(agent_name)
        if not fn:
            print(f"  Unknown agent '{agent_name}', skipping.")
            continue

        try:
            agent_ctx = ''.join([f'--- start of {a_name} analysis ---\n{a_result}\n --- end of {a_name} analysis ---\n'   for a_name, a_result in agent_results])
            print(f'===================>agent_ctx:\n {agent_ctx} \n<<=========')
            result = fn(query, model, agent_ctx, chat_history)

            if result is None:
                print(f"  {agent_name} returned no output, skipping.")
                agent_results.append((agent_name, f'No output result from agent: {agent_name}.'))
                continue

            if result.startswith("[ERROR]"):
                print(f"  Error: {result}")
                agent_results.append((agent_name, f"Error: {result}"))
            else:
                print(f"  -> {result[:80].replace(chr(10), ' ')}...")
                agent_results.append((agent_name, result))

        except Exception as e:
            print(f"  {agent_name} crashed: {e}")
            agent_results.append((agent_name, f"Crashed, with error message: {e}"))

    return agent_results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent AI Pipeline")
    parser.add_argument("--model", choices=PROVIDERS.keys(), default="claude", help="Model to use (default: claude)")
    args = parser.parse_args()

    model = PROVIDERS[args.model]
    print(f"Model : {model}  |  type 'quit' to exit\n")

    # Conversation chat_history — stores (question, answer) pairs across turns
    chat_history = []

    # Keep the session alive for follow-up questions
    while True:
        query = input("Ask me anything...\n")

        if query.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            return

        # /btw — quick side question, bypasses the pipeline and chat_history
        elif query.strip().lower().startswith("/btw"):
            side_question = query.strip()[4:].strip()
            print("\n[btw]")
            answer = call_llm(f"You are a helpful assistant. Answer concisely.", side_question, model, chat_history) 
            print(f"{answer}\n")

        else:
            print(f"\nQuery : {query}")

            # Step 1 — planner decides which agents to run and in what order
            plan = planner_agent(query, model, chat_history)
            print(f"Plan  : {' -> '.join(plan)}\n")

            # Step 2 — run the pipeline, agents chain their outputs
            agent_results = run_pipeline(query, plan, model, chat_history)

            # Step 3 — extract and print the last agent's section as the final answer
            answer = agent_results[-1][1] if len(agent_results)>0 else 'No result.'

            print(f"\nANSWER:\n{answer}\n")

            # Store this exchange for future turns
            chat_history.append((query, answer))


if __name__ == "__main__":
    main()
