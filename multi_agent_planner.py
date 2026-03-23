import anthropic
import io
import json
import os
import sys

from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MAX_RETRIES = 3

# All available agents the planner can choose from
AVAILABLE_AGENTS = [
    "game_theory",
    "first_principles",
    "assumption_questioner",
    "math",
    "report_writer"
]


# ============================================================
# HELPER: CALL CLAUDE
# ============================================================

def call_claude(system_prompt, user_message):
    """
    Helper function to call Claude with a specific system prompt.
    Wrapped in try-catch so one bad API call does not crash
    the entire pipeline.
    """
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return message.content[0].text

    except anthropic.APIConnectionError as e:
        return f"[ERROR] Could not connect to API: {e}"
    except anthropic.RateLimitError as e:
        return f"[ERROR] Rate limit hit: {e}"
    except anthropic.APIStatusError as e:
        return f"[ERROR] API returned status {e.status_code}: {e.message}"
    except Exception as e:
        return f"[ERROR] Unexpected error calling Claude: {e}"


# ============================================================
# PLANNER / ROUTER AGENT
# ============================================================

def planner_agent(query):
    """
    PLANNER AGENT: Upgraded from the old router.

    Old behavior: returned one word like "math" or "funny"
    New behavior: returns a JSON list of agents in order,
                  forming an execution plan.

    The planner decides:
      - WHICH agents to use
      - In WHAT ORDER they run
      - The last agent in the list produces the final output
    """
    system_prompt = f"""
    You are a query planner. Your job is to analyze the user's
    question and decide which agents should handle it, and in
    what order.

    Available agents:
    - game_theory: Analyzes decisions from a strategic and
      competitive perspective. Use when the question involves
      competition, negotiations, pricing, strategy, or
      decisions between multiple players/parties.
    - first_principles: Breaks problems down to fundamental
      truths and rebuilds reasoning from scratch. Use when the
      question needs deep analysis or when assumptions need to
      be stripped away.
    - assumption_questioner: Challenges hidden assumptions and
      pokes holes in reasoning. Use when the question involves
      a claim, a plan, or a decision that should be stress-tested.
    - math: Generates and executes Python code for calculations.
      Use ONLY when actual number crunching is needed.
    - report_writer: Writes a clean, conversational summary
      of all prior analysis. Almost always should be LAST
      in the chain if the question is complex.

    Rules:
    1. Return a JSON array of agent names, in execution order.
    2. Use 1 to 4 agents. Not every question needs all agents.
    3. report_writer should be last when used.
    4. math can appear anywhere if calculations are needed.
    5. For simple math-only questions, just return: ["math"]
    6. For complex analytical questions, chain 2-4 agents.

    Respond with ONLY the JSON array, nothing else.

    Examples:
    - "What is 25 * 48?" -> ["math"]
    - "Should a startup lower prices to beat a competitor?"
      -> ["first_principles", "game_theory", "report_writer"]
    - "Is it worth investing in real estate right now?"
      -> ["first_principles", "assumption_questioner",
          "report_writer"]
    - "A company has 3 strategies with these payoffs..."
      -> ["math", "game_theory", "report_writer"]
    - "Why do people think AI will take all jobs?"
      -> ["assumption_questioner", "first_principles",
          "report_writer"]
    """

    result = call_claude(system_prompt, query)

    # Parse the JSON response from the planner
    try:
        # Clean up in case Claude wraps it in markdown
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        plan = json.loads(cleaned)

        # Validate: make sure every agent name is recognized
        validated_plan = []
        for agent_name in plan:
            if agent_name in AVAILABLE_AGENTS:
                validated_plan.append(agent_name)
            else:
                print(
                    f"  Warning: unknown agent '{agent_name}', "
                    f"skipping."
                )

        if not validated_plan:
            return ["first_principles", "report_writer"]

        return validated_plan

    except (json.JSONDecodeError, TypeError) as e:
        print(f"  Could not parse planner output, using default.")
        return ["first_principles", "report_writer"]


# ============================================================
# AGENT 1: GAME THEORY
# ============================================================

def game_theory_agent(query, prior_context=None):
    """
    Analyzes the problem through a game theory lens:
    players, strategies, incentives, payoffs, equilibria.
    """
    system_prompt = """
    You are a game theory strategist.

    Analyze the given problem by identifying:
    - Who are the players/decision makers?
    - What strategies does each player have?
    - What are the incentives and payoffs?
    - Is there a dominant strategy or Nash equilibrium?
    - What is the likely outcome?

    Be specific and analytical. Use game theory terminology
    but explain it simply. Keep your response focused and
    under 200 words.
    """

    if prior_context:
        user_message = (
            f"Original question: {query}\n\n"
            f"Prior analysis from another agent:\n"
            f"{prior_context}\n\n"
            f"Now analyze this from a game theory perspective. "
            f"Build on the prior analysis where relevant."
        )
    else:
        user_message = query

    return call_claude(system_prompt, user_message)


# ============================================================
# AGENT 2: FIRST PRINCIPLES
# ============================================================

def first_principles_agent(query, prior_context=None):
    """
    Strips the problem to its fundamental truths and
    rebuilds reasoning from the ground up.
    """
    system_prompt = """
    You are a first principles thinker.

    Break down the problem to its most fundamental truths:
    1. What do we know for certain? (base facts)
    2. What are we assuming? (flag these clearly)
    3. Rebuild the reasoning from the base facts only.
    4. What conclusion do we reach without assumptions?

    Be rigorous. Strip away conventional wisdom and
    common narratives. Focus on what is actually true.
    Keep your response focused and under 200 words.
    """

    if prior_context:
        user_message = (
            f"Original question: {query}\n\n"
            f"Prior analysis from another agent:\n"
            f"{prior_context}\n\n"
            f"Now apply first principles thinking. "
            f"Build on or challenge the prior analysis."
        )
    else:
        user_message = query

    return call_claude(system_prompt, user_message)


# ============================================================
# AGENT 3: ASSUMPTION QUESTIONER
# ============================================================

def assumption_questioner_agent(query, prior_context=None):
    """
    Challenges hidden assumptions, pokes holes in logic,
    and identifies blind spots.
    """
    system_prompt = """
    You are a critical thinker who questions assumptions.

    Your job is to:
    1. Identify hidden assumptions in the question or
       the prior analysis.
    2. Challenge each assumption - is it really true?
    3. Point out blind spots or things being overlooked.
    4. Suggest what changes if key assumptions are wrong.

    Be constructively skeptical. Do not just criticize -
    explain WHY each assumption might be wrong and what
    the alternative looks like.
    Keep your response focused and under 200 words.
    """

    if prior_context:
        user_message = (
            f"Original question: {query}\n\n"
            f"Prior analysis from another agent:\n"
            f"{prior_context}\n\n"
            f"Now challenge the assumptions in this analysis. "
            f"What is being taken for granted?"
        )
    else:
        user_message = query

    return call_claude(system_prompt, user_message)


# ============================================================
# AGENT 4: REPORT WRITER (Colloquial style)
# ============================================================

def report_writer_agent(query, prior_context=None):
    """
    Takes all prior analysis and writes a clean,
    conversational, easy-to-read summary.
    """
    system_prompt = """
    You are a report writer who writes in plain,
    conversational English.

    Your job:
    1. Take the analysis provided and turn it into a
       clear, readable summary.
    2. Write like you are explaining to a smart friend
       over coffee - no jargon, no stiff academic tone.
    3. Structure it with a clear bottom-line answer first,
       then the supporting reasoning.
    4. Highlight any key risks or things to watch out for.

    Keep it concise. No bullet points unless absolutely
    necessary. Write in flowing paragraphs.
    """

    if prior_context:
        user_message = (
            f"Original question: {query}\n\n"
            f"Here is the full analysis from multiple "
            f"perspectives:\n{prior_context}\n\n"
            f"Write a clean, conversational report "
            f"summarizing all of this."
        )
    else:
        user_message = query

    return call_claude(system_prompt, user_message)


# ============================================================
# AGENT 5: MATH AGENT (carried over from your Task 3 code)
# ============================================================

def execute_code(code):
    """
    Executes Python code and returns (success, result_or_error).
    Carried over from your original multi_agent_try_catch.py.
    """
    try:
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        exec(code, {})

        sys.stdout = old_stdout
        result = captured_output.getvalue().strip()
        return True, result

    except Exception as e:
        sys.stdout = old_stdout
        return False, str(e)


def fix_code(original_query, broken_code, error_message):
    """
    Asks Claude to fix broken code. Self-correction helper.
    Carried over from your original multi_agent_try_catch.py.
    """
    system_prompt = """
    You are a Python debugger.

    The user will provide:
    1. The original math problem
    2. The Python code that was generated
    3. The error message

    Fix the code so it works correctly.
    Respond with ONLY the corrected Python code,
    no explanations, no markdown backticks.
    """

    user_message = (
        f"Original problem: {original_query}\n\n"
        f"Broken code:\n{broken_code}\n\n"
        f"Error message: {error_message}\n\n"
        f"Please fix the code."
    )

    return call_claude(system_prompt, user_message)


def math_agent(query, prior_context=None):
    """
    MATH AGENT: Generates Python code, executes it,
    returns result. Includes self-correction loop.

    If prior_context exists, it incorporates it into the
    prompt so Claude knows what numbers to work with.
    """
    system_prompt = """
    You are a Python math solver.

    Write Python code to solve the math problem.
    The code must print the final answer using print().

    Respond with ONLY the Python code,
    no explanations, no markdown backticks.

    Example response:
    result = 2 + 2
    print(result)
    """

    if prior_context:
        user_message = (
            f"Original question: {query}\n\n"
            f"Context from prior analysis:\n"
            f"{prior_context}\n\n"
            f"Write Python code to calculate whatever "
            f"numbers are needed here."
        )
    else:
        user_message = query

    code = call_claude(system_prompt, user_message)

    for attempt in range(MAX_RETRIES):
        success, result = execute_code(code)

        if success:
            return f"Calculation result: {result}"
        else:
            print(f"  Math error (attempt {attempt + 1}): {result}")
            if attempt < MAX_RETRIES - 1:
                code = fix_code(query, code, result)
            else:
                return (
                    f"Math failed after {MAX_RETRIES} attempts. "
                    f"Last error: {result}"
                )

    return "Unexpected error in math agent."


# ============================================================
# AGENT REGISTRY
# Maps agent names (from the planner) to their functions.
# ============================================================

AGENT_REGISTRY = {
    "game_theory": game_theory_agent,
    "first_principles": first_principles_agent,
    "assumption_questioner": assumption_questioner_agent,
    "math": math_agent,
    "report_writer": report_writer_agent,
}


# ============================================================
# PIPELINE EXECUTOR
# This is the new core logic. It takes the plan from
# the planner and chains agents together, passing each
# agent's output to the next one as context.
# ============================================================

def run_pipeline(query, plan):
    """
    Executes agents in the order specified by the plan.

    Each agent receives:
    - The original query (so it never loses sight of
      the actual question)
    - The accumulated context from all prior agents
      (so it builds on previous analysis)

    If any agent fails, we catch the error, log it,
    and continue with the remaining agents.
    """
    accumulated_context = ""

    # Track agent names that have completed so far
    completed_agents = []

    for step_number, agent_name in enumerate(plan, start=1):

        # Show what this agent receives
        if step_number == 1:
            print(
                f"[{step_number}/{len(plan)}] {agent_name}"
                f"(query)"
            )
        else:
            prior_names = " + ".join(completed_agents)
            print(
                f"[{step_number}/{len(plan)}] {agent_name}"
                f"(query, {prior_names})"
            )

        agent_function = AGENT_REGISTRY.get(agent_name)

        if not agent_function:
            print(f"  Skipping unknown agent: {agent_name}")
            continue

        try:
            if step_number == 1:
                result = agent_function(query)
            else:
                result = agent_function(
                    query, prior_context=accumulated_context
                )

            if result.startswith("[ERROR]"):
                print(f"  Error: {result}")
                accumulated_context += (
                    f"\n\n--- {agent_name} ---\n"
                    f"(This agent encountered an error: {result})"
                )
            else:
                # Show a brief preview of what the agent produced
                preview = result[:80].replace("\n", " ")
                print(f"  -> {preview}...")
                accumulated_context += (
                    f"\n\n--- {agent_name} ---\n{result}"
                )
                completed_agents.append(agent_name)

        except Exception as e:
            print(f"  Error: {agent_name} crashed: {e}")
            accumulated_context += (
                f"\n\n--- {agent_name} ---\n"
                f"(This agent crashed with error: {e})"
            )
            continue

    return accumulated_context


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main function that orchestrates everything.

    Flow:
    1. Get query from user
    2. Planner creates an execution plan (list of agents)
    3. Pipeline runs the agents in order, chaining context
    4. Last agent's contribution is the final output
       (or the full accumulated context if no report_writer)
    """
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Ask me anything: ")

    print(f"\nQuery: {query}")

    # Step 1: Planner decides the execution plan
    plan = planner_agent(query)
    print(f"Plan: {' -> '.join(plan)}\n")

    # Step 2: Run the pipeline
    final_output = run_pipeline(query, plan)

    # Step 3: Print the final result
    print(f"\nANSWER:\n")

    if plan[-1] == "report_writer":
        sections = final_output.split("--- report_writer ---")
        if len(sections) > 1:
            print(sections[-1].strip())
        else:
            print(final_output.strip())
    else:
        # No report writer - extract last agent's output
        last_agent = plan[-1]
        sections = final_output.split(f"--- {last_agent} ---")
        if len(sections) > 1:
            print(sections[-1].strip())
        else:
            print(final_output.strip())


if __name__ == "__main__":
    main()