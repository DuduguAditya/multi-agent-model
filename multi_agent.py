import anthropic
import os
import sys

from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def call_claude(system_prompt, user_message):
    """Helper function to call Claude with a specific system prompt."""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    return message.content[0].text


def router_agent(query):
    """
    ROUTER AGENT: Decides which agent should handle the query.
    Returns either 'math' or 'funny'
    """
    system_prompt = """You are a router that classifies user queries.
    
    If the query is a math problem, calculation, or anything involving numbers/equations, respond with exactly: math
    
    If the query is a general knowledge question or anything else, respond with exactly: funny
    
    Respond with ONLY one word: either 'math' or 'funny'. Nothing else."""
    
    result = call_claude(system_prompt, query)
    return result.strip().lower()


def funny_agent(query):
    """
    FUNNY AGENT: Answers questions with humor and wit.
    """
    system_prompt = """You are a hilarious comedian who answers questions.
    
    Give the correct answer, but make it funny and entertaining.
    Use jokes, puns, or silly analogies.
    Keep it short - 2-3 sentences max."""
    
    return call_claude(system_prompt, query)


def math_agent(query):
    """
    MATH AGENT: Generates Python code, executes it, and returns the result.
    """
    # Step 1: Ask Claude to write Python code
    system_prompt = """You are a Python math solver.
    
    Write Python code to solve the math problem.
    The code must print the final answer using print().
    
    Respond with ONLY the Python code, no explanations, no markdown backticks.
    Example response:
    result = 2 + 2
    print(result)"""
    
    code = call_claude(system_prompt, query)
    
    print(f"\nGenerated Code:\n{code}\n")
    
    # Step 2: Execute the code
    try:
        # Create a dictionary to capture the output
        exec_globals = {}
        
        # Capture print output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        # Execute the code
        exec(code, exec_globals)
        
        # Get the output
        sys.stdout = old_stdout
        result = captured_output.getvalue().strip()
        
        return f" Result: {result}"
    
    except Exception as e:
        return f" Error executing code: {str(e)}"


def main():
    """Main function that orchestrates the agents."""
    
    # Get query from command line or input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Ask me anything: ")
    
    print(f"\nQuery: {query}")
    
    # Step 1: Router decides which agent to use
    route = router_agent(query)
    print(f"Router decision: {route.upper()} agent\n")
    
    # Step 2: Send to appropriate agent
    if route == "math":
        response = math_agent(query)
    else:
        response = funny_agent(query)
    
    # Step 3: Print final response
    print(f"Answer: {response}")


if __name__ == "__main__":
    main()