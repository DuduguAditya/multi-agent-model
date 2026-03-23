import anthropic
import io
import os
import sys

from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Maximum retry attempts for fixing code
MAX_RETRIES = 3


def call_claude(system_prompt, user_message):
    """
    Helper function to call Claude with a specific system prompt.
    """
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
    system_prompt = """
    You are a router that classifies user queries.
    
    If the query is a math problem, calculation, or anything 
    involving numbers/equations, respond with exactly: math
    
    If the query is a general knowledge question or anything 
    else, respond with exactly: funny
    
    Respond with ONLY one word: either 'math' or 'funny'. 
    Nothing else.
    """
    
    result = call_claude(system_prompt, query)
    return result.strip().lower()


def funny_agent(query):
    """
    FUNNY AGENT: Answers questions with humor and wit.
    """
    system_prompt = """
    You are a hilarious comedian who answers questions.
    
    Give the correct answer, but make it funny and entertaining.
    Use jokes, puns, or silly analogies.
    Keep it short - 2-3 sentences max.
    """
    
    return call_claude(system_prompt, query)


def execute_code(code):
    """
    Executes Python code and returns (success, result_or_error)
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
    Asks Claude to fix the broken code based on the error.
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
    
    user_message = f"""
    Original problem: {original_query}

    Broken code:
    {broken_code}

    Error message: {error_message}

    Please fix the code.
    """
    
    return call_claude(system_prompt, user_message)


def math_agent(query):
    """
    MATH AGENT: Generates Python code, executes it, returns result.
    Includes self-correction loop if code fails.
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
    
    code = call_claude(system_prompt, query)
    
    # Try to execute, with retry loop for errors
    for attempt in range(MAX_RETRIES):
        print(f"\n--- Attempt {attempt + 1} ---")
        print(f"Code:\n{code}\n")
        
        success, result = execute_code(code)
        
        if success:
            return f"Result: {result}"
        else:
            print(f"Error: {result}")
            
            if attempt < MAX_RETRIES - 1:
                print("Attempting to fix the code...\n")
                code = fix_code(query, code, result)
            else:
                return f"Failed after {MAX_RETRIES} attempts. " \
                       f"Last error: {result}"
    
    return "Unexpected error in math agent."


def main():
    """
    Main function that orchestrates the agents.
    """
    # Get query from command line or input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Ask me anything: ")
    
    print(f"\nQuery: {query}")
    
    # Step 1: Router decides which agent to use
    route = router_agent(query)
    print(f"Router decision: {route.upper()} agent")
    
    # Step 2: Send to appropriate agent
    if route == "math":
        response = math_agent(query)
    else:
        response = funny_agent(query)
    
    # Step 3: Print final response
    print(f"\nFinal Answer: {response}")


if __name__ == "__main__":
    main()