import litellm
import os
import sys

from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
#
# Usage:
#   python ask_litellm.py --claude "capital of France?"
#   python ask_litellm.py --gemini "capital of France?"
#   python ask_litellm.py --mistral "capital of France?"
#
# No flag defaults to Claude.
# ============================================================

load_dotenv()  # reads API keys from .env

# Provider registry
PROVIDERS = {
    "--claude":  "anthropic/claude-sonnet-4-20250514",
    "--gemini":  "gemini/gemini-2.0-flash",
    "--mistral": "mistral/mistral-small-latest",
    "--openai":  "openai/gpt-4o",
}

DEFAULT_PROVIDER = "--claude"


def select_model():
    for flag in PROVIDERS:
        if flag in sys.argv:
            sys.argv.remove(flag)
            return PROVIDERS[flag]
    return PROVIDERS[DEFAULT_PROVIDER]


MODEL = select_model()

# Get question
if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
else:
    question = input("Ask anything: ")

print(f"Model: {MODEL}")

# Call LLM
response = litellm.completion(
    model=MODEL,
    max_tokens=1024,
    messages=[
        {"role": "user", "content": question}
    ]
)

print(response.choices[0].message.content)