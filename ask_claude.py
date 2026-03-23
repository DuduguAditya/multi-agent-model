import anthropic
import os
import sys

from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Get the question from command line or ask for input
if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
else:
    question = input("Ask Claude: ")

# Send to Claude and get response
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": question}
    ]
)

# Print the answer
print(message.content[0].text)