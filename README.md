# Multi-Agent AI Pipeline

A planner-driven multi-agent system that analyzes a query, selects the right specialist agents, and chains them sequentially — each agent receives the full output of all prior agents as context.

Supports **Claude**, **Gemini**, and **Mistral** via [LiteLLM](https://github.com/BerriAI/litellm).

---

## Setup

**1. Install dependencies**
```bash
pip install anthropic litellm python-dotenv
```

**2. Configure API keys**

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
```
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
```

---

## Usage

```bash
# Defaults to Claude
python main.py "Should a startup lower prices to beat a competitor?"

# Use a different model
python main.py --gemini  "Is it worth investing in real estate right now?"
python main.py --mistral "What is the fastest route to learn machine learning?"
```

---

## Architecture & Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Planner Agent  │  ← Reads the query, returns a JSON array of agents to run
└────────┬────────┘    e.g. ["first_principles", "game_theory", "report_writer"]
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                   Pipeline Executor                  │
│                                                     │
│  Agent 1 (query only)                               │
│      ↓  output_1                                    │
│  Agent 2 (query + output_1)                         │
│      ↓  output_1 + output_2                         │
│  Agent 3 (query + output_1 + output_2)  ← final     │
└─────────────────────────────────────────────────────┘
         │
         ▼
  Last agent's output = Final Answer
```

Each agent always sees the **original query** (so it never loses sight of the question) plus all **accumulated context** from prior agents.

---

## Agents

| Agent | Role |
|---|---|
| `game_theory` | Identifies players, strategies, incentives, payoffs, and Nash equilibria |
| `first_principles` | Strips assumptions and rebuilds reasoning from base facts |
| `assumption_questioner` | Challenges hidden assumptions and finds blind spots |
| `math` | Writes and executes Python code for calculations; self-corrects on errors (up to 3 retries) |
| `report_writer` | Synthesizes all prior analysis into a clear, conversational summary |

---

## Example Plans

| Query type | Plan |
|---|---|
| Pure calculation | `["math"]` |
| Strategic business question | `["first_principles", "game_theory", "report_writer"]` |
| Investment / decision | `["first_principles", "assumption_questioner", "report_writer"]` |
| Claim or belief to stress-test | `["assumption_questioner", "first_principles", "report_writer"]` |
| Numbers + strategy | `["math", "game_theory", "report_writer"]` |

---

## Key Design Decisions

**Why LiteLLM?** It provides a single unified interface across providers. Swapping from Claude to Gemini requires no code changes — just a flag at runtime.

**Why a planner instead of a fixed router?** A fixed router (like the earlier `multi_agent.py`) can only send to one agent. The planner creates a dynamic execution chain appropriate to the question's complexity.

**Why pass the full context forward?** Each agent can either build on or challenge what came before, rather than re-analyzing the question from scratch. This produces more nuanced output than parallel agents would.

**Why `.env`?** API keys should never be hardcoded in source files. `.env` keeps secrets out of the codebase and out of version control (add `.env` to `.gitignore`).
