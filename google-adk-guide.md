[← Back to Home](/)

# Google ADK (Agent Development Kit) — Concise Guide

## Overview

Google ADK is an open-source, code-first Python framework for building, evaluating, and deploying AI agents. It powers Google products like Agentspace and Customer Engagement Suite (CES).

**Key characteristics:**
- Model-agnostic (optimized for Gemini but works with others)
- Deployment-agnostic (local, Cloud Run, Vertex AI Agent Engine)
- Rich tool ecosystem (custom functions, OpenAPI specs, MCP tools)
- Built-in evaluation and debugging tools

## Installation

```bash
pip install google-adk
```

## Core Concepts

- ADK is a flexible and modular framework that applies software development principles to AI agent creation. It is designed to simplify building, deploying, and orchestrating agent workflows. While optimized for Gemini, ADK is model-agnostic, deployment-agnostic, and compatible with other frameworks.
- [Agent Development Kit (ADK)](https://github.com/google/adk-python)

### 1. Basic Agent Definition

Every ADK agent requires a `root_agent` in `agent.py`:

```python
from google.adk.agents import Agent

root_agent = Agent(
    model='gemini-2.5-flash',
    name='my_agent',
    description="A helpful assistant.",
    instruction="You are a helpful assistant that answers questions.",
    tools=[],  # List of tools the agent can use
)
```

### 2. Tools

Tools give agents capabilities to interact with the world. Define them as Python functions:

```python
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """Retrieves weather for a specified city.
    
    Args:
        city: The city name to get weather for.
    
    Returns:
        Weather information dictionary.
    """
    # Implementation here
    return {"city": city, "temp": "22°C", "condition": "sunny"}

def search_database(query: str, limit: int = 10) -> dict:
    """Searches the database for relevant records."""
    return {"results": [...], "count": limit}

root_agent = Agent(
    model='gemini-2.5-flash',
    name='weather_agent',
    instruction="Help users with weather queries using the get_weather tool.",
    tools=[get_weather, search_database],
)
```

**Tool Context** — Access session state within tools:

```python
from google.adk.tools import ToolContext

def save_preference(key: str, value: str, tool_context: ToolContext) -> dict:
    """Save user preference to state."""
    tool_context.state[f"user:{key}"] = value
    return {"status": "saved", "key": key, "value": value}
```

### 3. Built-in Tools

ADK provides pre-built tools:

```python
from google.adk.tools import google_search
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools import load_memory

root_agent = Agent(
    model='gemini-2.5-flash',
    name='search_agent',
    tools=[google_search, PreloadMemoryTool()],
)
```

---

## Agent Types

ADK provides three categories of agents:
- **LLM Agents** utilize Large Language Models as their core engine to understand natural language, reason, plan, generate responses, and dynamically decide how to proceed or which tools to use. **Workflow Agents** control the execution flow of other agents in predefined, deterministic patterns without using an LLM for the flow control itself. **Custom Agents** are created by extending BaseAgent directly for unique operational logic.
- [ADK - Building Agents](https://google.github.io/adk-docs/agents/)

| Type | Purpose | Use Case |
|------|---------|----------|
| **LlmAgent** | LLM-powered reasoning and decision making | Flexible, language-centric tasks |
| **Workflow Agents** | Deterministic flow control (no LLM for orchestration) | Structured, predictable processes |
| **Custom Agents** | Full control via `BaseAgent` inheritance | Unique logic, complex integrations |

---

## Workflow Agents (Orchestration Patterns)

- SequentialAgent runs its sub-agents one after another. The output of one agent can be passed as the input to the next, making it perfect for multi-step pipelines like: fetch data → clean data → analyze data → summarize findings. ParallelAgent runs all its sub-agents concurrently, which is ideal for independent tasks. LoopAgent works like a while loop in programming.
- [Building Collaborative AI: A Developer's Guide to Multi-Agent Systems with ADK](https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk)

### SequentialAgent — Chaining Pattern

Executes sub-agents one after another. Use `output_key` to pass data via shared state.

```python
from google.adk.agents import LlmAgent, SequentialAgent

# Step 1: Research agent
researcher = LlmAgent(
    name="Researcher",
    model="gemini-2.5-flash",
    instruction="Research the topic and summarize key findings.",
    output_key="research_findings"  # Saves output to state
)

# Step 2: Writer agent (reads from state)
writer = LlmAgent(
    name="Writer",
    model="gemini-2.5-flash",
    instruction="Write a report based on: {research_findings}",  # Template injection
    output_key="final_report"
)

# Sequential pipeline
pipeline = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, writer]
)
```

### ParallelAgent — Fan-out Pattern

Executes sub-agents concurrently for independent tasks:

```python
from google.adk.agents import LlmAgent, ParallelAgent

flight_agent = LlmAgent(
    name="FlightAgent",
    model="gemini-2.5-flash",
    instruction="Find available flights.",
    output_key="flight_options"
)

hotel_agent = LlmAgent(
    name="HotelAgent",
    model="gemini-2.5-flash",
    instruction="Find available hotels.",
    output_key="hotel_options"
)

# Parallel execution
travel_search = ParallelAgent(
    name="TravelSearch",
    sub_agents=[flight_agent, hotel_agent]
)
```

### LoopAgent — Iteration Pattern

Repeatedly executes sub-agents until a condition is met:

```python
from google.adk.agents import LlmAgent, LoopAgent

generator = LlmAgent(
    name="Generator",
    model="gemini-2.5-flash",
    instruction="Generate content based on feedback: {feedback}",
    output_key="draft"
)

reviewer = LlmAgent(
    name="Reviewer", 
    model="gemini-2.5-flash",
    instruction="""Review the draft: {draft}
    If acceptable, respond with 'APPROVED'.
    Otherwise, provide specific feedback.""",
    output_key="feedback"
)

# Loop until quality threshold or max iterations
refinement_loop = LoopAgent(
    name="RefinementLoop",
    sub_agents=[generator, reviewer],
    max_iterations=5  # Safety limit
)
```

**Exit Condition:** Agents can signal early exit via `escalate=True` in `EventActions`.

---

## Multi-Agent Patterns

### Coordinator/Router Pattern

Central agent routes to specialists using LLM-driven delegation:

```python
from google.adk.agents import LlmAgent

billing_agent = LlmAgent(
    name="BillingAgent",
    description="Handles billing and payment inquiries.",  # Key for routing
    model="gemini-2.5-flash",
    instruction="Help with billing questions."
)

support_agent = LlmAgent(
    name="SupportAgent",
    description="Handles technical support issues.",
    model="gemini-2.5-flash",
    instruction="Help with technical problems."
)

coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.5-flash",
    instruction="""Route user requests to the appropriate specialist.
    Analyze the request and delegate to the best sub-agent.""",
    sub_agents=[billing_agent, support_agent]  # Available for delegation
)
```

### AgentTool Pattern — Agent as Tool

Wrap an agent to use it as an explicit tool:

```python
from google.adk.tools import AgentTool

research_agent = LlmAgent(
    name="ResearchAgent",
    model="gemini-2.5-flash",
    instruction="Research topics thoroughly."
)

# Wrap as tool for explicit invocation
research_tool = AgentTool(agent=research_agent)

writer_agent = LlmAgent(
    name="WriterAgent",
    model="gemini-2.5-flash",
    instruction="Write reports. Use the research tool when needed.",
    tools=[research_tool]  # Agent-as-tool
)
```

### Hierarchical Multi-Agent

```python
# Level 3: Tool agents
web_searcher = LlmAgent(name="WebSearch", description="Searches the web.")
summarizer = LlmAgent(name="Summarizer", description="Condenses text.")

# Level 2: Coordinator
research_assistant = LlmAgent(
    name="ResearchAssistant",
    description="Finds and summarizes information.",
    sub_agents=[web_searcher, summarizer]
)

# Level 1: Top-level
report_writer = LlmAgent(
    name="ReportWriter",
    instruction="Write reports using research assistant.",
    sub_agents=[research_assistant]
)
```

---

## State Management

- When a parent agent invokes a sub-agent, it passes the same InvocationContext. This means they share the same temporary state, which is ideal for passing data that is only relevant for the current turn.
- [Multi-Agent Systems in ADK](https://google.github.io/adk-docs/agents/multi-agents/)


### State Prefixes

| Prefix | Scope | Persistence |
|--------|-------|-------------|
| (none) | Current session only | Session lifetime |
| `temp:` | Current invocation only | Single turn |
| `user:` | Across all user sessions | Persistent per user |
| `app:` | Across all sessions/users | Global persistent |

### Reading/Writing State

**In Tools (via ToolContext):**
```python
def my_tool(query: str, tool_context: ToolContext) -> dict:
    # Read state
    user_name = tool_context.state.get("user:name", "Guest")
    
    # Write state
    tool_context.state["user:last_query"] = query
    tool_context.state["temp:processing"] = True
    
    return {"greeting": f"Hello {user_name}"}
```

**In Agent Instructions (template injection):**
```python
agent = LlmAgent(
    instruction="The user's name is {user:name}. Their preferences: {user:preferences}",
    # ADK auto-injects state values into {placeholders}
)
```

**Using output_key:**
```python
agent = LlmAgent(
    name="Analyzer",
    instruction="Analyze the data.",
    output_key="analysis_result"  # Auto-saves LLM output to state
)
```

---

## Memory (Long-term Knowledge)

- Session and State focus on the current interaction – the history and data of the single, active conversation. MemoryService manages the Long-Term Knowledge Store, handling ingesting information from completed Sessions into the long-term store, and provides methods to search this stored knowledge based on queries.
- [Introduction to Conversational Context: Session, State, and Memory](https://google.github.io/adk-docs/sessions/)


### Memory vs State

| Aspect | State | Memory |
|--------|-------|--------|
| Scope | Current session/user | Cross-session knowledge store |
| Storage | Key-value pairs | Searchable vector store |
| Use | Track conversation data | Remember past interactions |

### Memory Tools

```python
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools import load_memory

# Always load memories at turn start
agent = Agent(
    tools=[PreloadMemoryTool()],  # Auto-retrieves relevant memories
)

# Or load on-demand
agent = Agent(
    tools=[load_memory],  # Agent decides when to search memory
)
```

### Memory Services

```python
from google.adk.memory import InMemoryMemoryService
from google.adk.memory import VertexAiMemoryBankService

# Local development
memory_service = InMemoryMemoryService()

# Production (Vertex AI)
memory_service = VertexAiMemoryBankService(
    project="PROJECT_ID",
    location="LOCATION",
    agent_engine_id="ENGINE_ID"
)
```

### Saving to Memory (via Callback)

```python
async def save_session_callback(callback_context):
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

agent = Agent(
    after_agent_callback=save_session_callback,
)
```

---

## Callbacks

Callbacks hook into the agent execution lifecycle:

| Callback | Trigger Point |
|----------|---------------|
| `before_agent` | Before agent starts processing |
| `after_agent` | After agent completes |
| `before_model` | Before LLM call |
| `after_model` | After LLM response |
| `before_tool` | Before tool execution |
| `after_tool` | After tool execution |

### Callback Example

```python
from google.adk.agents import CallbackContext
from google.adk.events import Event

def logging_callback(callback_context: CallbackContext):
    """Log agent activity."""
    print(f"Agent: {callback_context.agent_name}")
    print(f"State: {callback_context.state}")
    return None  # Continue normal execution

def guardrail_callback(callback_context: CallbackContext):
    """Block certain requests."""
    user_input = callback_context.user_content
    if "forbidden" in str(user_input).lower():
        # Return Event to override default behavior
        return Event(
            author=callback_context.agent_name,
            content="I cannot process this request."
        )
    return None  # Continue normally

agent = Agent(
    before_agent_callback=guardrail_callback,
    after_agent_callback=logging_callback,
)
```

### Callback Return Values

| Return | Effect |
|--------|--------|
| `None` | Continue normal execution |
| `Event` object | Override/skip the normal step |

---

## Custom Agents

For complex logic not covered by workflow agents:

```python
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator

class ConditionalAgent(BaseAgent):
    """Custom agent with conditional routing."""
    
    def __init__(self, simple_agent, complex_agent, **kwargs):
        super().__init__(
            sub_agents=[simple_agent, complex_agent],
            **kwargs
        )
        self.simple_agent = simple_agent
        self.complex_agent = complex_agent
    
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Custom routing logic
        query_length = len(str(ctx.user_content))
        
        if query_length < 50:
            target = self.simple_agent
        else:
            target = self.complex_agent
        
        # Delegate to selected agent
        async for event in target.run_async(ctx):
            yield event
```

---

## Running Agents

### CLI
```bash
adk run agent_folder/
```

### Web UI (Development)
```bash
adk web agent_folder/
```

### API Server
```bash
adk api_server agent_folder/
```

### Python (Programmatic)

```python
from google.adk import Runner
from google.adk.sessions import InMemorySessionService

runner = Runner(
    agent=root_agent,
    app_name="my_app",
    session_service=InMemorySessionService(),
)

async def chat():
    session = await runner.session_service.create_session(
        app_name="my_app",
        user_id="user_123"
    )
    
    async for event in runner.run_async(
        user_id="user_123",
        session_id=session.id,
        new_message="Hello!"
    ):
        print(event)
```

---

## Environment Configuration

**.env file:**
```bash
# Google AI Studio (Gemini API)
GOOGLE_API_KEY=your_api_key

# OR Vertex AI
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

---

## Project Structure

```
my_agent/
├── __init__.py
├── agent.py          # Must define root_agent
├── tools.py          # Custom tool functions
├── .env              # API keys and config
└── prompts.yaml      # Optional: externalized prompts
```

---

## Quick Reference: Common Patterns

| Pattern | ADK Implementation |
|---------|-------------------|
| **Sequential Chain** | `SequentialAgent` with `output_key` |
| **Parallel Fan-out** | `ParallelAgent` |
| **Loop/Refinement** | `LoopAgent` with `max_iterations` |
| **Router/Coordinator** | `LlmAgent` with `sub_agents` + descriptions |
| **Agent-as-Tool** | `AgentTool(agent=...)` |
| **Human-in-the-Loop** | Tool confirmation flow or custom callback |
| **Generator-Critic** | `LoopAgent` containing generator + reviewer |
| **Hierarchical Teams** | Nested `sub_agents` across multiple levels |

---

## Resources

- **Documentation:** https://google.github.io/adk-docs/
- **GitHub:** https://github.com/google/adk-python
- **Vertex AI Integration:** https://cloud.google.com/vertex-ai/generative-ai/docs/agent-development-kit/quickstart