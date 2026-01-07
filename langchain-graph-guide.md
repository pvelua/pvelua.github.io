[← Back to Home](/)

# LangChain & LangGraph — Concise Guide

## Overview

**LangChain** is a framework for building AI agents with a standard tool-calling architecture, provider-agnostic design, and middleware for customization. As of v1.0 (November 2025), it focuses on the core agent loop with the new `create_agent` abstraction.

**LangGraph** is a low-level orchestration framework for building stateful, long-running agents as graphs. It provides durable execution, human-in-the-loop, comprehensive memory, and production-ready deployment.

**Key relationship:** LangChain's `create_agent` is built on top of LangGraph. Use LangChain for quick agent development; use LangGraph directly when you need fine-grained control over workflows.

## Installation

```bash
# LangChain 1.0+ with OpenAI
pip install langchain langchain-openai

# LangGraph
pip install langgraph

# For persistence
pip install langgraph-checkpoint  # Base checkpointing
pip install langgraph-checkpoint-sqlite  # SQLite backend
pip install langgraph-checkpoint-postgres  # PostgreSQL backend
```

---

## Part 1: LangChain Agents (High-Level)

### Basic Agent with `create_agent`

The simplest way to build an agent in LangChain 1.0:

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

# Create agent with tools
agent = create_agent(
    model="openai:gpt-4o",  # Provider-agnostic model string
    tools=[get_weather, search_web],
)

# Run the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}]
})
```

### The Core Agent Loop

```
┌─────────────────────────────────────────┐
│  User Input                             │
└─────────────────┬───────────────────────┘
                  ▼
         ┌────────────────┐
         │   LLM Call     │◄────────────┐
         └───────┬────────┘             │
                 ▼                      │
         ┌────────────────┐             │
         │  Tool Calls?   │             │
         └───────┬────────┘             │
                 │                      │
        Yes      │      No              │
         ▼       │       ▼              │
┌────────────┐   │  ┌──────────┐        │
│Execute Tool│   │  │  Output  │        │
└─────┬──────┘   │  └──────────┘        │
      │          │                      │
      └──────────┴──────────────────────┘
```

### Middleware (LangChain 1.0+)

Middleware provides hooks into the agent loop for customization:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HITLMiddleware,           # Human-in-the-loop
    SummarizationMiddleware,  # Compress long conversations
    ToolRetryMiddleware,      # Retry failed tool calls
    PIIMiddleware,            # Protect sensitive data
    TodoListMiddleware,       # Task planning
)

# Human-in-the-loop: approve sensitive tool calls
agent = create_agent(
    model="openai:gpt-4o",
    tools=[process_refund, delete_account],
    middleware=[
        HITLMiddleware(
            tools=["process_refund", "delete_account"],
            allowed_decisions=["approve", "edit", "reject"]
        ),
    ],
)

# Automatic summarization when approaching token limits
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool],
    middleware=[
        SummarizationMiddleware(
            max_tokens=4000,
            summarize_after=10  # messages
        ),
    ],
)

# Tool retry with exponential backoff
agent = create_agent(
    model="openai:gpt-4o",
    tools=[api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)

# Combine multiple middleware
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        PIIMiddleware("ssn", detector=r"\d{3}-\d{2}-\d{4}", strategy="mask"),
        ToolRetryMiddleware(max_retries=2),
        SummarizationMiddleware(max_tokens=8000),
    ],
)
```

**Middleware Hooks:**

| Hook | When It Runs | Use Case |
|------|--------------|----------|
| `before_model` | Before LLM call | Modify prompt, add context |
| `after_model` | After LLM response | Filter output, trigger actions |
| `modify_model_request` | During request construction | Dynamic tool selection |

### Custom Middleware

```python
from langchain.agents.middleware import AgentMiddleware

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, config):
        print(f"Calling model with {len(state['messages'])} messages")
        return None  # Continue normal execution
    
    def after_model(self, state, config):
        last_msg = state["messages"][-1]
        print(f"Model response: {last_msg.content[:100]}...")
        return None

agent = create_agent(
    model="openai:gpt-4o",
    tools=[my_tool],
    middleware=[LoggingMiddleware()],
)
```

---

## Part 2: LangGraph (Low-Level Control)

### Core Concepts

LangGraph models workflows as directed graphs with:

- **State**: Data that flows through and persists across the graph
- **Nodes**: Functions that process and update state
- **Edges**: Connections defining execution flow (normal or conditional)
- **Checkpointers**: Persistence layer for state snapshots

### Basic StateGraph

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. Define State Schema
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Reducer appends messages
    context: str

# 2. Define Node Functions
def research_node(state: State) -> dict:
    """Process state and return updates."""
    return {
        "messages": [{"role": "assistant", "content": "Researching..."}],
        "context": "Research findings here"
    }

def write_node(state: State) -> dict:
    context = state["context"]
    return {
        "messages": [{"role": "assistant", "content": f"Based on: {context}"}]
    }

# 3. Build Graph
builder = StateGraph(State)

# Add nodes
builder.add_node("research", research_node)
builder.add_node("write", write_node)

# Add edges
builder.add_edge(START, "research")
builder.add_edge("research", "write")
builder.add_edge("write", END)

# 4. Compile and Run
graph = builder.compile()
result = graph.invoke({"messages": [{"role": "user", "content": "Write about AI"}]})
```

### State and Reducers

Reducers define how state updates are merged:

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    # Default: overwrite
    current_step: str
    
    # add_messages: append to list, handle message deduplication
    messages: Annotated[list, add_messages]
    
    # Custom reducer: sum values
    total_cost: Annotated[float, add]
    
    # Custom reducer: extend list
    items: Annotated[list, lambda x, y: x + y]
```

### Conditional Edges (Routing)

```python
from typing import Literal

def route_by_intent(state: State) -> Literal["billing", "support", "end"]:
    """Route based on user intent."""
    last_message = state["messages"][-1].content.lower()
    
    if "billing" in last_message or "payment" in last_message:
        return "billing"
    elif "help" in last_message or "problem" in last_message:
        return "support"
    else:
        return "end"

# Add conditional edge
builder.add_conditional_edges(
    "classifier",           # Source node
    route_by_intent,        # Routing function
    {                       # Mapping: return value -> target node
        "billing": "billing_agent",
        "support": "support_agent",
        "end": END,
    }
)
```

### Conditional Entry Point

```python
from langgraph.graph import START

def initial_router(state: State) -> Literal["fast_path", "slow_path"]:
    if state.get("priority") == "high":
        return "fast_path"
    return "slow_path"

builder.add_conditional_edges(START, initial_router)
```

### Command: Combined Routing + State Updates

```python
from langgraph.types import Command

def process_node(state: State) -> Command:
    """Update state AND control flow in one return."""
    if state["status"] == "complete":
        return Command(
            update={"messages": [{"role": "assistant", "content": "Done!"}]},
            goto=END
        )
    else:
        return Command(
            update={"retry_count": state.get("retry_count", 0) + 1},
            goto="retry_node"
        )

# Must declare possible destinations
builder.add_node("process", process_node, ends=["retry_node", END])
```

---

## Workflow Patterns

### Sequential Chain

```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: Annotated[list, add_messages]
    research: str
    draft: str
    final: str

def research(state):
    return {"research": "Research findings..."}

def draft(state):
    return {"draft": f"Draft based on: {state['research']}"}

def finalize(state):
    return {"final": f"Final version: {state['draft']}"}

builder = StateGraph(State)
builder.add_node("research", research)
builder.add_node("draft", draft)
builder.add_node("finalize", finalize)

# Sequential flow
builder.add_edge(START, "research")
builder.add_edge("research", "draft")
builder.add_edge("draft", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()
```

### Parallel Execution (Fan-out)

```python
from langgraph.types import Send

def fan_out(state: State) -> list[Send]:
    """Send to multiple nodes in parallel."""
    tasks = state["tasks"]
    return [Send("worker", {"task": task}) for task in tasks]

def worker(state: dict) -> dict:
    task = state["task"]
    return {"result": f"Completed: {task}"}

def aggregate(state: State) -> dict:
    # Results from parallel workers are collected here
    return {"summary": "All tasks completed"}

builder.add_node("worker", worker)
builder.add_node("aggregate", aggregate)
builder.add_conditional_edges("dispatcher", fan_out)
builder.add_edge("worker", "aggregate")
```

### Loop Pattern (Iteration)

```python
def should_continue(state: State) -> Literal["generate", "end"]:
    """Check if we should continue the loop."""
    if state.get("approved", False):
        return "end"
    if state.get("iterations", 0) >= 5:
        return "end"
    return "generate"

def generate(state: State) -> dict:
    iterations = state.get("iterations", 0)
    return {
        "draft": f"Draft version {iterations + 1}",
        "iterations": iterations + 1
    }

def review(state: State) -> dict:
    # Simulated review
    approved = state["iterations"] >= 3
    return {"approved": approved, "feedback": "Needs work" if not approved else "Approved"}

builder.add_edge(START, "generate")
builder.add_edge("generate", "review")
builder.add_conditional_edges("review", should_continue, {
    "generate": "generate",  # Loop back
    "end": END
})
```

### Router Pattern

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

def classifier(state: State) -> dict:
    """Classify user intent."""
    response = llm.invoke([
        {"role": "system", "content": "Classify intent: billing, support, or general"},
        {"role": "user", "content": state["messages"][-1].content}
    ])
    return {"intent": response.content.strip().lower()}

def route_by_intent(state: State) -> str:
    return state["intent"]

builder = StateGraph(State)
builder.add_node("classifier", classifier)
builder.add_node("billing", billing_agent)
builder.add_node("support", support_agent)
builder.add_node("general", general_agent)

builder.add_edge(START, "classifier")
builder.add_conditional_edges("classifier", route_by_intent)
builder.add_edge("billing", END)
builder.add_edge("support", END)
builder.add_edge("general", END)
```

---

## Multi-Agent Patterns

### Prebuilt ReAct Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = create_react_agent(
    model=llm,
    tools=[search],
    prompt="You are a helpful research assistant.",
)

# Run
result = agent.invoke({
    "messages": [{"role": "user", "content": "Research quantum computing"}]
})
```

### Supervisor Pattern

A supervisor agent delegates to specialized worker agents:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

class State(MessagesState):
    next: str

def supervisor(state: State) -> Command:
    """Decide which agent to call next."""
    response = llm.invoke([
        {"role": "system", "content": """You are a supervisor managing:
        - researcher: for finding information
        - writer: for creating content
        Respond with the agent name or FINISH."""},
        *state["messages"]
    ])
    
    next_agent = response.content.strip().lower()
    
    if next_agent == "finish":
        return Command(goto=END)
    
    return Command(
        update={"messages": [response]},
        goto=next_agent
    )

def researcher(state: State) -> Command:
    """Research agent."""
    result = llm.invoke([
        {"role": "system", "content": "You are a research specialist."},
        *state["messages"]
    ])
    return Command(
        update={"messages": [{"role": "assistant", "content": result.content, "name": "researcher"}]},
        goto="supervisor"  # Report back
    )

def writer(state: State) -> Command:
    """Writing agent."""
    result = llm.invoke([
        {"role": "system", "content": "You are a writing specialist."},
        *state["messages"]
    ])
    return Command(
        update={"messages": [{"role": "assistant", "content": result.content, "name": "writer"}]},
        goto="supervisor"  # Report back
    )

builder = StateGraph(State)
builder.add_node("supervisor", supervisor, ends=["researcher", "writer", END])
builder.add_node("researcher", researcher, ends=["supervisor"])
builder.add_node("writer", writer, ends=["supervisor"])

builder.add_edge(START, "supervisor")
graph = builder.compile()
```

### Using langgraph-supervisor Library

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Create specialized agents
research_agent = create_react_agent(llm, tools=[search_tool])
math_agent = create_react_agent(llm, tools=[calculator_tool])

# Create supervisor
supervisor = create_supervisor(
    agents=[research_agent, math_agent],
    model=llm,
    supervisor_name="coordinator"
)

graph = supervisor.compile()
```

### Hierarchical Teams

```python
from langgraph_supervisor import create_supervisor

# Level 2: Team supervisors
research_team = create_supervisor(
    agents=[search_agent, scrape_agent],
    model=llm,
    supervisor_name="research_supervisor"
).compile(name="research_team")

writing_team = create_supervisor(
    agents=[draft_agent, edit_agent],
    model=llm,
    supervisor_name="writing_supervisor"
).compile(name="writing_team")

# Level 1: Top-level supervisor
top_supervisor = create_supervisor(
    agents=[research_team, writing_team],
    model=llm,
    supervisor_name="project_manager"
).compile()
```

### Subgraphs

Embed one graph inside another:

```python
# Define inner graph
inner_builder = StateGraph(InnerState)
inner_builder.add_node("process", process_fn)
inner_builder.add_edge(START, "process")
inner_builder.add_edge("process", END)
inner_graph = inner_builder.compile()

# Use as node in outer graph
outer_builder = StateGraph(OuterState)
outer_builder.add_node("subworkflow", inner_graph)
outer_builder.add_edge(START, "subworkflow")
outer_builder.add_edge("subworkflow", END)
```

---

## Persistence & Memory

### Checkpointers

Checkpointers save state snapshots at each step:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# In-memory (development)
memory_checkpointer = InMemorySaver()

# SQLite (local persistence)
sqlite_checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL (production)
postgres_checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)

# Compile graph with checkpointer
graph = builder.compile(checkpointer=memory_checkpointer)
```

### Thread-Based Conversations

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Each thread maintains separate conversation history
config = {"configurable": {"thread_id": "user-123-session-1"}}

# First message
result1 = graph.invoke(
    {"messages": [{"role": "user", "content": "Hello, I'm Alice"}]},
    config=config
)

# Continue same thread - remembers context
result2 = graph.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config
)
# Agent remembers "Alice" from previous message
```

### Long-Term Memory (Store)

For cross-thread memory:

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

# Access store in nodes
def personalized_node(state, config, store):
    user_id = config["configurable"]["user_id"]
    
    # Retrieve user preferences
    prefs = store.get(("users", user_id, "preferences"))
    
    # Save new information
    store.put(("users", user_id, "preferences"), {"theme": "dark"})
    
    return {"messages": [...]}
```

### Time Travel

Replay or fork from any checkpoint:

```python
# Get all checkpoints for a thread
checkpoints = list(checkpointer.list(config))

# Load specific checkpoint
old_checkpoint_id = checkpoints[2].config["configurable"]["checkpoint_id"]
old_config = {
    "configurable": {
        "thread_id": "user-123",
        "checkpoint_id": old_checkpoint_id
    }
}

# Resume from that point (creates a fork)
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Try a different approach"}]},
    config=old_config
)
```

---

## Human-in-the-Loop

### Interrupt Before/After Node

```python
# Pause before executing a node
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["sensitive_action"]
)

# Pause after a node (for review)
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=["draft_generator"]
)
```

### Dynamic Interrupt

```python
from langgraph.types import interrupt

def requires_approval(state: State) -> dict:
    """Pause for human approval."""
    if state["risk_level"] == "high":
        # Pause execution, wait for human input
        human_response = interrupt({
            "question": "Approve this action?",
            "proposed_action": state["action"]
        })
        
        if human_response["approved"]:
            return {"status": "approved"}
        else:
            return {"status": "rejected", "reason": human_response["reason"]}
    
    return {"status": "auto_approved"}
```

### Resume After Interrupt

```python
# Initial run - will pause at interrupt
config = {"configurable": {"thread_id": "review-123"}}
result = graph.invoke({"messages": [...]}, config=config)

# ... human reviews and decides ...

# Resume with human input
graph.invoke(
    Command(resume={"approved": True, "comment": "Looks good"}),
    config=config
)
```

---

## Streaming

### Stream Events

```python
# Stream all events
async for event in graph.astream_events(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config=config,
    version="v2"
):
    print(event)

# Stream specific modes
for chunk in graph.stream(
    {"messages": [...]},
    config=config,
    stream_mode="updates"  # Only state updates
):
    print(chunk)

# Available modes: "values", "updates", "debug"
```

### Token-by-Token Streaming

```python
async for event in graph.astream_events(input_data, config=config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)
```

---

## Tools

### Basic Tool Definition

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression like '2 + 2' or '10 * 5'
    
    Returns:
        The result of the calculation
    """
    return str(eval(expression))

@tool
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the database for records.
    
    Args:
        query: Search query string
        limit: Maximum number of results (default: 10)
    """
    return [{"id": 1, "name": "Result"}]
```

### Tool with Context (LangGraph)

```python
from langgraph.prebuilt import InjectedStore
from typing import Annotated

@tool
def save_preference(
    key: str,
    value: str,
    store: Annotated[Any, InjectedStore]  # Injected by LangGraph
) -> str:
    """Save a user preference."""
    store.put(("preferences",), {key: value})
    return f"Saved {key}={value}"
```

### ToolNode (Prebuilt)

```python
from langgraph.prebuilt import ToolNode

tools = [search_tool, calculator_tool]
tool_node = ToolNode(tools)

builder.add_node("tools", tool_node)
```

---

## Quick Reference: Common Patterns

| Pattern | Implementation |
|---------|----------------|
| **Sequential Chain** | `add_edge(A, B)`, `add_edge(B, C)` |
| **Conditional Routing** | `add_conditional_edges(node, router_fn, mapping)` |
| **Parallel Fan-out** | Return `list[Send(...)]` from conditional edge |
| **Loop/Iteration** | Conditional edge that routes back to earlier node |
| **Supervisor** | Central node with `Command(goto=agent_name)` |
| **Hierarchical** | Nested supervisors via `langgraph-supervisor` |
| **Human-in-the-Loop** | `interrupt_before`, `interrupt_after`, or `interrupt()` |
| **Memory** | Compile with `checkpointer=` and use `thread_id` |
| **Long-term Memory** | Compile with `store=` for cross-thread persistence |

---

## Project Structure

```
my_agent/
├── agent.py           # Main agent definition
├── nodes/
│   ├── __init__.py
│   ├── research.py    # Research node functions
│   └── writer.py      # Writer node functions
├── tools/
│   ├── __init__.py
│   └── search.py      # Tool definitions
├── state.py           # State schema definitions
├── middleware.py      # Custom middleware (LangChain)
└── config.py          # Configuration and model setup
```

---

## Environment Configuration

```bash
# OpenAI
export OPENAI_API_KEY=your_key

# Anthropic
export ANTHROPIC_API_KEY=your_key

# LangSmith (optional, for tracing)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT=my_project
```

---

## When to Use What

| Use Case | Recommendation |
|----------|----------------|
| Simple tool-calling agent | `langchain.create_agent()` |
| Quick prototype | `langgraph.prebuilt.create_react_agent()` |
| Custom workflow logic | `StateGraph` with nodes/edges |
| Multi-agent coordination | `langgraph-supervisor` or custom `Command` routing |
| Production with persistence | LangGraph + Checkpointer |
| Need middleware (HITL, PII, etc.) | LangChain 1.0 middleware |
| Maximum control | Raw LangGraph `StateGraph` |

---

## Resources

- **LangChain Docs:** https://docs.langchain.com
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **GitHub (LangGraph):** https://github.com/langchain-ai/langgraph
- **LangChain Academy:** Free courses on LangGraph
- **LangSmith:** https://smith.langchain.com (tracing & evaluation)