[← Back to Home](/)

## Intra-Application Agent Communication in LangGraph

LangGraph provides **three primary mechanisms** for agents to communicate whithin the sam process/graph:

### 1. Shared State (Most Common)

Agents communicate by reading from and writing to a shared state object. This is the default pattern.

```python
from typing import Annotated
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages

class State(MessagesState):
    # Shared message history - all agents can read/write
    messages: Annotated[list, add_messages]
    
    # Shared data channels
    research_findings: str
    draft_content: str

def researcher(state: State) -> dict:
    # Write to shared state
    return {"research_findings": "Found important data..."}

def writer(state: State) -> dict:
    # Read from shared state
    findings = state["research_findings"]
    return {"draft_content": f"Based on: {findings}"}
```

**Characteristics:**
- All agents see the complete shared state
- Good for transparency — every agent can see all work done
- Can become verbose if agents produce many intermediate steps
- Uses **reducers** (like `add_messages`) to define how updates merge

---

### 2. LLM-Driven Delegation (Dynamic Handoffs)

A parent agent uses its LLM reasoning to decide which sub-agent should handle a task, based on agent descriptions:

```python
from langgraph.prebuilt import create_react_agent

# Sub-agents with clear descriptions
billing_agent = create_react_agent(
    llm,
    tools=[billing_tools],
    name="billing_agent",
    description="Handles billing inquiries, payments, and refunds"
)

support_agent = create_react_agent(
    llm,
    tools=[support_tools],
    name="support_agent", 
    description="Handles technical support and troubleshooting"
)

# Coordinator delegates based on descriptions
coordinator = LlmAgent(
    model=llm,
    instruction="Route requests to the appropriate specialist.",
    sub_agents=[billing_agent, support_agent]
)
```

The LLM reads the `description` fields and decides which agent to invoke.

---

### 3. Explicit Handoffs via `Command`

Agents explicitly control routing by returning `Command` objects that specify the next agent and state updates:

```python
from langgraph.types import Command

def agent_alice(state: State) -> Command:
    result = process_task(state)
    
    # Explicit handoff to bob with state update
    return Command(
        update={"messages": [{"role": "assistant", "content": result}]},
        goto="agent_bob"  # Explicit next agent
    )

def agent_bob(state: State) -> Command:
    # Can hand back to alice or finish
    if needs_more_research(state):
        return Command(goto="agent_alice")
    return Command(goto=END)
```

**For cross-graph communication** (subgraph to parent):

```python
def subgraph_node(state: State) -> Command:
    return Command(
        update={"result": "completed"},
        goto="parent_node",
        graph=Command.PARENT  # Navigate to parent graph
    )
```

---

### 4. Tool-Based Handoffs (Agent as Tool)

Wrap agents as tools that other agents can explicitly call:

```python
from langgraph.prebuilt import create_react_agent

# Create specialist agent
research_agent = create_react_agent(llm, tools=[search_tool])

# Wrap as a tool
@tool
def delegate_research(query: str) -> str:
    """Delegate research task to the research specialist."""
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# Main agent can call research as a tool
main_agent = create_react_agent(
    llm,
    tools=[delegate_research, other_tools]
)
```

---

### 5. Message-Based Communication

Agents communicate through a shared message list, often with agent identification:

```python
from langchain_core.messages import HumanMessage, AIMessage

def agent_with_identity(state: State) -> dict:
    response = llm.invoke(state["messages"])
    
    # Tag message with agent name
    return {
        "messages": [
            AIMessage(
                content=response.content,
                name="research_agent"  # Identifies sender
            )
        ]
    }
```

For providers that don't support the `name` parameter, you can inject identity into content:

```python
tagged_message = f"<agent>research_agent</agent><message>{response}</message>"
```

---

### Communication Pattern Comparison

| Pattern | Control Flow | State Visibility | Best For |
|---------|--------------|------------------|----------|
| **Shared State** | Predetermined edges | Full transparency | Sequential pipelines, debugging |
| **LLM Delegation** | Dynamic (LLM decides) | Full transparency | Flexible routing, natural language |
| **Command Handoffs** | Explicit in code | Controlled | Complex workflows, conditional logic |
| **Agent as Tool** | Caller controls | Encapsulated | Specialist delegation, modular design |
| **Messages** | Via message history | Conversation-based | Chat-like collaboration |

---

### Supervisor vs. Peer-to-Peer

**Supervisor Pattern** (hub-and-spoke):
```
         ┌─────────────┐
         │  Supervisor │
         └──────┬──────┘
        ┌───────┼───────┐
        ▼       ▼       ▼
    Agent A  Agent B  Agent C
```
- Supervisor controls all communication
- Workers always report back to supervisor
- Centralized decision-making

**Network Pattern** (peer-to-peer):
```
    Agent A ◄──► Agent B
        ▲           ▲
        │           │
        └───► Agent C ◄───┘
```
- Any agent can communicate with any other
- Decentralized, more complex
- Use `Command(goto="agent_name")` for direct handoffs

---

### Handling Tool Call Messages in Handoffs

When agents hand off mid-conversation with pending tool calls, you may need to handle orphaned tool call messages:

```python
from langchain_core.messages import ToolMessage

def handoff_node(state: State) -> dict:
    last_message = state["messages"][-1]
    
    # If there are pending tool calls, add placeholder responses
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        placeholder_responses = [
            ToolMessage(
                content="Handoff occurred, tool not executed",
                tool_call_id=tc["id"]
            )
            for tc in last_message.tool_calls
        ]
        return {"messages": placeholder_responses}
    
    return {}
```

---

### Private State for Agents

If agents need isolated state:

```python
# Option 1: Subgraph with separate schema
class AgentPrivateState(TypedDict):
    internal_notes: str  # Only this agent sees this

agent_subgraph = StateGraph(AgentPrivateState)
# ... define agent's internal workflow

# Option 2: Private input schema
def agent_node(state: PrivateInputState) -> PublicOutputState:
    # state contains only what this agent needs
    pass

builder.add_node("agent", agent_node, input=PrivateInputState)
```

This gives you the flexibility to share only what's needed between agents while keeping internal reasoning private.