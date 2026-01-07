[← Back to Home](/)

## Inter-application Agents Communication with Google A2A

The Agent2Agent (A2A) protocol is a communication protocol for artificial intelligence (AI) agents, initially introduced by Google in April 2025. This open protocol is designed for multi-agent systems, allowing interoperability between AI agents from varied providers or those built using different AI agent frameworks.

### Scope of Communication vs. LangGraph

| Aspect | LangGraph | Google A2A |
|--------|-----------|------------|
| **Scope** | **Intra-application** — agents within the same process/graph | **Inter-application** — agents across different systems, vendors, frameworks |
| **Network** | In-memory, same runtime | HTTP/gRPC over network |
| **Coupling** | Tightly coupled (shared state) | Loosely coupled (opaque, no shared internals) |
| **Discovery** | Compile-time (defined in graph) | Runtime via Agent Cards |

In summary, LangGraph handles **internal orchestration** (how agents work together within your system), while A2A handles **external interoperability** (how your agents talk to agents built by other teams/vendors/frameworks). They solve different problems and can be used together.

---

### A2A: The "Universal Translator" for Agents

While earlier agent orchestration frameworks like crewAI and LangChain automate multi-agent workflows within their own ecosystems, the A2A protocol acts as a messaging tier that lets these agents "talk" to each other despite their distinct agentic architectures. Think of A2A as a common language or universal translator for agent ecosystems.

**A2A Core Concepts:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     A2A Protocol Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DISCOVERY                                                   │
│     Client Agent ──────► /.well-known/agent.json (Agent Card)   │
│                          Returns: name, capabilities, auth      │
│                                                                 │
│  2. TASK CREATION                                               │
│     Client ──── JSON-RPC/HTTP ────► Remote Agent                │
│              "message/send"         Creates Task                │
│                                                                 │
│  3. COLLABORATION                                               │
│     ◄──────── Messages, Artifacts, Status Updates ────────►    │
│               (sync, streaming SSE, or push notifications)      │
│                                                                 │
│  4. COMPLETION                                                  │
│     Remote Agent returns Artifacts (task outputs)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### How They Relate

A2A aims to: Break Down Silos: Connect agents across different ecosystems. Enable Complex Collaboration: Allow specialized agents to work together on tasks that a single agent cannot handle alone. Promote Open Standards: Foster a community-driven approach to agent communication, encouraging innovation and broad adoption. Preserve Opacity: Allow agents to collaborate without needing to share internal memory, proprietary logic, or specific tool implementations, enhancing security and protecting intellectual property.

**Comparison Table:**

| Feature | LangGraph (Internal) | A2A (External) |
|---------|---------------------|----------------|
| **State Sharing** | Full — shared `State` dict | None — agents are opaque black boxes |
| **Discovery** | Hardcoded in graph definition | Dynamic via Agent Cards (`/.well-known/agent.json`) |
| **Communication** | Function calls, `Command` objects | JSON-RPC 2.0 over HTTP/gRPC |
| **Task Model** | Implicit (state transitions) | Explicit `Task` lifecycle with status |
| **Streaming** | Native Python generators | SSE (Server-Sent Events) |
| **Authentication** | N/A (same process) | OAuth, API keys, signed cards |
| **Vendor Lock-in** | LangGraph ecosystem | Vendor-neutral standard |

---

### A2A vs MCP (Model Context Protocol)

Google's documentation points out that A2A solves a different problem than MCP does: it "allows agents to communicate as agents (or as users) instead of as tools." The difference between a tool and an agent is that tools have structured I/O and behavior, while agents are autonomous and can solve new tasks using reasoning.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Protocol Complementarity                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   MCP (Model Context Protocol)                                  │
│   └── Agent ←→ Tools/Data Sources                               │
│       • Structured I/O                                          │
│       • Deterministic behavior                                  │
│       • Example: Database queries, API calls                    │
│                                                                 │
│   A2A (Agent-to-Agent)                                          │
│   └── Agent ←→ Agent                                            │
│       • Autonomous reasoning                                    │
│       • Multi-turn collaboration                                │
│       • Example: Negotiation, complex delegation                │
│                                                                 │
│   Use both together:                                            │
│   • MCP for tools/data access                                   │
│   • A2A for agent collaboration                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Practical Example: When to Use Each

**LangGraph Internal Communication** (same application):

```python
# Agents within ONE LangGraph application
# Tight coupling, shared state, same process

from langgraph.graph import StateGraph

class State(TypedDict):
    messages: list
    research: str  # Shared between agents

def researcher(state):
    return {"research": "findings..."}

def writer(state):
    # Direct access to researcher's output
    return {"draft": f"Based on {state['research']}"}

builder = StateGraph(State)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
# Same process, shared memory
```

**A2A External Communication** (across applications/vendors):

```python
# Agent in YOUR system talking to agent in ANOTHER system
# Loose coupling, opaque, over network

from a2a import A2AClient, SendMessageRequest

# 1. Discover remote agent via Agent Card
client = A2AClient("https://vendor-agent.example.com")
agent_card = client.get_agent_card()
# Returns: {"name": "SupplierAgent", "skills": ["inventory", "pricing"], ...}

# 2. Send task to remote agent
response = client.send_message(SendMessageRequest(
    message={
        "role": "user",
        "parts": [{"text": "Check inventory for SKU-12345"}]
    }
))

# 3. Remote agent processes (you don't see internals)
# 4. Receive artifacts (results)
task = response.task
if task.status.state == "completed":
    result = task.artifacts[0].parts[0].text
```

---

### LangChain/LangGraph + A2A Integration

We're releasing native support for A2A in Agent Development Kit (ADK), a powerful open source agent framework released by Google. This makes it easy to build A2A agents if you are already using ADK.

LangChain was listed as a partner for A2A, and integration is possible:

```python
# LangGraph agent that can call external A2A agents

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from a2a import A2AClient

@tool
def delegate_to_supplier(query: str) -> str:
    """Delegate inventory queries to external supplier agent via A2A."""
    client = A2AClient("https://supplier.example.com")
    response = client.send_message({
        "role": "user",
        "parts": [{"text": query}]
    })
    return response.task.artifacts[0].parts[0].text

# LangGraph agent with A2A tool
agent = create_react_agent(
    model=llm,
    tools=[delegate_to_supplier, local_tools...]
)
```

---

### Summary: Complementary, Not Competing

| Use Case | Solution |
|----------|----------|
| Agents **within** your LangGraph application | LangGraph shared state, `Command` routing |
| Agents **across** different frameworks/vendors | A2A protocol |
| Agent **accessing tools/databases** | MCP (Model Context Protocol) |
| **Combining all three** | LangGraph for internal orchestration, MCP for tools, A2A for external agents |

A2A tries to complement MCP where A2A is focused on a different problem, while MCP focuses on lowering complexity to connect agents with tools and data, A2A focuses on how to enable agents to collaborate in their natural modalities. It allows agents to communicate as agents (or as users) instead of as tools.
