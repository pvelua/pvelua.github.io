[← Back to Home](/)

## Inter-application Agents Communication with Google A2A

The Agent2Agent (A2A) protocol is a communication protocol for artificial intelligence (AI) agents, initially introduced by Google in April 2025. This open protocol is designed for multi-agent systems, allowing interoperability between AI agents from varied providers or those built using different AI agent frameworks.

### Scope of Communication vs. LangGraph

| Aspect | LangGraph | Google A2A |
|:-------|:----------|:-----------|
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
|:--------|:--------------------|:---------------|
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

The **Agent2Agent (A2A)** protocol is an open standard (recently donated by Google to the Linux Foundation) that allows AI agents to "discover" and talk to each other regardless of the framework they were built in (LangChain, ADK, CrewAI, etc.).

In the context of **Google Agent Developer Kit (ADK)**, using A2A allows your LangGraph agent to act as a "client" that calls other specialized agents, or as a "server" that other agents can call.

Here is the concise guide to connecting them.

#### 1\. The Concept: "Agent Cards"

A2A relies on a **Discovery** mechanism. Every A2A-compliant agent must publish an **Agent Card** (a JSON file usually at `.well-known/agent.json`) that tells others:

  * **Identity:** Name and ID.
  * **Capabilities:** What it can do (Input/Output schemas).
  * **Auth:** How to connect (OIDC, Keys).

#### 2\. The Setup (Server Side)

First, you typically build a specialized agent using the **Google ADK** and "expose" it as an A2A endpoint.

**Install ADK:**

```bash
uv add google-adk
```

**Create the Agent Server:**
This code spins up a server that LangGraph can talk to.

```python
# server.py
from google.adk.agents import Agent
from google.adk.models import VertexGenAIModel
from google.adk.a2a.utils.agent_to_a2a import to_a2a
import uvicorn

# 1. Define the ADK Agent
model = VertexGenAIModel(model_name="gemini-1.5-flash")
specialized_agent = Agent(
    name="financial_analyst",
    instruction="You are an expert financial analyst. Analyze the provided stock data.",
    model=model
)

# 2. Wrap it with A2A Protocol (Auto-generates the Agent Card)
# This creates a FastAPI app compliant with A2A specs
a2a_app = to_a2a(specialized_agent, port=8000)

# 3. Run it
if __name__ == "__main__":
    uvicorn.run(a2a_app, host="0.0.0.0", port=8000)
```

#### 3\. The Integration (LangGraph Client Side)

Now, inside your **LangGraph** application, you treat this external A2A agent as a **Tool**. You don't need to manually parse the JSON; you can just send a standardized "Task".

```python
import httpx
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START

# --- 1. Define the A2A Client Tool ---
@tool
def call_financial_agent(query: str):
    """Delegates a financial question to the specialized A2A Analyst Agent."""
    
    # A2A uses a standardized '/run' or '/tasks' endpoint
    url = "http://localhost:8000/run" 
    
    payload = {
        "task": {
            "input": query,
            # A2A allows passing context/history if needed
        }
    }
    
    # In production, you would add OIDC/Auth headers here
    response = httpx.post(url, json=payload, timeout=30.0)
    return response.json().get("result")

# --- 2. Use in LangGraph ---
# Add this tool to your LangGraph node like any other
def orchestrator_node(state: MessagesState):
    # Logic to decide to call the tool...
    # For demo, we assume we call it directly:
    result = call_financial_agent.invoke(state["messages"][-1].content)
    return {"messages": [result]}

# ... (Standard LangGraph Setup: Workflow, Edges, Compile) ...
```

### Summary of the Flow

1.  **Discovery:** The generic agent (LangGraph) checks the `agent.json` of the specialist (ADK) to see if it can handle the task (optional, or hardcoded for simple use).
2.  **Handshake:** LangGraph sends a `Task` object over HTTP.
3.  **Execution:** The ADK agent processes it using its internal logic/tools.
4.  **Response:** The ADK agent returns a `Result` object.

### Why use this over a standard API?

  * **Standardization:** If you switch the backend of the "Financial Agent" from ADK to AutoGen or CrewAI later, the protocol (inputs/outputs) remains the same.
  * **Security:** A2A has built-in patterns for passing **Identity Tokens**, so the sub-agent knows *exactly* which user is asking (crucial for enterprise permissions).

[How to build an AI agent with MCP, ADK, and A2A](https://www.youtube.com/watch?v=6mQwHqK1I5w) - this YouTube video provides a complete walkthrough of building an agent with the ADK and exposing it via the A2A protocol, perfectly matching the setup described above.

---

### Summary: Complementary, Not Competing

| Use Case | Solution |
|:---------|:---------|
| Agents **within** your LangGraph application | LangGraph shared state, `Command` routing |
| Agents **across** different frameworks/vendors | A2A protocol |
| Agent **accessing tools/databases** | MCP (Model Context Protocol) |
| **Combining all three** | LangGraph for internal orchestration, MCP for tools, A2A for external agents |

A2A tries to complement MCP where A2A is focused on a different problem, while MCP focuses on lowering complexity to connect agents with tools and data, A2A focuses on how to enable agents to collaborate in their natural modalities. It allows agents to communicate as agents (or as users) instead of as tools.

---

### Referneces

- [Agent2Agent (A2A) Protocol](https://a2a-protocol.org/latest/)
- [What is A2A protocol (Agent2Agent)?](https://www.ibm.com/think/topics/agent2agent-protocol)
- [Google Open-Sources Agent2Agent Protocol for Agentic Collaboration](https://www.infoq.com/news/2025/04/google-agentic-a2a/)
