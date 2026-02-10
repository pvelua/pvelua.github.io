[← Back to Home](/)

## Google A2A - Communication Between Agents Regadless of Agent Platform

The **Google Agent2Agent (A2A) Protocol**, is an open standard designed to standardize communication between artificial intelligence (AI) agents across different AI agent frameworks, platforms, and vendors.

### In Summary

- **What it is**: A standardized communication protocol that enables AI agents to "talk" to one another, regardless of whether they were built with LangChain, CrewAI, Google ADK, or custom Python scripts.
- **Core Goal**: To move from isolated "siloed" agents to interoperable multi-agent ecosystems where agents can discover teammates, delegate tasks, and stream results.
- **Key Difference**: Unlike the Model Context Protocol (MCP), which connects agents to tools/data (internal capabilities), A2A connects agents to other agents (collaboration).

### Core Architecture

The A2A protocol adopts a standard Client-Server model over HTTP/HTTPS, treating agents as web services.

| Component	                | Role             |
| :------------------------ | :--------------- |
| **Client Agent**          | The initiator. It discovers a remote agent, sends a task (request), and listens for updates. |
| **Remote Agent (Server)** | The worker. It exposes an A2A-compliant endpoint, accepts tasks, executes them, and streams back results. |
| **Transport Layer**       | HTTP/HTTPS for control messages (JSON-RPC 2.0) and Server-Sent Events (SSE) for real-time streaming of progress/partial thoughts. |

### Key Concepts & Data Structures

To implement A2A, you must understand its four fundamental primitives:

#### A. The Agent Card (Discovery)

This is the "business card" or "resume" of an agent. It is a JSON metadata file typically hosted at `/.well-known/agent.json` that allows other agents to automatically discover what this agent can do.

- **Contains**: Agent name, description, list of **skills**, input/output modalities (text, image, audio), and authentication schemes (e.g., OAuth2, Bearer token).
- **Purpose**: Allows a "Manager Agent" to crawl a directory and dynamically select the right "Worker Agent" for a specific job without hard-coded integrations.

#### B. The Task (Unit of Work)

A2A formalizes work into **Tasks**. A task has a lifecycle: `SUBMITTED -> WORKING -> INPUT_REQUIRED (optional) -> COMPLETED / FAILED`
 - **Why it matters**: Unlike a simple API call that times out, a Task is stateful. It supports long-running processes (e.g., "Research this topic for 3 hours") where the connection might drop and reconnect.

#### C. The Message (Communication)

While a task is running, agents exchange Messages.
- **Structure**: Messages contain a `role` (user/agent) and a list of `parts`.
- **Multi-modal**: A message isn't just text. It can contain `TextPart`, `FilePart` (images/PDFs), or `DataPart` (structured JSON).

#### D. The Artifact (Deliverable)

The final or intermediate output of a task is an **Artifact**.
- **Example**: If you ask a "Coder Agent" to write a script, the conversation is the Message history, but the final Python file it generates is the Artifact.

### A2A vs. MCP: The "Tools vs. Team" Distinction

It is critical not to confuse A2A with the Model Context Protocol (MCP).

| Feature       | Model Context Protocol (MCP) | Agent2Agent Protocol (A2A) |
| :------------ | :--------------------------- | :------------------------- |
| **Primary Focus** | Tools & Data                 | Collaboration & Delegation |
| **Analogy**       | Giving a human a calculator or a library card. | Two humans talking to solve a problem together. |
| **Connection**    | Agent <-> Database/API       | Agent <-> Agent |
| **Use Case**      | "Query the SQL database for sales data." | "Ask the Sales Agent to analyze this quarter's trends." |

*Design Tip: Use MCP to give your agent skills (internal). Use A2A to let your agent delegate work to others (external).*

### How to Use A2A for Support Inter-Agent Communication

To build a ficticious support system where a "Triage Agent" delegates to specialized agents (e.g., "Refund Agent," "Technical Agent"):

**Step 1: Define the Remote Agents (The Specialists)**

Create your specialized agents (e.g., using Google ADK or LangChain). Expose them as A2A servers.
- **Action**: Create an `agent.json` for the "Refund Agent" that lists a skill: `process_refund`.
- **Endpoint**: Ensure the agent listens for POST requests on an A2A-compliant route (e.g., `/a2a/v1/tasks`).

**Step 2: Implement the Client Agent (The Triage)**

The "Triage Agent" does not need to know how to process a refund; it only needs to know who can.
- **Discovery**: The Client loads the a`gent.jso`n` from the Refund Agent's URL.
- **Task Creation**: The Client sends a `CREATE_TASK` request with the user's complaint as the input.

**Step 3: Handle the Lifecycle (The Loop)**

- **Streaming**: The Refund Agent sends SSE events: "Verifying order ID..." -> "Checking policy..." -> "Refund Approved."
- **Intervention**: If the Refund Agent gets stuck (e.g., "Receipt image unclear"), it updates the task status to `INPUT_REQUIRED`. The Triage Agent forwards this request to the user, gets the image, and sends it back to the Refund Agent as a new Message.

**Step 4: Final Artifact**

The Refund Agent marks the task `COMPLETED` and returns an **Artifact** (e.g., a PDF receipt of the refund). The Triage Agent presents this to the user.

**Summary of Benefits**

1. **Decoupling**: You can upgrade the "Refund Agent" from GPT-4.5 to Gemini 3.5 Pro without changing a single line of code in the "Triage Agent."
2. **Polyglot**: The Triage Agent can be Python; the Refund Agent can be Node.js/TypeScript.
3. **Resilience**: Long-running tasks don't break if the HTTP connection momentarily blips, thanks to the stateful Task lifecycle.

---

## Python Code Skeleton for Google A2A

This section contaions a sample Python code skeleton for using the Google Agent Development Kit (ADK) to demonstrate a simple implementation of Manager -> Worker communication via A2A protocol. This example simulates a scenario where a Manager Agent (Client) delegates a complex calculation to a specialized Math Worker Agent (Server).

**Rerequisites**

Puthon project requires installing `google-adk` library e.g., using `pip install google-adk`.

### 1. The Worker Agent (Server)

This script runs as a web service. It defines an agent, exposes a "skill" (tool), and listens for A2A requests.
```python
import os
from google.adk import Agent, Task, Server
from google.adk.types import ModelClient

# --- Configuration ---
# In a real app, load these from environment variables
PORT = 8080
AGENT_NAME = "Math-Whiz-Worker"

# --- Define the 'Skill' ---
# This function is the actual work the agent performs.
# A2A automatically wraps this into a tool definition.
def perform_complex_calculation(expression: str) -> str:
    """
    Evaluates a mathematical expression safely.
    
    Args:
        expression: The mathematical string to evaluate (e.g., "2 + 2").
    """
    try:
        # In a real scenario, use a safe evaluation library, not eval()
        result = eval(expression) 
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Initialize the Worker Agent ---
worker_agent = Agent(
    name=AGENT_NAME,
    model_client=ModelClient(model="gemini-1.5-pro"), # The brain of the worker
    instructions="You are a helpful math assistant. Use the provided tools to solve problems.",
    tools=[perform_complex_calculation] # Register the skill
)

# --- Define the A2A Server ---
# This wrapper handles the HTTP endpoints for:
# - /.well-known/agent.json (Discovery)
# - /a2a/v1/tasks (Task Creation)
# - /a2a/v1/tasks/{id} (Status Checks)
server = Server(agent=worker_agent)

if __name__ == "__main__":
    print(f"Starting {AGENT_NAME} on port {PORT}...")
    # This automatically generates the 'agent.json' based on the tools and description
    server.run(host="0.0.0.0", port=PORT)
```

### 2. The Manager Agent (Client)

This script runs as a client. It discovers the worker, creates a task, and streams the results.
```python
import asyncio
from google.adk import Agent, A2AClient
from google.adk.types import Message

# --- Configuration ---
WORKER_URL = "http://localhost:8080" # The address of the Worker Agent

async def main():
    # --- Step 1: Initialize the Manager ---
    # The manager doesn't need tools; its "tool" is the other agent.
    manager = Agent(
        name="Project-Manager",
        instructions="You are a manager. Delegate math problems to the specialist worker."
    )

    # --- Step 2: Connect to the Worker ---
    # The A2AClient handles the handshake and protocol negotiation.
    # It fetches the 'agent.json' from the worker to understand its capabilities.
    print(f"Connecting to worker at {WORKER_URL}...")
    client = A2AClient(base_url=WORKER_URL)
    
    # Optional: Verify the worker's identity/capabilities
    worker_info = await client.get_agent_info()
    print(f"Connected to: {worker_info.name} - Capabilities: {worker_info.skills}")

    # --- Step 3: Create a Task ---
    # The manager formulates the request.
    task_prompt = "Calculate (15 * 4) + 100 and explain the steps."
    
    print(f"\nSending task: '{task_prompt}'")
    task_handle = await client.create_task(
        prompt=task_prompt,
        context={"priority": "high"} # Optional metadata
    )

    # --- Step 4: Stream the Execution (The Loop) ---
    # A2A allows us to watch the worker "think" and act in real-time.
    print("\n--- Worker Execution Stream ---")
    async for event in task_handle.stream_events():
        if event.type == "thought":
            # The worker is reasoning (CoT)
            print(f"[Worker Thought]: {event.content}")
        elif event.type == "tool_call":
            # The worker is using the 'perform_complex_calculation' tool
            print(f"[Worker Action]: Calling tool '{event.tool_name}' with '{event.args}'")
        elif event.type == "output":
            # The worker is speaking back to us
            print(f"[Worker Output]: {event.content}")
        elif event.type == "error":
             print(f"[Worker Error]: {event.content}")

    # --- Step 5: Get Final Result ---
    # Once the stream finishes, we can retrieve the final artifact or summary.
    final_result = await task_handle.get_result()
    print(f"\n--- Final Result ---\n{final_result.output}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Key Implementation Details

1. **Server(agent=worker_agent)**: This is the magic line. In the ADK, wrapping an agent in `Server` automatically sets up the FastAPI (or similar) routes required by the A2A protocol. You don't need to manually write `@app.post("/tasks")`.
2. **Discovery**: When the Client connects, it hits the `/.well-known/agent.json` endpoint on the Worker. This JSON file (generated by the ADK) tells the Client: "I am Math-Whiz-Worker, and I have a tool called `perform_complex_calculation`."
3. **Stateful Task Handle**: The `task_handle` object in the client script maintains the connection ID. If the network drops, you can theoretically reconnect to the same `task_id` later to check the status, rather than restarting the job.

---

## Google A2A vs. Agent Platform

Both Google A2A and an Agent Platform support inter-agent communication and collaboration. The key difference is that A2A Protocol is designed to support communication between agents built using different platforms or agent frameworks. Agent Framework, like LangChain or CrewAI allow communication between agents built using the same framework i.e., running within the same application platform e.g., agent running within the same LangChain powered application can communicate and collaborate.

Let's look at the A2A vs. Agent Platformc differences in more details using LangChain as an example.

### Scope of Communication A2A vs. LangGraph

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
