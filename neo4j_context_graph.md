[← Back to Home](/)

# Neo4j Context Graph

The development of **Context Graphs** in early 2026 marks a shift from simple data retrieval to "cognitive" architectures for AI. While a Knowledge Graph focuses on static facts (entities and relationships), a **Context Graph** adds a "third dimension" by recording the temporal and reasoning layers of an agent's life.

## (a) The Neo4j Context Graph
In 2026, Neo4j introduced the Context Graph as a superset of the Knowledge Graph. It doesn't just store "what is true" (e.g., *User A works at Company B*), but also "how we know it" and "what we did with it."

### Core Components
* **Knowledge Layer:** The traditional entities, properties, and semantic relationships (The "Source of Truth").
* **Trace Layer:** Decision traces that link a user's request to specific tool calls, intermediate reasoning steps, and the final output.
* **Temporal Layer:** Versioning of facts and relationships. It allows an agent to understand how context has changed over time (e.g., *User A used to prefer Python but now prefers Svelte*).
* **Provenance Layer:** Direct links between an agent's response and the specific data nodes that influenced it, enabling **queryable explainability**.

## (b) Supporting Agentic Memory
Neo4j’s graph-native architecture allows it to serve as a unified "brain" that categorizes information into four distinct functional memory types.

### 1. Short-Term (Working) Memory
* **Role:** Maintains the state of the current session or task.
* **Graph Implementation:** Stores the current conversation thread, recent tool outputs, and intermediate variables as a sequence of connected nodes.
* **Benefit:** Prevents "context rot" by allowing the agent to traverse back through the recent chain of thought without re-processing thousands of tokens in the LLM's context window.

### 2. Long-Term (Durable) Memory
* **Role:** Persistent storage of facts and preferences that survive across sessions.
* **Graph Implementation:** Uses **GraphRAG** (Knowledge Graph + Vector Search).
* **Benefit:** When a user mentions a preference (e.g., "I use UV for Python"), it is stored as a persistent node. Future sessions retrieve this via a simple graph hop rather than a fuzzy semantic search.

### 3. Episodic Memory (Experience)
* **Role:** Remembers specific events, sequences, and past successes/failures.
* **Graph Implementation:** Records "Episodes"—triples of *[Prompt -> Action -> Feedback]*.
* **Benefit:** Enables **Few-Shot Learning**. If an agent previously struggled with a complex Cypher query but eventually succeeded, it can retrieve that "episode" to guide its reasoning for a similar new request.

### 4. Procedural & Reasoning Memory
* **Role:** Stores the "how-to"—system prompts, tool descriptions, and learned reasoning patterns.
* **Graph Implementation:** Tools and system instructions are represented as nodes. 
* **Benefit:** Allows for **Agentic GraphRAG**, where the agent can "see" its own tools and documentation as part of the graph, helping it decide which tool is most appropriate for a multi-hop reasoning task.

---

### Comparison: Vector Memory vs. Context Graph Memory
| Feature | Vector-Only Memory | Neo4j Context Graph |
| :--- | :--- | :--- |
| **Structure** | Flat list of text chunks | Interconnected network |
| **Relationships** | Implicit (semantic similarity) | Explicit (defined links) |
| **Explainability** | Low ("It's similar to this") | High ("Because of Step A and Fact B") |
| **Temporal Awareness** | Difficult (requires metadata filtering) | Native (traversable time-links) |


## Context Graph and Its Components

The **Context Graph** is not a replacement for your main data model, but a specialized **subgraph** (or "meta-layer") that wraps around it. 

If your main knowledge graph is the "world" (the *what*), the Context Graph is the "observer's log" (the *why*, *how*, and *when*). In a Neo4j implementation, these are often differentiated by labels and specific relationship types that bridge the two layers.

### 1. Concrete Examples of the Core Components

Using your **Retail Shopping / BI Reporting** project as a backdrop, here is how the components manifest as actual nodes and relationships:

#### A. The Knowledge Layer (Your Existing Graph)
This is your "Ground Truth." It stores the persistent domain entities.
* **Nodes:** `(:Customer)`, `(:Product)`, `(:Order)`, `(:AnalyticView)`.
* **Example:** `(c:Customer {id: "Igor"})-[:PURCHASED]->(p:Product {name: "Neo4j license"})`.

#### B. The Trace Layer (Reasoning Memory)
This records the agent's internal "Chain of Thought" and actions. It connects the *intent* to the *result*.
* **Nodes:** `(:Decision)`, `(:TraceStep)`, `(:ToolCall)`.
* **Example:** A **Decision** node represents the overall task. It is connected to **TraceSteps** (the agent's reasoning) and **ToolCalls** (the actual Cypher query it ran against your Oracle Analytic Views).
* **Bridge:** A `(:ToolCall)` node will have a relationship `[:ACCESSED]` pointing directly to a `(:Product)` or `(:AnalyticView)` node in your Knowledge Layer.

#### C. The Temporal Layer (Versioned State)
This tracks how things change. It prevents your agent from having "time-blindness."
* **Nodes:** `(:Snapshot)`, `(:TimeNode)`.
* **Example:** If a customer's `loyalty_tier` changes, you don't just overwrite the property. You create a new version of the relationship linked to a **TimeNode**.
* **Usage:** This allows the agent to answer: *"Why did I recommend this product last week but not today?"* (The context changed).

#### D. The Provenance Layer (Evidence)
This acts as the "Citations" for the agent's brain.
* **Relationships:** `[:BASED_ON]`, `[:CITED]`.
* **Example:** `(agent_response)-[:BASED_ON]->(trace_step)-[:CITED]->(oracle_db_record)`.


### 2. Visualizing the Architecture
Imagine your Knowledge Graph is a flat map. The Context Graph is a glass sheet laid on top of it, with pins and strings (relationships) dropping down into the map to show where the agent was looking at any given moment.


### 3. How it Fits Into Your Data Model
In Neo4j, you bridge your **ENGRAM (Hippocampal)** system and your **Long-term Knowledge Graph** using "Anchor" nodes.

| Feature | Knowledge Graph (LTM) | Context Graph (ENGRAM/Reasoning) |
| :--- | :--- | :--- |
| **Stability** | High (Facts) | Low (Ephemeral/Session-based) |
| **Query Pattern** | "What is the price of X?" | "What was the agent's plan for Task Y?" |
| **Connectivity** | Entity-to-Entity | Trace-to-Entity |

### Concrete Implementation Scenario:
1.  **Extraction:** Your ENGRAM extracts **NounPhrases** (e.g., "Oracle Analytic View").
2.  **Linking:** The system checks if "Oracle Analytic View" already exists in the **Knowledge Layer**.
3.  **Context Injection:**
    * If it exists, the new **Decision Trace** in the Context Graph creates a `[:REFERS_TO]` relationship to the existing node.
    * If it doesn't, it creates a new "Candidate Entity" node in the Context Graph for later consolidation into the Knowledge Layer.

### Why this is a "Subgraph"
You can effectively filter your views:
* **`MATCH (n:Domain)`** -> Shows your retail business model.
* **`MATCH (n:Trace)`** -> Shows your agent's audit log.
* **`MATCH (n:Trace)-[:ACCESSED]->(m:Domain)`** -> Shows exactly which parts of your business data the AI is "thinking" about.

---
