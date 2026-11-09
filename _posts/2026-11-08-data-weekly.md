---
title: "Data and Orchestration Weekly — 8 November 2026"
date: 2026-11-08
categories: [data]
summary: "Apache Airflow 3.4 adds task loops, non-Python Dags and generally available LLM and agent tasks; LangChain ships async subagents."
item_count: 6
tags: [airflow, orchestration, agents, serving, vector-search, ontology]
---

### [Airflow 3.4 adds task loops, Dags in other languages and generally available agent tasks](https://airflow.apache.org/blog/airflow-3.4.0/)

**Apache Airflow** · 6 Nov 2026 · *Orchestration*

A task group can now repeat, passing each pass's result to the next and stopping early on a condition, with a mandatory iteration cap. Whole Dags can be written in TypeScript, Java or Go, and the common.ai provider, which runs LLM calls and tool-using agents as ordinary tasks and can pause for human approval without holding a worker slot, is now generally available. Loops cannot be nested or mapped, and the language SDKs, the Dag importer and several agent features are still labelled experimental.

### [LangChain adds async subagents so an orchestrator can keep talking while work runs](https://www.langchain.com/blog/agents-are-going-async)

**LangChain** · 5 Nov 2026 · *Agent frameworks*

Deep Agents in Python and JavaScript, and the managed offering, can now launch background subagents through five orchestrator tools: start, check, update, cancel and list. A per-subagent setting decides whether a follow-up instruction interrupts and restarts the run or waits in a queue, and results reach the orchestrator differently depending on whether it is busy, idle or paused for approval. Task notifications are in beta, the interrupt option discards progress, and the post offers no performance data.

### [Buoyant open-sources Kinference, a control plane that sits in front of vLLM on Kubernetes](https://vllm.ai/blog/2026-11-04-introducing-kinference)

**vLLM blog** · 4 Nov 2026 · *Serving*

The Apache 2.0 project accepts OpenAI- and Anthropic-style traffic and adds token rate limits, per-tenant cost tracking and a load balancer that weighs GPU queue depth and KV-cache contents. YAML policies can route by model, tenant or prompt classification, for example keeping PII-tagged prompts on local GPUs or bursting to a cloud provider when the pool is full. Buoyant's own tests report about 7.4 times the goodput of standard Kubernetes balancing on a shared-prefix workload; the policy API is still v1alpha1.

### [LangChain compares seven browser toolsets for agents and tunes the winner](https://www.langchain.com/blog/how-we-optimized-browser-use-for-deep-agents)

**LangChain** · 3 Nov 2026 · *Agent tooling*

On 100 WebVoyager tasks every toolset passed between 78% and 90%, but Stagehand reached 90% for $1.98 against $26.71 for Playwright MCP at 78%, using about 10 steps per task versus 26. Compact text snapshots, a code-writing run tool, and a wait that ends when a page settles rather than a fixed sleep drove the savings. Samples are small, one benchmark belongs to Browserbase, which the authors work closely with, and the wait fix was measured alongside other changes.

### [Neo4j publishes an open specification for describing ontologies to software and agents](https://neo4j.com/blog/developer/the-neo4j-ontology-graph-specification-a-pragmatic-open-ai-first-approach/)

**Neo4j** · 5 Nov 2026 · *Knowledge graphs*

The Apache 2.0 Ontology Graph Specification defines a domain's nodes, relationships, constraints and business meaning in one YAML or JSON file checked against a versioned JSON Schema, with SDKs for the JVM, JavaScript and Go. Indexes and data-source mappings live in separate extensions, and RDFS and SHACL are covered by documented mappings rather than native support. It is a single-vendor proposal with no adoption figures yet, and no RDF serialisation exists.

### [Qdrant 1.20 adds rotation-based 8- and 16-bit vector types and per-element nested indexes](https://qdrant.tech/blog/qdrant-1.20-performance/)

**Qdrant** · 2 Nov 2026 · *Vector stores*

The new turbo8 and turbo16 datatypes store vectors in a quarter and a half of float32 space, about 147 MB and 293 MB against 585 MB for 100,000 1,536-dimension vectors, at roughly 95% recall and within about 10% of float32 latency, with no rescoring step. A new per-element index for nested payload fields served up to four times the queries per second of regular nested indexes on one million arXiv papers. All figures come from Qdrant's own tests; this complements the full-text changes in the same release covered earlier.
