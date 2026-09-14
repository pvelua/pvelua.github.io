---
title: "Data and Orchestration Weekly — 14 September 2026"
date: 2026-09-14
categories: [data]
summary: "AWS adds direct long-term memory ingestion for Bedrock agents, headlining a week of vector-store, MCP, and multi-agent context updates."
item_count: 6
tags: [agent-memory, vector-search, mcp, orchestration]
---

### [AWS lets developers write straight into an agent's long-term memory](https://aws.amazon.com/about-aws/whats-new/2026/09/agentcore-memory-direct-ingest/)

**Amazon Web Services** · 8 Sep 2026 · *Agent memory*

Amazon Bedrock AgentCore Memory gained an IngestData API that accepts conversational or arbitrary JSON payloads and feeds them straight into an agent's configured long-term memory strategies, skipping the earlier requirement to first log everything as a short-term memory event. Extracted records surface through the existing retrieval calls, with optional Kinesis notifications and a job-listing endpoint for reprocessing failed extractions. It decouples long-term memory from the live conversation log, so teams can backfill an agent's memory from batch sources like activity logs rather than only from chat turns.

### [Weaviate's disk-based vector index reaches general availability](https://weaviate.io/blog/hfresh)

**Weaviate** · 9 Sep 2026 · *Vector search*

HFresh, Weaviate's disk-based vector index, moved from technical preview to general availability in version 1.38, trading some query latency for a large cut in memory footprint on big collections. In the company's own benchmark, a billion 256-dimension vectors needed about 239MB of heap under HFresh versus 6.67GB for a comparable uncompressed HNSW index, while quantized postings cut storage up to 32x against 32-bit floats. It targets teams whose vector collections have outgrown what they can justify keeping fully in memory.

### [FastMCP 4 ships alongside a field guide to building MCP servers that don't bloat the context window](https://www.prefect.io/blog/is-your-mcp-server-actually-good)

**Prefect** · 9 Sep 2026 · *MCP / protocols*

Prefect engineers marked the release of FastMCP 4, built on MCP's newer stateless protocol, with a set of hard-won rules for MCP server design: start from zero tools rather than mapping every REST endpoint one-to-one, since that mapping is what produces token bloat, and route especially large APIs through a "code mode" of just two tools, search and execute. They also argue servers need CI evals run against real model calls — about $3 a run on Claude, by their estimate — and middleware-based auth before anything reaches production. The advice matters because it is aimed squarely at the gap between an MCP server that technically works and one an agent can use efficiently.

### [LangChain gives multi-agent subagents two distinct ways to inherit context](https://www.langchain.com/blog/organizing-context-in-a-multi-agent-harness)

**LangChain** · 8 Sep 2026 · *Agent orchestration*

LangChain's deepagents framework now exposes two context modes for subagents: "isolated," which starts a subagent with only its task description in a fresh window, and "fork," which hands it the supervisor's full conversation history as a continuation. The post maps each mode to a role — isolated for independent verifiers and parallel researchers, fork for workers and memory-extraction agents that need the prior investigation — and notes forked subagents also benefit from prompt caching. It's a concrete answer to a recurring multi-agent design question: how much of a supervisor's context a delegated subagent should actually see.

### [LlamaIndex's two-pass pattern skips expensive OCR on most pages](https://www.llamaindex.ai/blog/just-in-time-agentic-ocr)

**LlamaIndex** · 11 Sep 2026 · *Retrieval pipelines*

For agents working across ad hoc data rooms of tens to hundreds of documents, LlamaIndex described running its free, layout-aware LiteParse tool first and reserving costlier VLM-based OCR only for pages LiteParse flags as complex. Across a test set of 84 SEC filings totaling over 12,000 pages, the first pass finished in 32 seconds and flagged roughly a fifth of pages for the expensive second pass. The company still recommends running full VLM OCR up front for large offline batch pipelines; this "retrieve first, then zoom in" pattern is aimed instead at smaller, ad hoc document sets.

### [LangChain adds per-caller identity to its managed agent credentials](https://www.langchain.com/blog/connections-managed-credentials-and-per-caller-identity-for-managed-deep-agents)

**LangChain** · 9 Sep 2026 · *Agent orchestration*

LangChain's Managed Deep Agents gained Connections, a credential layer that replaces a single shared service-account key with per-caller identity, organized along two axes: agent-owned versus user-owned credentials, and static secrets versus OAuth grants. A deployed agent calls a single `connections.get()` to either resolve a cached per-user OAuth token or trigger a fresh authorization flow, without handling client registration itself. It closes a specific gap in agent deployments, where shared credentials show what an agent can do but not who asked it to do it.
