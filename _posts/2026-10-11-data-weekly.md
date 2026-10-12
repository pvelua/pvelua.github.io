---
title: "Data and Orchestration Weekly — 11 October 2026"
date: 2026-10-11
categories: [data]
summary: "Temporal makes a coding agent survive worker crashes, Qdrant shrinks embedding RAM up to 77x, and DuckDB's shell adapts its output for AI agents."
item_count: 6
tags: [orchestration, retrieval, agents, durable-execution]
---

### [Temporal lets a coding agent fail over between machines mid-task](https://temporal.io/blog/the-immortal-life-of-pi-running-the-pi-coding-agent-on-temporal)

**Temporal** · 8 Oct 2026 · *Durable execution*

Temporal released an experimental pi-temporal layer that runs the Pi coding agent's session as a workflow, with each model call and tool call as a separate retryable activity. State is rebuilt from a shared transcript and per-step Git bundles, so another worker can pick up a turn if the first one dies. Tool calls get a claim record before they run, and a retry that finds a claim with no result is flagged as unknown rather than repeated blindly. In a demo where the active worker was killed every 15 to 40 seconds, the task still finished; it is a retrofit on a fork, and side effects of an already-run tool cannot be undone.

### [Qdrant tests how far EmbeddingGemma 2 vectors can be compressed](https://qdrant.tech/blog/embeddinggemma-2/)

**Qdrant** · 6 Oct 2026 · *Retrieval*

Using Google's new EmbeddingGemma 2, Qdrant compared 1-bit quantization against fewer dimensions on five BEIR datasets. Full 768-dimension vectors at one bit kept about 99% of retrieval quality with 30x less vector RAM, while a 256-dimension, rescored setup kept 94.5% at 77x less. The advice is to keep more dimensions and compress each one harder rather than the reverse. The RAM figures leave out graph indexes, payloads and the original vectors needed for rescoring.

### [LangChain's Deep Agents load tools only when a skill is read](https://www.langchain.com/blog/revamping-skills-in-deep-agents)

**LangChain** · 7 Oct 2026 · *Agent orchestration*

Deep Agents skills can now bind tools to themselves, so tool schemas stay out of the context window until the agent opens the relevant skill. Applications can also pin chosen skills into the first prompt, group tools behind labels that gate access by user permission, and rescan the skill library mid-thread. The aim is to keep prompts small for agents with dozens or thousands of skills. A reload that finds new skills does invalidate the provider's prompt cache.

### [Qdrant tries a generalized classifier across the search pipeline](https://qdrant.tech/blog/jevjitsu/)

**Qdrant** · 8 Oct 2026 · *Retrieval*

Qdrant used TypeSafe's Jev decision model, with no task-specific training, for reranking, de-duplication, query understanding and chunking. On 100 NFCorpus queries it lifted nDCG@10 by roughly 0.07 to 0.08, well above the two cross-encoder baselines. As a semantic chunker on QASPER it beat fixed, recursive and embedding-based splitters by 9 to 13% on matched chunk sizes, at about $0.27 per million tokens. Results on short, ambiguous queries were mixed and the authors had no relevance labels for them.

### [DuckDB's shell gets an agent mode for token-frugal output](https://duckdb.org/2026/10/09/agent-mode.html)

**DuckDB** · 9 Oct 2026 · *Data tooling*

The DuckDB v2.0 command-line shell now detects when an AI coding agent is driving it, using environment variables and a non-terminal stdout. In that case it prints compact Markdown tables, caps large results at the first and last 20 rows with a hash of the full output, and sends errors as JSON. It also warns before queries expected to scan a million rows or more and stops counting runaway results after 100,000 extra rows. Flags let users force the mode on or off.

### [LlamaIndex opens a router that puts ten document parsers behind one API](https://www.llamaindex.ai/blog/introducing-opendocrouter)

**LlamaIndex** · 7 Oct 2026 · *Ingestion*

OpenDocRouter converts PDFs and images to Markdown through five frontier models and five open-source OCR models behind one interface, with optional layout boxes shared across all of them. Billing is by token, and the post's ParseBench cost estimates range from about $0.80 per 1,000 pages for the cheapest option to nearly $49 for the most expensive. That spread is the practical case for choosing a parser per workload rather than defaulting to one.
