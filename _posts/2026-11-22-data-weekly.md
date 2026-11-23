---
title: "Data and Orchestration Weekly — 22 November 2026"
date: 2026-11-22
categories: [data]
summary: "Neo4j Virtual Graph reaches GA for zero-copy graph queries over warehouses, and LangSmith opens a governed MCP tool gateway in public beta."
item_count: 6
tags: [knowledge-graphs, mcp, agent-governance, vector-search, duckdb, retrieval]
---

### [Neo4j Virtual Graph reaches general availability, querying warehouse tables as a graph without copying them](https://neo4j.com/blog/news/neo4j-virtual-graph-is-now-generally-available/)

**Neo4j** · 16 Nov 2026 · *Graph stores*

Six months after its private preview, Virtual Graph is now generally available in Aura on Google Cloud, with AWS and Azure promised later. It turns Cypher into SQL that runs inside Snowflake, Databricks, BigQuery or Iceberg tables on S3, so data stays put and the warehouse's existing roles, row filters and masking still apply; the translation is deterministic rather than LLM-generated, bounded multi-hop path patterns are supported, and unsupported queries fail with an error instead of returning wrong results. GA adds a management API, pause and resume, Private Link and grounding for Neo4j's own agents. It is read-only, answers in seconds to minutes, lacks optional matches, shortest paths, unbounded hops and vector or full-text search, cannot yet be reached through Aura's MCP servers, and the self-managed edition stays in preview until early 2027.

### [LangSmith Tool Gateway puts identity, policy and audit in front of every agent tool call](https://www.langchain.com/blog/langsmith-tool-gateway)

**LangChain** · 19 Nov 2026 · *Agent governance*

Now in public beta, the gateway lets teams register any MCP server by URL or switch on hosted ones for Google, Microsoft, Salesforce, GitHub and Linear, then bundle approved tools into kits that agents reach through a single MCP URL, with a search mode so large kits are not loaded wholesale. Each call runs either as the end user, signed in once through Okta or Entra ID, or as the agent with shared credentials, and admins can deny tools per user or require a human to approve sensitive calls. Calls appear in the agent's trace with tool, server, user, outcome and duration but not payloads. Deny policies and the organization-wide audit log require Enterprise or a sales conversation, and the post offers no performance figures.

### [Pinecone converts a vector index to a full-text-capable document index in place](https://www.pinecone.io/blog/index-conversion/)

**Pinecone** · 17 Nov 2026 · *Vector stores*

Adding keyword search to an existing Pinecone vector index used to mean re-importing every record; a single conversion job now builds a new document index while the old one keeps serving traffic. Users pick which metadata fields become full-text fields, and vectors, IDs, namespaces and remaining metadata copy over without re-embedding; Pinecone's internal test moved 100 million records in under 20 minutes. Writes made during the job must be replayed or paused, application code has to move to the Documents API, and the feature is limited to serverless indexes on paid plans, billed at the bulk-import rate of $0.25 per GB.

### [DuckDB 2.0.1 adds a GPU engine and lets publishers host signed extension repositories on their own domains](https://duckdb.org/2026/11/20/announcing-duckdb-201.html)

**DuckDB** · 20 Nov 2026 · *Analytical engines*

The first patch to the 2.0 long-term-support line ships Sirius as a core extension that runs queries on NVIDIA GPUs under Linux, downloading a roughly 1 GB backend and falling back to the CPU for unsupported features. Extension repositories can now be registered by host name alone: DuckDB reads a well-known file on that domain, follows it to the repository and pins its signing keys, so organizations can distribute their own extensions without a central registry. Several fixes change behaviour, including stricter matching of scoped secrets and a return to per-match semantics for UPDATE FROM and MERGE, and the release works around a Linux kernel bug that could silently zero reused memory.

### [Qdrant measures a code-tuned embedding model and finds 1-bit storage costs it almost nothing](https://qdrant.tech/blog/papr-embed-code/)

**Qdrant** · 18 Nov 2026 · *Retrieval*

Testing Papr's open-weight papr-embed-code-1 against the Qwen3-Embedding models it was tuned from, Qdrant found gains of about 0.03 nDCG@10 on code the model had not trained on and small losses on prose. Stored with 1-bit TurboQuant, the 4B model's vectors shrink 31x and keep their full score with 2x rescoring, beating the 0.6B model in float32 while using a twelfth of the RAM; on a filtered collection of about a million snippets, rescored 1-bit search kept 99.5% of exact quality at 1.0 ms against 1.8 ms for float32. Fusing BM25 with a code tokenizer hurt results at equal weight and only broke even when heavily down-weighted. The figures are Qdrant's own, and Papr trained on part of the benchmark suite.

### [LangSmith reorganizes around a registered agent record that also covers agents built elsewhere](https://www.langchain.com/blog/we-rebuilt-langsmith-around-the-agent)

**LangChain** · 16 Nov 2026 · *Agent platforms*

LangSmith now groups traces, evals, deployments and environments under one agent record carrying a named owner, an immutable ID and a type that determines its dashboard; the coding type can link a GitHub repository to track merged pull requests and cost per change. Agents running on any platform that emits OpenTelemetry can be registered and measured the same way, giving platform teams one inventory across in-house and vendor agents. Only general-purpose and coding types exist so far, and organization-level dashboards and registration in identity providers remain on the roadmap.
