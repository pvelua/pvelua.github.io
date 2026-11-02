---
title: "Data and Orchestration Weekly — 1 November 2026"
date: 2026-11-01
categories: [data]
summary: "Temporal takes its integrations for seven agent frameworks to general availability, and Pinecone open-sources a Postgres vector-search extension."
item_count: 6
tags: [temporal, agent-memory, postgres, vector-search, orchestration, document-parsing]
---

### [Temporal's durable-execution integrations for seven agent frameworks reach general availability](https://temporal.io/blog/ai-framework-integrations)

**Temporal** · 30 Oct 2026 · *Orchestration*

Agents written with Google ADK, the OpenAI Agents SDK, AWS Strands, LangGraph, LangSmith, Vercel's AI SDK or the Gemini SDK can now run inside Temporal Workflows as supported packages. Model calls become Activities, so after a crash or deploy the agent resumes from recorded progress instead of repeating finished model or tool calls, and it can wait on human approval without polling. Coverage is uneven by language (LangGraph and the Gemini SDK are Python only), token streaming is still in public preview, and upgrading running workflows may need Worker Versioning.

### [LangChain builds agent memory as a claim-linked wiki plus an evidence graph](https://www.langchain.com/blog/building-agent-memory-openwiki-graphs)

**LangChain** · 28 Oct 2026 · *Agent memory*

Source documents are compiled into linked wiki pages whose claims point to the exact supporting lines, so a changed source flags only the affected claims for model review. A typed-edge graph over paragraphs and claims lets an agent locate a starting node and follow links to evidence. On a synthetic insurance corpus, accuracy on connected questions rose from 0.10 with plain search to 0.52 with the wiki, while simple lookups barely changed. The benchmark is synthetic, costs exclude compilation, and the open-source OpenWiki CLI is the only released artefact.

### [Pinecone open-sources pg_pinecone, a Postgres extension that indexes existing pgvector columns](https://www.pinecone.io/blog/introducing-pg-pinecone/)

**Pinecone** · 27 Oct 2026 · *Vector stores*

The Apache 2.0 extension keeps the same pgvector SQL but replaces the HNSW graph with a clustered index over compressed codes, followed by reranking against full vectors. In Pinecone's own test on one million 768-dimension vectors the index took 136 MB against 2.0 GB for HNSW on half-precision vectors, with lower latency at matched recall. Cluster centres are fixed after build, so large data changes need a reindex, and the embedding, BM25 and reranking features depend on Pinecone's hosted service.

### [LangChain's managed agents gain browser control and background subagents](https://www.langchain.com/blog/managed-deep-agents-browser-use-and-asynchronous-subagents)

**LangChain** · 29 Oct 2026 · *Agent frameworks*

Version 0.10 of Managed Deep Agents, in public beta, lets an agent drive a sandboxed browser for sites with no API, with a handoff step so a person can sign in without the password reaching the model or traces. Subagents can now be marked asynchronous so the orchestrator keeps talking while they work, with completion handled differently depending on whether the agent is busy, idle or waiting on approval. The release removes the old MCP connector layout, so existing projects need to migrate.

### [LlamaIndex proposes a Document Parsing Protocol to define what a correct parse looks like](https://www.llamaindex.ai/blog/introducing-the-document-parsing-protocol)

**LlamaIndex** · 28 Oct 2026 · *Retrieval*

Working draft 0.1 sets out 140 written rules, drawn from the annotation guidelines behind the company's ParseBench benchmark, for ambiguous choices such as spanning tables, multi-row headers, forms and charts. The aim is a shared reference so benchmarks, labellers and parser vendors judge output by the same standard. It is a draft from a single vendor, open for feedback on GitHub, and no other parser maker is named as adopting it.

### [Temporal lets agents and apps call durable Nexus services directly from its SDKs](https://temporal.io/blog/standalone-nexus-operations)

**Temporal** · 26 Oct 2026 · *Orchestration*

Standalone Nexus Operations let code outside a workflow start a Nexus operation, keep its ID and fetch the result later, with Temporal tracking the operation if the caller restarts or disconnects. SDK guides cover Go, Java, .NET, Python and TypeScript. The post describes it as an early look, with public preview starting 1 November, and states no quotas or limits.
