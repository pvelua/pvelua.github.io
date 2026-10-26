---
title: "Data and Orchestration Weekly — 25 October 2026"
date: 2026-10-25
categories: [data]
summary: "DuckDB 2.0 LTS ships with a client-server protocol, async object-storage reads and an agent-aware CLI, while Qdrant adds native BM25 full-text ranking."
item_count: 6
tags: [duckdb, vector-search, workflow-orchestration, retrieval, embeddings]
---

### [DuckDB 2.0 LTS adds a client-server protocol, async reads and agent-aware CLI output](https://duckdb.org/2026/10/21/announcing-duckdb-200.html)

**DuckDB** · 21 Oct 2026 · *Databases*

The long-term-support 2.0 release, built from close to 20,000 commits since 1.5, introduces the Quack client-server protocol, background I/O that keeps several requests in flight against object storage, a VARIANT type for semi-structured data, triggers, and a new default parser. The CLI now detects when an agent is driving it and switches to compact tables and JSON errors; in the team's TPC-H tests this cut the output a model had to read by 59%. Databases written by 2.0 cannot be opened by older versions, so the project advises trying it on a shadow system first. Support for 2.0 runs until at least October 2027.

### [Qdrant 1.20 brings BM25 ranking into full-text indexes](https://qdrant.tech/blog/qdrant-1.20.x/)

**Qdrant** · 22 Oct 2026 · *Retrieval*

A full-text index can now score with BM25 computed from the stored data, so keyword ranking no longer needs a separate sparse-vector step. The new text query works alone, in hybrid search with dense vectors, or as a rescoring pass, and the BM25 parameters can be changed per query without rebuilding. A code-aware tokenizer lets differently styled identifiers match each other, and per-element nested indexes were 7 to 38 times faster in the vendor's best-case test of 100,000 points. Sparse-vector BM25 remains supported.

### [StackAI cut file-indexing failures from about 20% to under 0.1% by moving to Temporal](https://temporal.io/blog/how-stackai-used-temporal-cut-indexing-failures-from-20-percent-to-below-1-percent)

**Temporal** · 19 Oct 2026 · *Orchestration*

In an interview with its former CTO, StackAI describes replacing a Celery-based pipeline that failed to index roughly one file in five on the first try. The rebuilt pipeline reruns only the failed stage, sends files over 50 MB through their own workflow, classifies errors as retryable or not, and chains parsers as child workflows that fall through on failure, with results cached by content hash. Parsing costs dropped to about a tenth, though the account credits caching and parser changes alongside Temporal. It is a vendor-hosted interview and gives no throughput figures.

### [Pinecone lets one API call embed and search raw text](https://www.pinecone.io/blog/docs-api-integrated-embedding/)

**Pinecone** · 22 Oct 2026 · *Vector stores*

On serverless indexes, the Documents API now accepts raw text and handles embedding at write and query time, with three hosted models at launch and semantic, learned-sparse and BM25 query modes over one text field. Pinecone measured a 26 ms median for the combined request against 92 ms when embedding first and searching second, in its own US East test. Existing Records API indexes cannot yet migrate, backfills are billed per token, and server-side fusion and reranking are still listed as planned.

### [Qdrant tests whether a 32K-token embedding model makes chunking obsolete](https://qdrant.tech/blog/ultrabert-embed/)

**Qdrant** · 19 Oct 2026 · *Retrieval*

The post evaluates a 151M-parameter encoder that embeds a whole document of up to 32K tokens as one vector. Whole-document vectors beat 512-token chunks on only one of the tasks tried, and accuracy on a needle-finding test fell from 84% at 256 tokens to 30% at 32K. The authors conclude chunking is not dead: keep chunks for detail lookup and use whole-document vectors to find topics. Tests were on the authors' own setup.

### [Qdrant shows Cohere's page-image embeddings held in 64 bytes of RAM per page](https://qdrant.tech/blog/cohere-embed-5-qdrant/)

**Qdrant** · 23 Oct 2026 · *Retrieval*

Using Cohere Embed 5 Pro vectors for PDF pages, the recommended setup truncates to 512 dimensions, stores one bit per dimension in memory, and rescores the top 20 candidates from full vectors on disk. That kept 99.2% of full-vector quality, and the authors estimate about 610 MB of RAM for 10 million pages. Results come from five public ViDoRe V3 corpora and the authors' own tests, and Cohere's training data is unpublished, so contamination is possible.
