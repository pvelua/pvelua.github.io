---
title: "Data and Orchestration Weekly — 18 October 2026"
date: 2026-10-18
categories: [data]
summary: "LangSmith adds natural-language search over agent traces, and Qdrant shows page-image embeddings compressed 21x while keeping 96.5 percent of retrieval quality."
item_count: 6
tags: [observability, retrieval, evaluation, vector-security, context-engineering]
---

### [LangSmith adds natural-language search over agent traces](https://www.langchain.com/blog/agent-traces-ai-search-langsmith)

**LangChain** · 13 Oct 2026 · *Observability*

A new public-beta feature in LangSmith lets teams describe a behavior, such as a frustrated user or a leaked piece of personal data, and returns the traces where it occurred. Rather than embedding traces, it has a fast decision model read each candidate after ordinary filters narrow the set, working from a condensed view of each session capped at 8K tokens. The post's sample of 150 Open SWE conversations explains the design: the median run was around 72K tokens and roughly 85% of it was tool output, so raw text search buries the signal. It is free on LangSmith Cloud in three regions and must be switched on by an org admin.

### [Qdrant shrinks page-image embeddings 21x while keeping 96.5% of search quality](https://qdrant.tech/blog/pplx-embed-v2/)

**Qdrant** · 13 Oct 2026 · *Retrieval*

Perplexity's new late-interaction embedding models represent each document page as an image, skipping OCR, but at about 856 KB of float32 vectors per page. Qdrant's 1-bit TurboQuant method brings that to roughly 40 KB and retained 96.5% of float32 nDCG@10 on four ViDoRe v3 sets, rising to 99.9% with rescoring; plain binary quantization kept 76.4%. The tests are Qdrant's own, cover 200 English queries per set, and the storage figure ignores the original vectors that rescoring needs. Pipelines that need extracted text or structured fields still need OCR.

### [LlamaIndex releases a harder benchmark for document extraction](https://www.llamaindex.ai/blog/introducing-extractbench-pro)

**LlamaIndex** · 15 Oct 2026 · *Evaluation*

ExtractBench Pro adds 100 long-tail documents to the original extraction benchmark, covering house vocabularies, house counting rules, drawn relations and charts. Twelve systems were run, including extraction APIs, coding agents and LlamaIndex's own LlamaExtract tiers, whose scores fell by 18.5 to 47.0 points of value F1 against the easier set. Coding agents scored highest but cost far more per page, while LlamaExtract sat on the cost frontier below 25 cents a page. The dataset, code and paper are public, though the benchmark is run by a vendor that competes in it, and the authors argue for a stricter percent-perfect metric.

### [Weaviate fixes a flaw that let restricted users infer protected vectors](https://weaviate.io/blog/weaviate-security-release-nearobject-2026)

**Weaviate** · 15 Oct 2026 · *Vector stores*

A medium-severity bug in the GraphQL nearObject operator loaded the vector of a referenced object without checking the caller's access to that object's collection. The caller never saw the object, but returned distances could be used to reconstruct the protected vector. Exploiting it required knowing the target collection name and object UUID plus permission to read collection definitions and write data elsewhere. Versions before 1.39.3 are affected; the CVE is pending and the vendor reports no known exploitation. Until upgrading, narrowing roles that hold read-collections and create-data permissions reduces exposure.

### [Pinecone's bring-your-own-cloud mode gets on-demand capacity](https://www.pinecone.io/blog/byoc-on-demand/)

**Pinecone** · 15 Oct 2026 · *Vector stores*

BYOC customers can now run indexes from a shared pool that supports millions of namespaces, such as one per tenant or per agent user, instead of sizing each index up front. Capacity is added as shards without moving existing data, and a busy namespace can be promoted to dedicated read nodes. Vectors, metadata and queries stay inside the customer's cloud account, and scaling needs no SSH, VPN or standing cross-account role. The post gives no performance numbers.

### [LangChain details how Deep Agents keep context from rotting](https://www.langchain.com/blog/context-engineering-deep-agents-remember)

**LangChain** · 14 Oct 2026 · *Context engineering*

The post lays out the defaults in the open-source Deep Agents harness for moving material out of a model's window without discarding it. Tool results over about 20,000 tokens and user messages over 50,000 are written to files and replaced by a preview and path, and summarization starts at 85% of the context limit with the originals saved for recovery. Skills, subagents and a plain-file memory handle the rest, all as middleware around the agent loop. Most of this shipped earlier; the value here is the concrete thresholds and the recoverable-compaction design.
