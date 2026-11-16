---
title: "Data and Orchestration Weekly — 15 November 2026"
date: 2026-11-15
categories: [data]
summary: "MCP maintainers draft a file-transfer extension, FastMCP 4.2 serves skills from servers, and Neo4j launches Relate, an embedded graph database for agents."
item_count: 6
tags: [mcp, protocols, agent-skills, graph-database, evaluation, serving]
---

### [MCP maintainers draft an extension for moving files without routing them through the model](https://blog.modelcontextprotocol.io/posts/exploring-file-uploads-and-downloads-in-mcp/)

**Model Context Protocol blog** · 10 Nov 2026 · *Protocols*

The File Uploads Working Group has published an experimental draft that lets a client send files to, or fetch them from, a server without the bytes passing through the model. A server marks a tool argument as a file, the client first sends only metadata so the server can check permissions and quota, and the server then returns an upload target plus a tamper-protected state token; the client uploads straight to storage and retries the call. Downloads come back as target details in result metadata, and every target states whether the client's MCP credentials may be attached. The extension is optional and has no reference implementation, leaves open where download details belong and whether an origin allowlist should be mandatory, and is aimed at stability before the end of the year.

### [FastMCP 4.2 serves a team's agent skills from its MCP server](https://www.prefect.io/blog/fastmcp-4-2-skills-over-mcp)

**Prefect** · 13 Nov 2026 · *Agent tooling*

Following FastMCP 4's move to general availability, version 4.2 lets one server publish skill folders alongside its tools, so a fix to a skill reaches every agent on its next read instead of waiting for each installed copy to be refreshed. It validates each skill against the Agent Skills spec at startup, applies existing auth rules to what each caller can discover, and offers two routes for clients that cannot yet read skills over MCP: exposing them through resource-reading tools, or copying them into a local skills folder after checking sizes and hashes. Support for the underlying Skills extension is labelled experimental, VS Code reads skills over MCP only in preview behind a setting, and 4.2 drops Python 3.10.

### [Neo4j open-sources Relate, an embedded graph database for agents](https://neo4j.com/blog/news/meet-relate-by-neo4j-an-embedded-database-for-ai-agents/)

**Neo4j** · 11 Nov 2026 · *Graph stores*

Relate is a new Rust implementation that runs inside the host process, with libraries for Python, JavaScript, Rust, Java, Go and Swift, as a command-line tool, or as a WebAssembly build, so there is no server to deploy. It speaks Cypher and GQL, and a single query can combine graph traversal with vector and full-text search; data can stay in memory, live in one file, or be backed by Parquet, JSON or Markdown. An MCP endpoint, an installable agent skill and an agent mode that returns compact output with capped rows ship with it. The startup and query-latency figures are Neo4j's own, no independent benchmark is cited, and the post does not name a licence.

### [LlamaIndex releases ParseBench-2, a document-parsing benchmark that scores errors by severity](https://www.llamaindex.ai/blog/introducing-parsebench-v2)

**LlamaIndex** · 12 Nov 2026 · *Evaluation*

The open benchmark grows from 2.1k to 4.3k cases and adds forms as a sixth dimension, and its metrics are calibrated against more than 330 error types labelled critical, major, minor or harmless, so a swapped digit now costs far more than returning a table as Markdown instead of HTML. In the authors' own example a wrong number drops from a 0.13-point penalty to 10, and users can set their own severity policy. Of 41 ranked configurations, LlamaIndex's LlamaParse Agentic Plus leads at 89.8 against 83.9 for the best rival parser, but the post quotes different scores for the same modes elsewhere and the authors sell the product being ranked. It matters as a yardstick for parsing pipelines rather than for models, with the dataset, code and paper all public.

### [vLLM extends encoder disaggregation for multimodal serving](https://vllm.ai/blog/2026-11-11-encoder-disaggregation-epd)

**vLLM blog** · 11 Nov 2026 · *Serving*

Multimodal requests can now run media encoding on separate instances from prefill and decode, with handoffs that pass layout metadata instead of raw media and move embeddings over NIXL or Mooncake. Routing support has landed in vLLM Router, SMG and llm-d, and a shared Mooncake Store lets encoders reuse one another's outputs. In a DeepSeek V4.1 Flash document-QA test on GB200, adding four encoder GPUs gave 6.6 to 9.5 percent more throughput per GPU at concurrency 64 to 128, counting all twelve GPUs, and cut P90 time to first token by 44 to 47 percent; at concurrency 16 it fell behind. DeepSeek support is on the main branch only, the cache-reuse result is a single best-case run, and a small accuracy dip in an audio test is left unexplained.

### [MCP moves its specification to two fixed releases a year](https://blog.modelcontextprotocol.io/posts/mcp-release-cycle/)

**Model Context Protocol blog** · 11 Nov 2026 · *Protocols*

Specification releases will land on the last Tuesday of March and September, with the next, version 2027-03-30, targeted for 30 March 2027. Planning starts eight months ahead, a soft freeze falls about two and a half months before release, Tier 1 SDK betas follow, and a hard freeze in the final month admits only critical fixes; a proposal that misses a freeze slips to the next release rather than delaying it. A release manager role is formalised, with Luca Chang joining the two lead maintainers. Nothing changes for existing clients or servers, so the practical effect is a predictable window for SDK work.
