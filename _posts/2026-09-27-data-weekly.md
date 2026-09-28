---
title: "Data and Orchestration Weekly — 27 September 2026"
date: 2026-09-27
categories: [data]
summary: "LangChain's Managed Deep Agents v0.8 adds per-user memory and credentials, headlining a week of framework, evaluation, and orchestration updates."
item_count: 5
tags: [agent-frameworks, data-engineering, agent-evals, durable-execution, observability]
---

### [LangChain gives Managed Deep Agents per-user memory and credential scoping](https://www.langchain.com/blog/langsmith-managed-deep-agents-whats-new)

**LangChain** · 24 Sep 2026 · *Agent frameworks*

Managed Deep Agents 0.8 splits memory into two mounted paths, one shared across an agent's users and one scoped per identity, with default access policies that block group channels and HTTP callers from reading user-level memory. Credential handling now distinguishes agent-owned keys shared across everyone from user-owned OAuth grants unique to each caller, resolved through a single connections call instead of custom auth code, and LangSmith ships 23 ready-made integrations including GitHub and Google Workspace. New HTTP channels let deployed agents receive webhook traffic from internal tools and customer portals, and Slack channels gained file transfer for logs and contracts. The release targets a problem specific to agents serving many people through one deployment: keeping each person's context and credentials separate without a team rebuilding that plumbing itself.

### [DuckDB becomes a built-in adapter in dbt's new Rust engine](https://duckdb.org/2026/09/22/dbt-fusion)

**DuckDB** · 22 Sep 2026 · *Data engineering*

dbt v2, the Rust-based rewrite of the transformation tool that reached general availability earlier in September, now ships DuckDB as a first-party adapter rather than requiring the community-maintained dbt-duckdb package, downloading and caching the driver automatically on first run. The pairing brings catalog-aware materializations backed by DuckLake and Iceberg REST catalogs, features the older Python adapter never had, plus a version of DuckDB pinned to the release so those capabilities stay stable across projects. Because dbt v2 already stores its own metadata as Parquet instead of large JSON manifests, teams can now build, test and publish transformation models entirely on a laptop without provisioning a warehouse. It is a small integration with an outsized reach, given how many data teams already default to dbt for SQL transformations.

### [LangSmith's Engine now proposes and tests its own fixes for failing agents](https://www.langchain.com/blog/langsmith-engine-v2-redteam)

**LangChain** · 24 Sep 2026 · *Agent evaluation*

LangSmith Engine v2 adds a private-beta red-teaming mode that probes deployed agents for hallucinations and system-prompt violations before they reach production, on top of its existing detection of inefficient tool-call trajectories and drifting error-rate, latency and cost trends. For Deployment customers, Engine now reproduces a reported failure, proposes a fix, tests it against the failing inputs, and iterates until it passes, surfacing the result as a one-click pull request rather than just a diagnosis. LangChain says Engine has analyzed over 70 million traces since its May launch, and reports it catches twice as many issues on its own IssueBench suite and produces fixes rated 25% more effective on Terminal-Bench than the prior version. It moves LangSmith from flagging agent problems toward closing the loop on them automatically.

### [Temporal's serverless workers can now run inside Amazon Bedrock AgentCore](https://temporal.io/blog/amazon-bedrock-agentcore-with-temporal-serverless-workers)

**Temporal** · 21 Sep 2026 · *Agent infrastructure*

Temporal released a prerelease integration letting its Serverless Workers use Amazon Bedrock AgentCore Runtime as a compute provider, so a Temporal Workflow can act as the durable control loop for an agent while AgentCore supplies the managed, scale-to-zero compute underneath. Model calls and tool operations run as Temporal activities with their own retry policies, and a published Python sample shows a TemporalAgent built on AWS's Strands framework converting those operations into durable steps. Because workflow state lives in Temporal's event history rather than on any single worker, capacity can scale with activity bursts and drop during idle periods without losing an agent's progress mid-task. It answers a specific gap: durable-execution frameworks and serverless agent runtimes have mostly been evaluated separately, not run together.

### [LangSmith adds a chronological view of what an agent actually did](https://www.langchain.com/blog/langsmith-trajectories-tracing)

**LangChain** · 24 Sep 2026 · *Agent observability*

LangSmith's new Trajectories view collapses a session's nested trace structure into a single chronological feed of messages and tool calls spanning a main agent and any subagents, working out of the box with LangChain, LangGraph, Deep Agents, OpenAI and Claude SDKs, and coding agents like Codex and Cursor. Teams can score a trajectory with an online evaluator, route it to a human annotation queue, or export it as a dataset for fine-tuning, directly from the same view. The framing is explicitly about debugging complexity that raw traces obscure: as the team put it, "a single session can span many user turns, tool calls, retries, and subagent handoffs." It is a readability layer over trace data LangSmith already collected, aimed at making long agentic sessions inspectable without wading through nested runs.
