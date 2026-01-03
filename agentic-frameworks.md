# Python Agentic AI Frameworks

A comprehensive overview of Python frameworks for implementing complex agentic flow patterns, excluding LangChain/LangGraph and Google ADK that are covered separately

## 1. **Microsoft AutoGen / Semantic Kernel / Agent Framework**

Microsoft has recently consolidated its offerings. Microsoft Agent Framework is the successor to Semantic Kernel for building AI agents. The goal is to provide a unified, enterprise-grade platform for developing, deploying, and managing AI agents.

Released in public preview on October 1, 2025, Microsoft Agent Framework merges AutoGen's dynamic multi-agent orchestration with Semantic Kernel's production foundations. The framework supports both Python and .NET.

**Key characteristics:**
- AutoGen allows multiple agents to communicate by passing messages in a loop. Each agent can respond, reflect, or call tools based on its internal logic. It has asynchronous agent collaboration, making it particularly useful for research and prototyping scenarios.
- Semantic Kernel offers enterprise-grade language flexibility through comprehensive support for Python, C#, and Java. It provides robust security protocols for legacy system integration and sophisticated workflow orchestration capabilities.

```python
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function

class MenuPlugin:
    @kernel_function(description="Provides a list of specials.")
    def get_specials(self) -> str:
        return "Special Soup: Clam Chowder..."
```

---

## 2. **CrewAI**

CrewAI is a fast, lightweight Python framework built from scratch, independent of other agent frameworks like LangChain. It enables developers to create autonomous AI agents with high-level simplicity (Crews) and precise, event-driven control (Flows) for tailored, collaborative intelligence and task orchestration.

**Key characteristics:**
- CrewAI offers a high-level abstraction that simplifies building agent systems by handling most of the low-level logic for you.
- Role-based multi-agent design with Crews and Flows abstractions
- Extensibility: It integrates with over 700 applications, including Notion, Zoom, Stripe, Mailchimp, Airtable, and more.

---

## 3. **Hugging Face Smolagents**

Smolagents is an open-source Python library designed to make it extremely easy to build and run agents using just a few lines of code. Key features include simplicity—the logic for agents fits in roughly a thousand lines of code, with abstractions kept to their minimal shape above raw code.

**Distinctive approach — Code Agents:**
Writing actions as code snippets is demonstrated to work better than the current industry practice of letting the LLM output a dictionary of tools it wants to call: uses 30% fewer steps and reaches higher performance on difficult benchmarks.

```python
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel()
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
)
result = agent.run("What is the current weather in Paris?")
```

For security, they provide options at runtime including a secure Python interpreter and sandboxed environments using E2B or Docker.

---

## 4. **Agno (formerly Phidata)**

Agno, previously known as Phidata, is an open-source framework for building agentic systems that allows developers to build, ship, and monitor AI agents with memory, knowledge, tools, and reasoning capabilities.

**Performance claims:**
Agno is designed with simplicity, performance, and model agnosticism in mind. Agent instantiation is measured at less than 5μs, significantly outperforming competing frameworks. Memory usage is optimized to 50x lower than LangGraph.

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.mcp import MCPTools

agent = Agent(
    name="Agno Agent",
    model=Claude(id="claude-sonnet-4-5"),
    tools=[MCPTools(transport="streamable-http", url="...")],
    add_history_to_context=True,
)
```

Agno is a multi-agent framework, runtime, and control plane. Build agents, teams, and workflows with memory, knowledge, guardrails and 100+ integrations. Run in production with a stateless FastAPI runtime that's horizontally scalable.

---

## 5. **LlamaIndex Workflows**

LlamaIndex Workflows 1.0 is a lightweight framework for building complex, multi-step agentic AI applications in Python and TypeScript. It allows developers to define complex AI application logic for agents without losing control of the execution flow.

**Architecture:**
A workflow is an event-driven, step-based way to control the execution flow. Your application is divided into Steps which are triggered by Events, and themselves emit Events which trigger further steps. By combining steps and events, you can create arbitrarily complex flows.

```python
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow

workflow = AgentWorkflow.from_tools_or_functions(
    [search_web],
    llm=llm,
    system_prompt="You are a helpful assistant..."
)
response = await workflow.run(user_msg="What is the latest news about AI?")
```

Workflows leverages the speed of async workflows for blazingly fast runs and easy integration with Python apps like FastAPI. They're customizable workflows that can start, pause, and resume statefully and seamlessly.

---

## 6. **Pydantic AI**

Pydantic AI is a Python agent framework designed to help you quickly build production-grade applications with Generative AI. Built by the Pydantic Team—Pydantic Validation is the validation layer of the OpenAI SDK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, CrewAI, and many more.

**Key features:**
It's fully type-safe, designed to give your IDE as much context as possible for auto-completion and type checking. It includes powerful evals for systematically testing agent performance, MCP and A2A integration, human-in-the-loop tool approval, durable execution for long-running workflows, and graph support for complex applications.

```python
from pydantic_ai import Agent

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-0',
    instructions='Be concise, reply with one sentence.',
)
```

---

## 7. **OpenAI Agents SDK (Swarm successor)**

The OpenAI Agents SDK is a lightweight yet powerful framework for building multi-agent workflows. Provider-agnostic, it supports the OpenAI Responses and Chat Completions APIs, along with 100+ other LLMs. Core features include Agents with tools, instructions, and guardrails; Handoffs for specialized control transfers between agents; Guardrails for safety checks; and Tracing for debugging and optimizing workflows.

---

## 8. **Haystack**

Haystack is an open-source Python framework for building customizable, production-ready AI applications. With its modular architecture, it supports retrieval-augmented generation (RAG), agent workflows, and advanced search systems. Haystack integrates seamlessly with tools like OpenAI, Hugging Face, and Elasticsearch.

---

## Framework Comparison for Your Testing Framework

| Framework | Best For | Complexity | Multi-Agent | Async |
|-----------|----------|------------|-------------|-------|
| **AutoGen/MS Agent Framework** | Research, enterprise orchestration | Medium-High | ✅ Native | ✅ |
| **CrewAI** | Role-based teams, rapid prototyping | Low | ✅ Native | ✅ |
| **Smolagents** | Minimal code, code-as-action pattern | Very Low | Limited | ✅ |
| **Agno** | Performance-critical, production | Low-Medium | ✅ Native | ✅ |
| **LlamaIndex Workflows** | Event-driven, RAG-heavy apps | Medium | ✅ | ✅ |
| **Pydantic AI** | Type-safety, validation-focused | Low-Medium | Via A2A | ✅ |

Given your systematic testing framework goals, **CrewAI** and **Agno** might be particularly interesting for comparing against LangChain patterns—they offer similar abstractions with different architectural philosophies. **Smolagents** would be excellent for testing the "code agents" paradigm versus traditional tool-calling approaches.