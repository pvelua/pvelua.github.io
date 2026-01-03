# LangChain Core: Prompts and Runnables for Agent Flows

LangChain is an open source framework with a pre-built agent architecture and integrations for any model or tool.
- [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain integrations packages](https://docs.langchain.com/oss/python/integrations/providers/overview)

```bash
uv add langchain
# Requires Python 3.10+
```
LangChain provides integrations to hundreds of LLMs and thousands of other integrations. These live in independent provider packages e.g.:

```bash
# Installing the OpenAI integration
uv add langchain-openai

# Installing the Anthropic integration
uv add langchain-anthropic
```

## langchain_core Module

langchain-core contains the core interfaces and abstractions used across the LangChain ecosystem. Most users will primarily interact with the main langchain package, which builds on top of langchain-core and provides implementations for all the core interfaces.
- [Reference documentation for the langchain-core package.](https://reference.langchain.com/python/langchain_core/)

## langchain_core.prompts

This module provides template classes for constructing prompts that can be dynamically populated with variables.

### Key Classes

**ChatPromptTemplate** — The primary class for chat-based models:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Simple template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant."),
    ("human", "{user_input}")
])

# With message history placeholder (for agents with memory)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")  # for tool calls
])
```

**PromptTemplate** — For non-chat completions:

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Summarize: {text}")
# or explicit
prompt = PromptTemplate(template="...", input_variables=["text"])
```

**Partial application** — Pre-fill some variables:

```python
partial_prompt = prompt.partial(role="technical")
# Now only needs {user_input}
```

---

## langchain_core.runnables

This is the foundation of LangChain Expression Language (LCEL). Everything is a `Runnable` with a consistent interface: `.invoke()`, `.batch()`, `.stream()`, and async variants.

### Core Chaining: RunnableSequence

The pipe operator `|` creates sequential chains:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | model | StrOutputParser()

# Equivalent to:
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(first=prompt, last=model)

# Invoke
result = chain.invoke({"role": "coding", "user_input": "Explain decorators"})
```

### Parallel Execution: RunnableParallel

Run multiple branches simultaneously and merge outputs:

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

# Returns {"summary": ..., "keywords": ..., "sentiment": ...}
result = parallel.invoke({"text": document})
```

### Routing: RunnableBranch

Conditional routing based on input:

```python
from langchain_core.runnables import RunnableBranch

router = RunnableBranch(
    (lambda x: x["topic"] == "code", code_chain),
    (lambda x: x["topic"] == "math", math_chain),
    (lambda x: len(x["query"]) < 20, simple_chain),
    default_chain  # fallback (required)
)

result = router.invoke({"topic": "code", "query": "..."})
```

### Dynamic Routing with RunnableLambda

For more complex routing logic:

```python
from langchain_core.runnables import RunnableLambda

def route(input_dict):
    if input_dict["complexity"] == "high":
        return expert_chain
    return basic_chain

dynamic_router = RunnableLambda(route)

# Chain it
full_chain = preprocess | dynamic_router | postprocess
```

### Passthrough and Assign

**RunnablePassthrough** — Pass input unchanged (useful in parallel structures):

```python
from langchain_core.runnables import RunnablePassthrough

chain = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()  # passes input through unchanged
)
```

**RunnablePassthrough.assign()** — Add new keys while keeping existing ones:

```python
chain = RunnablePassthrough.assign(
    processed=lambda x: x["raw"].upper(),
    length=lambda x: len(x["raw"])
)
# Input: {"raw": "hello"} → Output: {"raw": "hello", "processed": "HELLO", "length": 5}
```

### Transformation with RunnableLambda

Wrap any function as a runnable:

```python
from langchain_core.runnables import RunnableLambda

def extract_json(text: str) -> dict:
    import json
    return json.loads(text)

chain = prompt | model | StrOutputParser() | RunnableLambda(extract_json)
```

### Configuration and Binding

**bind()** — Attach fixed parameters to a runnable:

```python
model_with_tools = model.bind(tools=[tool1, tool2])
model_with_format = model.bind(response_format={"type": "json_object"})
```

**with_config()** — Runtime configuration:

```python
chain.with_config(
    run_name="extraction_chain",
    tags=["experiment-1"],
    callbacks=[my_callback]
)
```

### Fallbacks

Handle failures gracefully:

```python
robust_chain = primary_chain.with_fallbacks([
    fallback_chain_1,
    fallback_chain_2
])
```

---

## Common Patterns for Agents Framework

**Pattern 1: Sequential Chaining (your current implementation)**
```python
chain = extract_prompt | model | parser | transform_prompt | model | final_parser
```

**Pattern 2: Router Pattern**
```python
classifier = classify_prompt | model | StrOutputParser()

def select_chain(classification):
    routes = {"A": chain_a, "B": chain_b, "C": chain_c}
    return routes.get(classification.strip(), default_chain)

full = classifier | RunnableLambda(select_chain)
```

**Pattern 3: Map-Reduce / Fan-out**
```python
chain = (
    RunnablePassthrough.assign(chunks=lambda x: split(x["doc"]))
    | RunnableParallel(results=RunnableLambda(lambda x: [process.invoke(c) for c in x["chunks"]]))
    | combine_chain
)
```

**Pattern 4: Extracting intermediate values for logging**
```python
def log_and_pass(x):
    output_writer.write(x)  # your logging
    return x

chain = step1 | RunnableLambda(log_and_pass) | step2
```
---