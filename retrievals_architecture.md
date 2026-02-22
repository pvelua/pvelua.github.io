[← Back to Home](/)

Table of Contents:
- [Retrival Architecture - Query & Document Representations Interaction](#retrival-architecture---query--document-representations-interaction)
- [PyLate — Python library implementing late interaction retrieval models](#pylate--python-library-implementing-late-interaction-retrieval-models)




# Retrival Architecture - Query & Document Representations Interaction

## The Interaction Spectrum

There are three main retrieval architecture paradigms for computing query-document representations similarity:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INTERACTION PARADIGMS                                │
├─────────────────┬─────────────────────┬─────────────────────────────────────┤
│  EARLY (Cross)  │    LATE (ColBERT)   │         NO INTERACTION (Bi-Encoder) │
│  Interaction    │    Interaction      │                                     │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│                 │                     │                                     │
│  Query + Doc    │  Query    Doc       │    Query         Doc                │
│      ↓          │    ↓       ↓        │      ↓            ↓                 │
│  [BERT.......]  │  [BERT]  [BERT]     │   [BERT]       [BERT]               │
│      ↓          │    ↓       ↓        │      ↓            ↓                 │
│  Single Score   │  Token   Token      │   Single       Single               │
│                 │  Embeds  Embeds     │   Vector       Vector               │
│                 │     ↓       ↓       │      ↓            ↓                 │
│                 │    [MaxSim Pool]    │   [Cosine Similarity]               │
│                 │         ↓           │          ↓                          │
│                 │      Score          │       Score                         │
│                 │                     │                                     │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ Accuracy: Best  │ Accuracy: Very Good │ Accuracy: Good                      │
│ Speed: Slowest  │ Speed: Fast         │ Speed: Fastest                      │
│ Index: None     │ Index: Token-level  │ Index: Single vector                │
└─────────────────┴─────────────────────┴─────────────────────────────────────┘
```

## No Interaction (Bi-Encoder) — Contriever, DPR, VoyageAI,...

**How it works:**

```python
# Encoding (happens separately, can be pre-computed)
query_embedding = encoder(query)      # → single vector [768]
doc_embedding = encoder(document)     # → single vector [768]

# Similarity (simple dot product or cosine)
score = cosine_similarity(query_embedding, doc_embedding)
```

**Characteristics:**
- Query and document **never see each other** during encoding
- Document embeddings can be **pre-computed and indexed**
- Similarity is a single vector comparison
- Fast but loses fine-grained matching

**Training objective:**
```python
# Contrastive loss: push positive pairs together, negative pairs apart
loss = -log(exp(sim(q, d_pos)) / (exp(sim(q, d_pos)) + Σ exp(sim(q, d_neg))))
```

## Early Interaction (Cross-Encoder) — BERT reranker,...

**How it works:**

```python
# Query and document concatenated, processed together
combined_input = "[CLS] " + query + " [SEP] " + document + " [SEP]"
 
# Single forward pass through transformer
hidden_states = bert(combined_input)

# Classification head on [CLS] token
score = linear_layer(hidden_states[0])  # → single relevance score
```
**Characteristics:**
- Query and document **interact at every transformer layer**
- Full cross-attention between all query and document tokens
- Most accurate but **cannot pre-compute document embeddings**
- Must run inference for every (query, document) pair
- Too slow for first-stage retrieval (used for reranking)

**Training objective:**
```python
# Binary classification or pairwise ranking loss
loss = BCE(score, label)  # Is this document relevant? 0/1
# or
loss = max(0, margin - score_pos + score_neg)  # Pairwise ranking
```

## Late Interaction (ColBERT) — The Middle Ground

**The key insight:** Delay the interaction until after encoding, but keep **token-level** representations instead of pooling to a single vector.

**How it works:**

```python
# Step 1: Encode query and document SEPARATELY (like bi-encoder)
# But keep ALL token embeddings, not just [CLS]

query_tokens = tokenize(query)           # ["what", "is", "stanford"]
query_embeddings = bert(query_tokens)    # → [3, 128] (3 tokens × 128 dims)

doc_tokens = tokenize(document)          # ["stanford", "university", "is", "located", "in", "california"]
doc_embeddings = bert(doc_tokens)        # → [6, 128] (6 tokens × 128 dims)

# Step 2: Compute similarity via MaxSim (the "late interaction")
def maxsim(query_embeddings, doc_embeddings):
    """
    For each query token, find its maximum similarity to any document token.
    Then sum these maximum similarities.
    """
    scores = []
    for q_emb in query_embeddings:  # For each query token
        # Find max similarity to any doc token
        token_sims = [cosine_sim(q_emb, d_emb) for d_emb in doc_embeddings]
        scores.append(max(token_sims))
    
    return sum(scores)

score = maxsim(query_embeddings, doc_embeddings)
```

**Visual example:**

```
Query:  "stanford research"
         ├── "stanford" embedding: [0.8, 0.2, ...]
         └── "research" embedding: [0.1, 0.9, ...]

Document: "stanford university conducts cutting-edge research in AI"
           ├── "stanford"     embedding: [0.79, 0.21, ...]  ← closest to query "stanford"
           ├── "university"   embedding: [0.5, 0.3, ...]
           ├── "conducts"     embedding: [0.2, 0.4, ...]
           ├── "cutting-edge" embedding: [0.15, 0.6, ...]
           ├── "research"     embedding: [0.12, 0.88, ...] ← closest to query "research"
           └── "AI"           embedding: [0.3, 0.7, ...]

MaxSim calculation:
  query "stanford" → max sim with doc tokens → 0.99 (matches "stanford")
  query "research" → max sim with doc tokens → 0.98 (matches "research")
  
  Total score = 0.99 + 0.98 = 1.97
```

## Why "Late" Interaction?

The interaction is "late" because:

1. **Encoding is independent**: Query and document don't see each other during the transformer forward pass
2. **Interaction happens after**: The MaxSim operation is a simple computation on pre-computed embeddings
3. **But it's richer than bi-encoder**: Token-level matching preserves fine-grained semantics

```
Timeline:
                                                          
Bi-Encoder:    [Encode Q] [Encode D] → [Single dot product] → Score
                    │          │              │
                    └──────────┴──────────────┴── No interaction until final score
                    
Late Inter.:   [Encode Q] [Encode D] → [MaxSim over all token pairs] → Score
                    │          │              │
                    └──────────┴──────────────┴── Interaction at token level, but AFTER encoding
                    
Cross-Encoder: [Encode Q+D together with cross-attention] → Score
                    │
                    └── Interaction at EVERY layer during encoding
```

---

## ColBERT Training Objective

ColBERT is trained with **contrastive learning**, but the similarity function uses MaxSim:

```python
def colbert_loss(query, pos_doc, neg_docs):
    # Encode everything
    q_embs = encode_query(query)           # [num_q_tokens, 128]
    pos_embs = encode_doc(pos_doc)         # [num_pos_tokens, 128]
    neg_embs = [encode_doc(d) for d in neg_docs]
    
    # Compute MaxSim scores
    pos_score = maxsim(q_embs, pos_embs)
    neg_scores = [maxsim(q_embs, n_embs) for n_embs in neg_embs]
    
    # Contrastive loss (softmax cross-entropy style)
    # Push pos_score high, neg_scores low
    all_scores = [pos_score] + neg_scores
    loss = -log(exp(pos_score) / sum(exp(s) for s in all_scores))
    
    return loss
```

**Key training details:**

1. **Hard negatives**: ColBERT uses BM25 or dense retrieval to find hard negatives (documents that look relevant but aren't)

2. **In-batch negatives**: Other documents in the same batch serve as negatives

3. **Dimensionality reduction**: ColBERT projects to 128 dims (vs. 768 for BERT) to reduce storage

4. **Query/document markers**: Special tokens `[Q]` and `[D]` prepended to distinguish query vs document encoding mode

---

## Why Late Interaction Works Well for Entity Matching

For HippoRAG's entity matching task, late interaction has advantages:

```
Query entity: "Stanford"
KG node:      "Stanford University"

Bi-encoder (single vector):
  embed("Stanford") vs embed("Stanford University")
  → Compressed to single vectors, might lose the exact "Stanford" match

Late interaction (token-level):
  "Stanford" tokens: ["stanford"]
  "Stanford University" tokens: ["stanford", "university"]
  
  MaxSim: query "stanford" → max match with "stanford" in doc = 0.99
  
  → Exact token match is preserved!
```

**This is why ColBERTv2 performs slightly better than Contriever for HippoRAG's synonymy detection**—it can identify exact substring matches while still being fast enough for retrieval.

---

## Practical Trade-offs

| Aspect | Bi-Encoder | Late Interaction | Cross-Encoder |
|--------|------------|------------------|---------------|
| **Accuracy** | Good | Very Good | Best |
| **Indexing** | 1 vector/doc | N vectors/doc (N=tokens) | Cannot index |
| **Storage** | ~3KB/doc | ~50KB/doc | N/A |
| **Query latency** | ~1ms | ~10ms | ~100ms |
| **Use case** | First-stage retrieval | First-stage or rerank | Reranking only |

---

# PyLate — Python library implementing late interaction retrieval models

## Overview

[PyLate](https://github.com/lightonai/pylate) is an open-source Python library implementing late interaction retrieval models for information retrieval and semantic search. It provides efficient tools for re-ranking and similarity computation inspired by transformer-based approaches such as **ColBERT** (Contextualized Late Interaction over BERT) and **SPLADE**. PyLate is designed to support research and experimentation in neural retrieval and relevance modeling.

**Purpose and Functionality**

PyLate streamlines the development and evaluation of late interaction models, which compute relevance by aggregating token-level interactions rather than single dense embeddings. This approach allows more precise ranking of documents while maintaining computational efficiency. The library offers modular components for encoding, indexing, and scoring queries against large text collections.

**Architecture**

It builds on top of deep learning frameworks such as PyTorch, integrating pre-trained transformer encoders (e.g., BERT, RoBERTa). The late interaction design enables flexible experimentation with token interaction functions and pooling mechanisms, balancing retrieval accuracy and latency. PyLate typically separates document and query encoding to support offline document representation caching.


## A. What PyLate Is and What It Provides

**PyLate** is a Python library for building **late-interaction neural retrieval** systems. In contrast to bi-encoder dense retrieval (single vector per document), late-interaction models:

* Encode **queries and documents into token-level embeddings**
* Compute similarity via **MaxSim** (maximum similarity over token pairs)
* Preserve fine-grained token matching signals
* Achieve higher retrieval quality while remaining indexable

### Core Capabilities

1. **Late-Interaction Embedding**
   * Token-level embeddings for queries and documents
   * Compatible with ColBERT-style architectures

2. **Indexing**
   * Efficient indexing of document token embeddings
   * Support for scalable ANN search backends

3. **Search**
   * MaxSim scoring between query tokens and document tokens
   * Top-k retrieval

4. **Training / Fine-Tuning**
   * Supervised ranking training
   * Hard-negative mining workflows

5. **RAG Integration**
   * Plug-in retriever for Retrieval-Augmented Generation pipelines

---

### When to Use PyLate

Use PyLate when:

* You need **higher retrieval quality** than standard dense embedding search
* Keyword-based BM25 is insufficient
* You are building a **production-grade RAG system**
* You want a research-friendly late-interaction framework in Python

---

## B. Quick Start Guide

Below is a minimal workflow to index documents and run search.

### 1. Installation

```bash
pip install pylate
```

(Use a GPU-enabled environment for practical workloads.)

---

### 2. Load a Late-Interaction Model

```python
from pylate import LateInteractionModel

model = LateInteractionModel.from_pretrained("colbert-ir/colbertv2.0")
```

This loads a ColBERT-style encoder capable of producing token-level embeddings.

---

### 3. Encode Documents

```python
documents = [
    "PyLate enables late interaction retrieval.",
    "ColBERT uses token-level MaxSim scoring."
]

doc_embeddings = model.encode_documents(documents)
```

Each document is converted into a matrix of token embeddings.

---

### 4. Build an Index

```python
from pylate import LateInteractionIndex

index = LateInteractionIndex()
index.add(doc_embeddings, documents)
```

The index stores token embeddings and metadata for retrieval.

---

### 5. Encode a Query

```python
query = "What is late interaction retrieval?"
query_embedding = model.encode_query(query)
```

---

### 6. Search

```python
results = index.search(query_embedding, top_k=3)

for r in results:
    print(r.text, r.score)
```

Under the hood:

* For each query token
* Compute similarity against all document tokens
* Apply MaxSim
* Aggregate to final document score

---

## C.Conceptual Architecture

Late Interaction Retrieval pipeline:

```
Query → Token Embeddings → MaxSim
                                  → Score → Rank → Top-k
Docs  → Token Embeddings →
```

Key difference from dense retrieval:

| Approach                  | Vectors per doc | Interaction | Quality  |
| ------------------------- | --------------- | ----------- | -------- |
| Dense bi-encoder          | 1               | Dot product | Moderate |
| Late interaction (PyLate) | Many (tokens)   | MaxSim      | High     |

---

## D. Advanced Usage (High-Level)

PyLate also supports:

* Distributed indexing
* Custom scoring functions
* Mixed precision inference
* Fine-tuning on labeled relevance data
* Integration with vector databases

Example fine-tuning outline:

```python
trainer = model.get_trainer(train_dataset)
trainer.train()
```

### Practical Advice

* Use GPU for encoding.
* Keep document chunk sizes moderate (e.g., 128–256 tokens).
* Combine with rerankers for best performance.
* Ideal for knowledge-intensive domains (legal, medical, technical).

---

## E. Minimal RAG Example with **PyLate**

Below is a compact but production-realistic Retrieval-Augmented Generation (RAG) pipeline using:

* **PyLate** → late-interaction retriever (ColBERT-style)
* **ColBERT** → token-level retrieval model
* **Transformers** → generator model

This example demonstrates:

1. Indexing documents
2. Retrieving top-k passages
3. Injecting them into a prompt
4. Generating an answer

---

**Install Dependencies**

```bash
pip install pylate transformers torch
```

GPU strongly recommended.

---

**Minimal End-to-End Script**

```python
import torch
from pylate import LateInteractionModel, LateInteractionIndex
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# 1. Load Retriever (ColBERT-style)
# ---------------------------
retriever = LateInteractionModel.from_pretrained("colbert-ir/colbertv2.0")

# ---------------------------
# 2. Prepare Documents
# ---------------------------
documents = [
    "PyLate is a Python library for late interaction retrieval.",
    "ColBERT uses token-level embeddings and MaxSim scoring.",
    "Late interaction retrieval improves ranking quality compared to dense retrieval."
]

# Encode and index
doc_embeddings = retriever.encode_documents(documents)

index = LateInteractionIndex()
index.add(doc_embeddings, documents)

# ---------------------------
# 3. Load Generator (LLM)
# ---------------------------
generator_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Example
tokenizer = AutoTokenizer.from_pretrained(generator_name)
generator = AutoModelForCausalLM.from_pretrained(
    generator_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ---------------------------
# 4. Ask Question
# ---------------------------
question = "Why is late interaction retrieval better than dense retrieval?"

query_embedding = retriever.encode_query(question)
results = index.search(query_embedding, top_k=2)

# Build context from retrieved passages
context = "\n".join([r.text for r in results])

# ---------------------------
# 5. Construct RAG Prompt
# ---------------------------
prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

# ---------------------------
# 6. Generate Answer
# ---------------------------
inputs = tokenizer(prompt, return_tensors="pt").to(generator.device)

with torch.no_grad():
    output = generator.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2
    )

answer = tokenizer.decode(output[0], skip_special_tokens=True)

print(answer)
```

---

### What Happens Internally

**Retrieval Phase**

* Query → token embeddings
* Each query token compares against document tokens
* MaxSim aggregation
* Top-k passages returned

**Generation Phase**

* Retrieved passages become structured context
* LLM produces grounded response

---

### RAG Architecture Diagram

```
User Question
     ↓
Token-Level Query Embeddings
     ↓
Late Interaction (MaxSim)
     ↓
Top-k Passages
     ↓
Prompt Construction
     ↓
LLM Generation
     ↓
Grounded Answer
```

---

### Why This Is Stronger Than Standard Dense RAG

| Dense Retrieval       | Late Interaction (PyLate) |
| --------------------- | ------------------------- |
| 1 vector per document | Token-level vectors       |
| Single dot-product    | MaxSim per token          |
| Faster                | Higher precision          |
| Lower memory          | Higher memory             |

Late interaction significantly improves semantic precision — especially for technical or compositional queries.

If you are building a serious system:

1. Chunk documents (256–384 tokens)
2. Store metadata (doc IDs, source URLs)
3. Add reranker (cross-encoder)
4. Add caching layer
5. Use quantized generator (4-bit)

---
