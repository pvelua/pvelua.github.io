[← Back to Home](/)

# Ecoder Training Objectives to Match Requirements

## Brain Analogy

It seems that my intuition about human brain workings and agent memory design requirements aligns perfectly with the neuroscience:

    "Person often remembers a noun like location name or person name and then does memory look up for this noun relationships."

This is the `hippocampal` indexing theory. The hippocampus stores a sparse index of discrete cues (like nouns/entities), not full representations. When you encounter a cue:

    1. Cue activation: "Stanford" activates that node in your hippocampal index
    2. Pattern completion: Activation spreads to associated nodes ("Thomas," "California," "research")
    3. Neocortical recall: The full memory is reconstructed from these activated associations

Research in the [Google and the Mind: Predicting Fluency With PageRank](https://journals.sagepub.com/doi/abs/10.1111/j.1467-9280.2007.02027.x) paper show that PageRank, computed on a semantic network constructed from word-association data, outperformed word frequency and the number of words for which a word is named as an associate as a predictor of the words that people produced in a verbal fluency task where participants named the first word beginning with a given letter that came to mind. 

I think that for agent memory design, this suggests:

    1. Store discrete, labeled memory entries (not just embedding vectors)
    2. Retrieval should activate networks of related memories, not just the single nearest neighbor
    3. The "index" (graph structure) can update without retraining the "representations" (LLM)

## Encoder Requirements 

What does `hippocampal` indexing need from embeddings? This is where it gets nuanced:

| Use Case                        | Input Length      | Requirement  |
| ------------------------------- | ----------------- | ------------- |
| Query entity → KG node matching | Short (1-5 words) | Similar phrases → similar vectors |
| Synonymy edge detection         | Short (1-5 words) | Paraphrase/alias detection |

**Key insight**: The `hippocampal` indexing embeddings are  short noun phrases, not documents or passages as you need to work in case of a document repository. And that is where encoder training objectives matter:

| Training Objective | Good For | Example Models |
| :----------------- | :------- | :------------- |
| Contrastive retrieval (query-doc pairs) | Document retrieval      | Contriever, E5, BGE |
| Paraphrase detection                    | Short phrase similarity | Sentence-BERT, all-MiniLM |
| Late interaction                        | Fine-grained matching   | ColBERTv2 |
| General-purpose                         | Broad coverage          | VoyageAI, OpenAI embeddings |

**Why ColBERTv2 Works Well for Brain Models**

ColBERTv2 is interesting because it's a late interaction model which produces token-level embeddings and computes similarity via MaxSim aggregation. Fine-grain entity matching is benefitial for retrival based on a brain model.

- Trained on MS MARCO (retrieval task) with hard negatives
- Good at distinguishing similar but different entities
- Handles short text well due to token-level representations

**VoyageAI Considerations**

VoyageAI models (voyage-2, voyage-code-2, etc.) are:

- General-purpose with broad training
- Optimized for longer text (documents, paragraphs)
- Good quality but not specifically tuned for short phrase synonymy

## Infomation About Popular Encorders

This section has infomation about fficial home pages, repositories, and documentation for the populare encoders that are mentioned on this page. Encoders are categorized into **Open Source** (run locally) and **Commercial** (API-based) to help with project architecture decision.

### Open Source & Self-Hosted

*Best for running locally on your own GPU/CPU.*

| Model / Library   | Official Home / Repository |
| :---------------- | :------------------------- |
| **Contriever**    | [Facebook Research GitHub](https://github.com/facebookresearch/contriever)  | 
| **E5**            | [Microsoft/unilm GitHub](https://github.com/microsoft/unilm/tree/master/e5) |
| **BGE**           | [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) |
| **Sentence-BERT** | [SBERT.net Documentation](https://www.sbert.net/) |
| **all-MiniLM**    | *Part of Sentence-BERT* | 
| **ColBERTv2**     | [Stanford Future Data GitHub](https://github.com/stanford-futuredata/ColBERT) |

### Commercial & API-Based

*Best for getting started quickly without managing infrastructure.*

| Provider         | Official Home and Documentation |
| :--------------- | :---------------------------------------- |
| **Voyage AI**    | [voyageai.com](https://www.voyageai.com/) [docs.voyageai.com](https://docs.voyageai.com/) |
| **OpenAI**       | [openai.com](https://openai.com/) [Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) |
| **Vertex AI**    | `text-embedding-004`: balanced model for most production workloads [Embedding API's Overview...](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings#:~:text=Applications%20can%20use%20embeddings%20to,or%20see%20music%20streaming%20recommendations.) |
|                  | `text-multilingual-embedding-002`: optimized for non-English or mixed-language datasets [Get text embeddings...](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings) |
|                  | `gemini-embedding-001`: unified model designed to work alongside Gemini 1.5/2.0 [Gemini Embedding now...](https://developers.googleblog.com/gemini-embedding-available-gemini-api/#:~:text=The%20Gemini%20Embedding%20model%20is,free%20through%20Google%20AI%20Studio.) |
| **Azure OpenA!** | `text-embedding-3-small`: highly efficient, low cost, variable dimension size [Azure OpenAI Embeddings Models](https://docs.azure.cn/en-us/search/tutorial-rag-build-solution-models) |
|                  | `text-embedding-3-large`: higher precision for complex retrieval tasks. |
|                  | `text-embedding-ada-002`: previous standard (legacy but widely used) |

---