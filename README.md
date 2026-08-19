# RAG Chunking Strategy Evaluator

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

## What It Is
A Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python without high-level orchestration frameworks. The system implements two distinct chunking strategies, vector similarity search, cross-encoder reranking, and an LLM-as-a-judge evaluation suite. This project is Pass 1 of a two-pass engineering study designed to isolate the core mechanics of RAG; Pass 2 rebuilds the exact same pipeline using LangChain to evaluate what the framework abstracts, optimizes, or obfuscates.

## Architecture

* **Chunking Strategies:**
  * **Sentence-Window Chunking:** NLTK sentence tokenization with `window_size=5` and `overlap=2`.
  * **Fixed-Size Chunking:** Word-level sliding window with `chunk_size=200` words and `overlap=50` words.
* **Embeddings:** `models/gemini-embedding-2` generating 3072-dimensional dense vector embeddings.
* **Vector Storage & Indexing:** Persistent ChromaDB collections using HNSW index configuration (`space: cosine`, `ef_construction: 200`).
* **Semantic Reranker:** Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring and reordering top-k candidate chunks.
* **Generation & LLM-as-a-Judge:** `gemini-3.5-flash-lite` used for strictly grounded context-based answer generation and automated binary faithfulness validation (0 or 1).
* **Relevance Metric:** Cross-Encoder semantic score normalized via a sigmoid function.

## Evaluation Results

| Metric | Sentence-Window Strategy | Fixed-Size Strategy |
| :--- | :--- | :--- |
| **Total Queries Evaluated** | 15 | 15 |
| **Average Relevance Score** | **0.93** | **1.00** |
| **Faithfulness Rate** | **100%** (15/15) | **100%** (15/15) |
| **Retrieval Failure Rate** | **6.67%** (1/15) | **0.0%** (0/15) |

---

## Key Finding

Empirical testing revealed a crucial trade-off between **context granularity** and **retrieval recall**:

1. **Fixed-Size Outperformed Sentence-Window on Recall:** Fixed-Size chunking achieved a **1.00 average relevance score** compared to Sentence-Window's **0.93**. 
2. **The Retrieval Bottleneck:** Sentence-Window chunking (`window_size=5`, `overlap=2`) suffered a retrieval miss on Query 15 (*"How many matches did it take Starc to reach 200 ODI wickets?"*). Because the sentence window was small, the embedding lacked sufficient surrounding semantic density to rank in the top-2 vector matches against ChromaDB. 
3. **Strict Faithfulness:** Both strategies achieved 100% faithfulness. When Sentence-Window failed to retrieve the relevant chunk for Query 15, the model correctly abstained with `"It is not found in the given context."` rather than hallucinating, scoring a faithfulness of 1 and a relevance of 0.0.

### Retrieval Difference on Query 15

* **Query:** *"How many matches did it take Starc to reach 200 ODI wickets?"*
* **Sentence-Window Output:** 
  > `"It is not found in the given context."` *(Relevance: 0.0 | Faithfulness: 1)*
* **Fixed-Size Output:** 
  > `"It took Starc 102 matches to reach 200 ODI wickets."` *(Relevance: 1.0 | Faithfulness: 1)*

**Conclusion:** While sentence-window chunking strictly preserves grammatical boundaries, setting the window too narrow (`window_size=5`) risks isolating facts from broader semantic context, harming vector retrieval in dense informational texts. Fixed-size chunking provided broader contextual mass per chunk, ensuring 100% retrieval success across this evaluation suite.

## How to Run

1. **Clone repository and set up environment:**

    git clone https://github.com/yourusername/Advanced-RAG-System.git
    cd Advanced-RAG-System
    cp .env.example .env

2. **Install dependencies:**

    pip install -r requirements.txt

3. **Run evaluation:**

    python main.py --data text.txt --queries queries.txt --output eval_results.md

## What I Learned / What LangChain Abstracts

<!-- Placeholder: To be updated after implementing Pass 2 -->
* [ ] Document loaders vs. manual file reading
* [ ] TextSplitter abstractions vs. explicit NLTK / sliding window loops
* [ ] ParentDocumentRetriever mechanics vs. raw vector lookups
* [ ] Built-in evaluators vs. custom Cross-Encoder / LLM judge loops
* [ ] Framework overhead, execution latency, and debugging visibility