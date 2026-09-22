# Build & Evaluate a RAG Pipeline with Qdrant

A small Retrieval-Augmented Generation (RAG) pipeline built with LangChain, Google Gemini, and Qdrant using the `rag-datasets/rag-mini-wikipedia` benchmark.

## Pipeline

* Loaded the `text-corpus` subset containing 3,200 Wikipedia passages.
* Sampled **200 passages** using `seed=42`.
* Split documents using `RecursiveCharacterTextSplitter`:

  * `chunk_size=500`
  * `chunk_overlap=50`
* Generated **768-dimensional embeddings**.
* Stored the chunks in a Qdrant `mini_wikipedia` collection using in-memory mode.
* Retrieved the top **4** relevant chunks for each question.
* Built an LCEL RAG chain that answers only from the retrieved context.
* Evaluated on **20 questions** sampled with `seed=42`.

## Evaluation

**Accuracy: 15.00%**

The main failure pattern was retrieval mismatch: many evaluation questions could not be answered from the limited 200-passage corpus sample, so the model returned `I don't know` when the required evidence was not retrieved. The evaluation also uses the assignment's simple substring metric, which can produce false positives for short answers such as `no`.

## Model Compatibility Note

The assignment originally specifies `text-embedding-004` and `gemini-2.0-flash`. These models were unavailable during implementation, so the notebook uses currently available Gemini models while preserving the required embedding dimension of **768**:

* Embeddings: `gemini-embedding-2` with `output_dimensionality=768`
* LLM: `gemini-3.1-flash-lite`

## Setup

Install the required packages:

```bash
pip install -q langchain langchain-google-genai langchain-qdrant qdrant-client datasets pandas
```

Add `GEMINI_API_KEY` to Google Colab Secrets and run the notebook from top to bottom.

## Files

```text
langchain-qdrant-rag/
├── README.md
├── langchain_qdrant_rag.ipynb
└── eval_results.csv
```

## Cost

**$0** — Google Colab free tier, Gemini API free tier, and Qdrant in-memory mode.
