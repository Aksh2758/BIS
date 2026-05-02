# BIS Standards Recommendation Engine

A lightweight **Retrieval-Augmented Generation (RAG)** system that recommends relevant **Bureau of Indian Standards (BIS)** documents given a natural language query.

Built for the **BIS Hackathon**, this pipeline extracts and indexes 564 IS standard summaries from the official SP 21 (2005) document and retrieves the most relevant standards using semantic similarity search.

---

## Evaluation Results (Public Test Set)

| Metric | Score | Target |
|--------|-------|--------|
| Hit Rate @3 | **80.00%** | >80% |
| MRR @5 | **0.8450** | >0.7 |
| Avg Latency | **0.02 sec** | <5 sec |

---

## Project Structure

```
.
├── src/
│   └── rag_pipeline.py      # SimpleRAG class (SentenceTransformer + FAISS)
├── data/
│   ├── dataset.pdf          # Source: SP 21 (2005) BIS building materials standards
│   ├── standards_chunks.json  # Extracted & labeled IS standard summaries (auto-generated)
│   └── bis_docs.txt         # Plain-text version of extracted standards (auto-generated)
├── inference.py             # Entry point — run with --input and --output flags
├── eval_script.py           # Official hackathon evaluation script
├── extract_dataset.py       # One-time script to parse dataset.pdf → standards_chunks.json
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract the Dataset (one-time step)

This parses `data/dataset.pdf` and generates `data/standards_chunks.json`:

```bash
python extract_dataset.py
```

> This takes ~2 minutes for the full 929-page PDF. Output: 564 labeled IS standards.

### 3. Run Inference

```bash
python inference.py --input public_test_set.json --output my_output.json
```

**Input format** (`public_test_set.json`):
```json
[
  { "id": "PUB-01", "query": "Portland cement for construction", "expected_standards": ["IS 269: 1989"] }
]
```

**Output format** (`my_output.json`):
```json
[
  {
    "id": "PUB-01",
    "query": "Portland cement for construction",
    "expected_standards": ["IS 269: 1989"],
    "retrieved_standards": ["IS 269 : 1989", "IS 8112 : 1989", "IS 12269 : 1987", "IS 455 : 1989", "IS 8042 : 1989"],
    "latency_seconds": 0.021
  }
]
```

### 4. Evaluate Results

```bash
python eval_script.py --results my_output.json
```

---

## How It Works

1. **Data Extraction** — `extract_dataset.py` parses each page of `data/dataset.pdf` looking for `SUMMARY OF IS XXXX : YYYY` headers. Each standard's title + scope + requirements text is stored as a labeled chunk.

2. **Indexing** — `SimpleRAG` in `src/rag_pipeline.py` encodes all 564 standard summaries using `sentence-transformers/all-MiniLM-L6-v2` and stores the vectors in a FAISS `IndexFlatL2` index.

3. **Retrieval** — For each query, the query is embedded and the top-5 nearest neighbors are retrieved from FAISS. The IS standard labels (e.g., `IS 269 : 1989`) are returned — not raw text.

4. **Output** — Results are written as a JSON array matching the required hackathon format.

---

## Key Design Decisions

- **Labels vs. Text separation**: Embeddings are built from full summary text (rich context), but results return only the clean IS code+year — exactly what the eval script matches against.
- **top_k=5**: Returns 5 standards per query to maximize both Hit@3 and MRR@5.
- **No overengineering**: Pure FAISS similarity search, no LLM calls, no re-ranking. Avg latency is ~0.02 seconds.
