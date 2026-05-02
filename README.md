# BIS Standards Recommendation Engine

An AI-powered RAG pipeline that helps Indian MSEs find relevant BIS standards from a product description in seconds.

## Architecture

```
Product Description
      ↓
  Query Embedding (all-MiniLM-L6-v2)
      ↓
  FAISS Cosine Search (top-10 candidates)
      ↓
  Metadata Boosting (lightweight/slag/pozzolana/part flags)
      ↓
  Top-5 Standards Selected
      ↓
  Claude LLM → Rationale per standard
      ↓
  Structured JSON Output
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```
> Without this, the system still works but returns generic rationale instead of LLM-generated explanations.

### 3. (One-time) Enrich chunks with metadata
```bash
python enrich_chunks.py
```
This generates `data/standards_chunks_enriched.json` with category tags, part numbers, and type flags for better retrieval.

## Running Inference (Judge Command)

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

## Running Evaluation

```bash
python eval_script.py --results team_results.json
```

## Running the Streamlit UI

```bash
streamlit run app.py
```

## Project Structure

```
├── src/
│   └── rag_pipeline.py       # Core RAG pipeline (retrieval + LLM rationale)
├── data/
│   ├── standards_chunks.json            # Raw chunks from dataset.pdf
│   ├── standards_chunks_enriched.json   # Enriched chunks with metadata
│   └── dataset.pdf                      # Source BIS SP 21 document
├── inference.py              # Judge entry point
├── enrich_chunks.py          # One-time metadata enrichment script
├── eval_script.py            # Evaluation script (provided by organizers)
├── app.py                    # Streamlit UI
├── requirements.txt
└── README.md
```

## Chunking & Retrieval Strategy

- **Source**: BIS SP 21 PDF — one chunk per standard summary
- **Metadata enrichment**: Each chunk tagged with category (cement/concrete/aggregates/masonry/steel/pipes), part number, and type flags (lightweight, slag, pozzolana, white cement, rapid hardening)
- **Embedding**: `all-MiniLM-L6-v2` with L2 normalization → cosine similarity via FAISS `IndexFlatIP`
- **Re-ranking**: Retrieve top-20 candidates, apply metadata boost (+0.15 per matched flag), re-rank to top-5
- **LLM**: Claude generates one-sentence rationale per standard explaining relevance to the query
