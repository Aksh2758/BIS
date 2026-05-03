"""
Run this ONCE before inference to precompute and cache the FAISS + BM25 index.
After running this, inference.py will be ~10x faster on every subsequent run.

Usage:
    python build_index.py
"""
import json, os, sys

ENRICHED_PATH = os.path.join('data', 'standards_chunks_enriched.json')
RAW_PATH      = os.path.join('data', 'standards_chunks.json')

def load_documents():
    for path in [ENRICHED_PATH, RAW_PATH]:
        if os.path.exists(path):
            print(f"Loading documents from {path}...")
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    print("ERROR: No standards JSON found in data/", file=sys.stderr)
    sys.exit(1)

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from rag_pipeline import BISRagPipeline, FAISS_INDEX_PATH, BM25_PATH, META_PATH
    import os

    # Delete old index if it exists to force rebuild
    for p in [FAISS_INDEX_PATH, BM25_PATH, META_PATH]:
        if os.path.exists(p):
            os.remove(p)
            print(f"Removed old index: {p}")

    documents = load_documents()
    print(f"Loaded {len(documents)} documents. Building index...")
    pipeline = BISRagPipeline(documents)
    print("\nIndex built and saved successfully!")
    print("You can now run inference.py — first-run encoding is done.")
