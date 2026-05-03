"""
BIS Standards Recommendation Engine - Inference Entry Point

Usage:
    python inference.py --input <input.json> --output <output.json>

LLM (Groq) is disabled during inference for speed and reliability.
All scoring metrics (Hit Rate, MRR, Latency) are purely retrieval-based.
Groq is only used in the Streamlit UI for rationale generation.
"""
import json
import os
import argparse
from src.pipeline import BISRagPipeline


def load_documents():
    for path in ["data/standards_chunks_enriched.json", "data/standards_chunks.json"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            print(f"Loaded {len(docs)} standards from {path}")
            return docs
    with open("data/bis_docs.txt", "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    docs = []
    for line in lines:
        if '|' in line:
            label, text = line.split('|', 1)
        else:
            label, text = line, line
        docs.append({'label': label.strip(), 'text': text.strip()})
    print(f"Loaded {len(docs)} standards from bis_docs.txt (fallback)")
    return docs


def main(input_path, output_path):
    documents = load_documents()

    # Disable LLM for inference — all steps are local
    # Avoids Groq rate limits and ensures consistent <5s latency
    os.environ.pop("GROQ_API_KEY", None)

    rag = BISRagPipeline(documents)

    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []

    for item in queries:
        query_text = str(item.get("query", "")).strip()

        try:
            retrieved, _, latency = rag.query(query_text, top_k=5)
        except Exception as e:
            print(f"[{item.get('id','?')}] ERROR: {e} — returning empty result")
            retrieved, latency = [], 0.0

        safe_retrieved = retrieved if retrieved else []

        # Strict output schema matching required format exactly
        # expected_standards always before retrieved_standards when present
        if "expected_standards" in item:
            result = {
                "id": item.get("id", "unknown"),
                "query": query_text,
                "expected_standards": item["expected_standards"],
                "retrieved_standards": safe_retrieved,
                "latency_seconds": round(latency, 4)
            }
        else:
            result = {
                "id": item.get("id", "unknown"),
                "query": query_text,
                "retrieved_standards": safe_retrieved,
                "latency_seconds": round(latency, 4)
            }

        results.append(result)
        top = safe_retrieved[0] if safe_retrieved else "none"
        print(f"[{item.get('id','?')}] Done in {latency:.2f}s — top: {top}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIS RAG Inference")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
