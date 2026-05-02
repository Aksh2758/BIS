"""
BIS Standards Recommendation Engine - Inference Entry Point
Usage: python inference.py --input <input.json> --output <output.json>
Set ANTHROPIC_API_KEY environment variable before running.
"""
import json
import argparse
from src.rag_pipeline import BISRagPipeline


def load_documents():
    """
    Load enriched chunks (with metadata) if available, else fall back to plain chunks.
    """
    try:
        with open("data/standards_chunks_enriched.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} enriched standards from standards_chunks_enriched.json")
        return docs
    except FileNotFoundError:
        pass

    try:
        with open("data/standards_chunks.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} standards from standards_chunks.json (no metadata)")
        return docs
    except FileNotFoundError:
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
    rag = BISRagPipeline(documents)

    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []

    for item in queries:
        query_id = item["id"]
        query_text = item["query"]

        retrieved, rationale_map, latency = rag.query(query_text, top_k=5)

        # Build rationale list aligned to retrieved standards
        rationale = [rationale_map.get(s, "") for s in retrieved]

        result = {
            "id": query_id,
            "retrieved_standards": retrieved,
            "rationale": rationale,
            "latency_seconds": round(latency, 4)
        }

        # Pass through expected_standards if present (for local eval)
        if "expected_standards" in item:
            result["expected_standards"] = item["expected_standards"]

        results.append(result)
        print(f"[{query_id}] Done in {latency:.2f}s — top: {retrieved[0] if retrieved else 'none'}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIS RAG Inference")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()
    main(args.input, args.output)
