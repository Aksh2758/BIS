import json
import argparse
from src.rag_pipeline import SimpleRAG


def load_documents():
    """
    Load the structured dataset extracted from dataset.pdf.
    Returns a list of dicts: [{'label': 'IS 269 : 1989', 'text': '...'}, ...]
    Falls back to plain text file if JSON not available.
    """
    try:
        with open("data/standards_chunks.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} standards from standards_chunks.json")
        return docs
    except FileNotFoundError:
        # Fallback: plain text, one IS standard per line
        with open("data/bis_docs.txt", "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        # Each line format: "IS XXX : YYYY | <text>"
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
    rag = SimpleRAG(documents)

    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []

    for item in queries:
        query_id = item["id"]
        query_text = item["query"]

        retrieved, latency = rag.query(query_text, top_k=5)

        result = {
            "id": query_id,
            "query": query_text,
            "retrieved_standards": retrieved,
            "latency_seconds": round(latency, 4)
        }

        # Pass through expected_standards if present in input (for eval script)
        if "expected_standards" in item:
            result["expected_standards"] = item["expected_standards"]

        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIS RAG Inference")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    main(args.input, args.output)
