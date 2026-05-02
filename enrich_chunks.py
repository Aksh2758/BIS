"""
Run this ONCE to generate data/standards_chunks_enriched.json from standards_chunks.json.
This adds metadata (category, type flags, part number) to each chunk for better retrieval.

Usage: python enrich_chunks.py
"""
import json
import re


def extract_metadata(label, text):
    label_upper = label.upper()
    text_upper = text.upper()

    categories = []
    if any(k in text_upper for k in ['CEMENT', 'PORTLAND', 'POZZOLANA', 'SLAG CEMENT', 'MASONRY CEMENT']):
        categories.append('cement')
    if any(k in text_upper for k in ['CONCRETE', 'REINFORCED', 'PRESTRESSED']):
        categories.append('concrete')
    if any(k in text_upper for k in ['AGGREGATE', 'COARSE AND FINE', 'NATURAL SOURCES FOR CONCRETE']):
        categories.append('aggregates')
    if any(k in text_upper for k in ['STEEL', 'IRON', 'DEFORMED BAR', 'WIRE STRAND']):
        categories.append('steel')
    if any(k in text_upper for k in ['BRICK', 'BLOCK', 'MASONRY', 'TILE']):
        categories.append('masonry')
    if any(k in text_upper for k in ['PIPE', 'DRAIN', 'CULVERT', 'SEWER']):
        categories.append('pipes')
    if any(k in text_upper for k in ['LIME', 'GYPSUM', 'PLASTER']):
        categories.append('binding_materials')
    if not categories:
        categories.append('general')

    part_match = re.search(r'PART\s*(\d+)', label_upper)
    part = int(part_match.group(1)) if part_match else None

    is_lightweight = any(k in text_upper for k in ['LIGHTWEIGHT', 'LIGHT WEIGHT', 'AERATED', 'CELLULAR', 'AUTOCLAVED'])
    is_rapid_hardening = 'RAPID HARDENING' in text_upper
    is_white = 'WHITE' in text_upper and 'CEMENT' in text_upper
    is_slag = 'SLAG' in text_upper and 'CEMENT' in text_upper
    is_pozzolana = 'POZZOLANA' in text_upper or 'POZZOLANIC' in text_upper

    scope_match = re.search(r'scope[^.—–-]*[.—–-]\s*(.{0,200})', text, re.IGNORECASE)
    scope = scope_match.group(1).strip() if scope_match else ''

    return {
        'categories': categories,
        'part': part,
        'is_lightweight': is_lightweight,
        'is_rapid_hardening': is_rapid_hardening,
        'is_white_cement': is_white,
        'is_slag_cement': is_slag,
        'is_pozzolana': is_pozzolana,
        'scope_snippet': scope[:200]
    }


if __name__ == "__main__":
    with open("data/standards_chunks.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    enriched = []
    for doc in raw:
        meta = extract_metadata(doc['label'], doc['text'])
        enriched.append({
            'label': doc['label'],
            'text': doc['text'],
            'metadata': meta
        })

    with open("data/standards_chunks_enriched.json", "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"Enriched {len(enriched)} chunks -> data/standards_chunks_enriched.json")
