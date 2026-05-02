import pdfplumber
import re
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

chunks = []  # Each entry: {'label': 'IS 269: 1989', 'text': '...full summary...'}

with pdfplumber.open('data/dataset.pdf') as pdf:
    total = len(pdf.pages)
    print(f'Total pages: {total}')

    current_label = None
    current_text = []

    for i, page in enumerate(pdf.pages):
        if i % 100 == 0:
            print(f'  Processing page {i}/{total}...')

        text = page.extract_text()
        if not text:
            continue

        # Detect 'SUMMARY OF' header followed by IS number on next line
        summary_match = re.search(
            r'SUMMARY\s+OF\s*\n(IS\s+[\d\w\s().]+?:\s*\d{4})',
            text,
            re.IGNORECASE
        )

        if summary_match:
            # Save previous chunk
            if current_label and current_text:
                full_text = ' '.join(current_text).strip()
                chunks.append({
                    'label': current_label,
                    'text': full_text[:3000]
                })

            # Start new chunk
            raw_label = summary_match.group(1).strip()
            current_label = re.sub(r'\s+', ' ', raw_label).strip()
            current_text = [text]
        elif current_label:
            current_text.append(text)

    # Save the last chunk
    if current_label and current_text:
        full_text = ' '.join(current_text).strip()
        chunks.append({
            'label': current_label,
            'text': full_text[:3000]
        })

print(f'\nExtracted {len(chunks)} standards')
print('\nFirst 10 labels:')
for c in chunks[:10]:
    print(f'  {c["label"]}')
print('\nLast 5 labels:')
for c in chunks[-5:]:
    print(f'  {c["label"]}')

# Save structured JSON
with open('data/standards_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
print('\nSaved to data/standards_chunks.json')

# Also write a simple text file: one line per standard (label + text)
with open('data/bis_docs.txt', 'w', encoding='utf-8') as f:
    for c in chunks:
        # Each line: "IS 269: 1989 | <scope text>"
        line = f'{c["label"]} | {c["text"][:500]}'
        # Collapse newlines
        line = line.replace('\n', ' ').replace('\r', ' ')
        f.write(line + '\n')
print(f'Written {len(chunks)} lines to data/bis_docs.txt')
