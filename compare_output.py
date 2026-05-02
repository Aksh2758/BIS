import json

with open('my_output.json', encoding='utf-8') as f:
    mine = json.load(f)
with open('sample_output.json', encoding='utf-8') as f:
    sample = json.load(f)

print('=== COMPARISON: MY OUTPUT vs SAMPLE OUTPUT ===\n')
for m, s in zip(mine, sample):
    qid = m['id']
    expected = s['expected_standards']
    my_results = m['retrieved_standards']
    sample_results = s['retrieved_standards']

    def normalize(x):
        return x.replace(' ', '').lower()

    norm_expected = [normalize(e) for e in expected]
    norm_my_top3 = [normalize(r) for r in my_results[:3]]
    hit = any(e in norm_my_top3 for e in norm_expected)

    print(f'ID: {qid}')
    print(f'  Expected    : {expected}')
    print(f'  My results  : {my_results}')
    print(f'  Sample      : {sample_results}')
    print(f'  Hit@3       : {"YES" if hit else "NO"}')
    print()
