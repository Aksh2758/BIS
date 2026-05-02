import time
import os
import re
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # Loads GROQ_API_KEY from .env file automatically
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq


def _detect_query_signals(query_text):
    q = query_text.upper()
    signals = {
        'is_lightweight': any(k in q for k in ['LIGHTWEIGHT', 'LIGHT WEIGHT', 'AERATED', 'CELLULAR', 'AUTOCLAVED']),
        'is_slag_cement': any(k in q for k in ['SLAG', 'PORTLAND SLAG']),
        'is_pozzolana': any(k in q for k in ['POZZOLANA', 'POZZOLANIC', 'FLY ASH']),
        'is_rapid_hardening': any(k in q for k in ['RAPID', 'RAPID HARDENING']),
        'is_white_cement': 'WHITE' in q and 'CEMENT' in q,
        'part_hint': None,
    }
    part_match = re.search(r'PART\s*(\d+)', q)
    if part_match:
        signals['part_hint'] = int(part_match.group(1))
    return signals


def _metadata_boost(doc_meta, signals):
    boost = 0.0
    for flag in ['is_lightweight', 'is_slag_cement', 'is_pozzolana', 'is_rapid_hardening', 'is_white_cement']:
        if signals.get(flag) and doc_meta.get(flag):
            boost += 0.15
    if signals.get('part_hint') is not None and doc_meta.get('part') == signals['part_hint']:
        boost += 0.15
    return min(boost, 0.3)


class BISRagPipeline:
    def __init__(self, documents):
        """
        documents: list of dicts with keys:
          - label: IS standard identifier e.g. "IS 269 : 1989"
          - text:  full summary text
          - metadata: dict with category/type flags (optional)
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.labels = [doc['label'] for doc in documents]
        self.metadata = [doc.get('metadata', {}) for doc in documents]
        texts = [doc['text'] for doc in documents]

        print(f"Encoding {len(texts)} documents...")
        raw_embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings = np.array(raw_embeddings).astype('float32')

        # Cosine similarity via normalized vectors + IndexFlatIP
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print("FAISS index built (cosine similarity).")

        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            self.llm_client = Groq(api_key=api_key)
            print("Groq LLM client initialized.")
        else:
            self.llm_client = None
            print("WARNING: GROQ_API_KEY not set. Skipping LLM rationale.")

    def _retrieve(self, query_text, top_k=10):
        q_emb = self.model.encode([query_text], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype('float32')

        # Fetch 2x candidates for re-ranking
        scores, indices = self.index.search(q_emb, min(top_k * 2, len(self.labels)))

        signals = _detect_query_signals(query_text)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            boost = _metadata_boost(self.metadata[idx], signals)
            candidates.append((float(score) + boost, idx))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [self.labels[idx] for _, idx in candidates[:top_k]]

    def _generate_rationale(self, query_text, standards):
        """Use Groq (llama-3.1-8b-instant) to generate rationale for each standard."""
        if self.llm_client is None:
            return {s: "Matched based on semantic similarity to product description." for s in standards}

        standards_list = "\n".join(f"- {s}" for s in standards)
        prompt = f"""You are a BIS (Bureau of Indian Standards) compliance expert helping Indian MSEs.

Product/Query: {query_text}

Retrieved BIS standards:
{standards_list}

For each standard, write ONE concise sentence (max 20 words) explaining why it is relevant.
If a standard seems unrelated, write "May not be directly relevant."

Return ONLY a valid JSON object. No markdown, no explanation outside the JSON.
Format: {{"IS XXX : YYYY": "reason here", ...}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if model adds them
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except Exception as e:
            print(f"LLM rationale failed: {e}")
            return {s: "Matched based on semantic similarity to product description." for s in standards}

    def query(self, text, top_k=5):
        start_time = time.time()
        retrieved_labels = self._retrieve(text, top_k=top_k)
        rationale_map = self._generate_rationale(text, retrieved_labels)
        latency = time.time() - start_time
        return retrieved_labels, rationale_map, latency
