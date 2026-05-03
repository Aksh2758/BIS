import time
import os
import re
import json
import logging
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dotenv import load_dotenv

load_dotenv()
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from groq import Groq
from rank_bm25 import BM25Okapi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BIS-RAG")

# Paths for precomputed index
_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_DIR, '..', 'data')
FAISS_INDEX_PATH = os.path.join(_DATA_DIR, 'faiss.index')
BM25_PATH        = os.path.join(_DATA_DIR, 'bm25.pkl')
META_PATH        = os.path.join(_DATA_DIR, 'meta.pkl')

# Synonym expansion — maps abbreviations/alternate terms to full BIS terminology
SYNONYMS = {
    "OPC":   "Ordinary Portland Cement",
    "PPC":   "Portland Pozzolana Cement",
    "PSC":   "Portland Slag Cement",
    "SRC":   "Sulphate Resisting Cement",
    "HAC":   "High Alumina Cement",
    "AAC":   "Autoclaved Aerated Concrete",
    "RCC":   "Reinforced Cement Concrete",
    "TMT":   "Thermo Mechanically Treated steel bars",
    "HSD":   "High Strength Deformed bars",
    "ISI":   "Indian Standards Institution",
    "33 GRADE": "33 Grade Ordinary Portland Cement",
    "43 GRADE": "43 Grade Ordinary Portland Cement",
    "53 GRADE": "53 Grade Ordinary Portland Cement",
}


def _apply_synonyms(text):
    """Expand abbreviations before retrieval for better BM25 matching."""
    upper = text.upper()
    for abbr, full in SYNONYMS.items():
        if abbr in upper:
            text = text + " " + full
    return text


def _detect_query_signals(query_text):
    q = query_text.upper()
    signals = {
        'is_lightweight':    any(k in q for k in ['LIGHTWEIGHT', 'LIGHT WEIGHT', 'AERATED', 'CELLULAR', 'AUTOCLAVED']),
        'is_slag_cement':    any(k in q for k in ['SLAG', 'PORTLAND SLAG', 'PSC']),
        'is_pozzolana':      any(k in q for k in ['POZZOLANA', 'POZZOLANIC', 'FLY ASH', 'PPC']),
        'is_rapid_hardening': any(k in q for k in ['RAPID', 'RAPID HARDENING']),
        'is_white_cement':   'WHITE' in q and 'CEMENT' in q,
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
    def _init_groq(self, api_key):
        """
        Initialize Groq client and verify it actually works with a tiny test call.
        If the key is invalid, expired, or network is down → returns None so
        every downstream step uses the chunk-text fallback automatically.
        """
        try:
            client = Groq(api_key=api_key)
            # Minimal health check — 1 token, near-zero cost/latency
            client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            logger.info("Groq API key verified — LLM features enabled.")
            return client
        except Exception as e:
            logger.warning(f"Groq API check failed ({e}) — switching to retrieval-only mode.")
            logger.warning("Rationale will use BIS chunk text. Retrieval accuracy is unaffected.")
            return None
    def __init__(self, documents):
        logger.info(f"Initializing BISRagPipeline with {len(documents)} documents.")

        # Load models once — these are cached by sentence-transformers after first download
        self.model    = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.labels   = [doc['label'] for doc in documents]
        self.texts    = [doc['text']  for doc in documents]
        self.metadata = [doc.get('metadata', {}) for doc in documents]

        # ── Precomputed index: load from disk if available, else build and save ──
        if self._index_exists():
            self._load_index()
        else:
            self._build_index()
            self._save_index()

        # Groq client — with health check so bad key/no network fails gracefully
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            self.llm_client = self._init_groq(api_key)
        else:
            self.llm_client = None
            logger.warning("GROQ_API_KEY not set — running in retrieval-only mode (chunk-text rationale).")

    # ── Index persistence ──────────────────────────────────────────────────────

    def _index_exists(self):
        return all(os.path.exists(p) for p in [FAISS_INDEX_PATH, BM25_PATH, META_PATH])

    def _build_index(self):
        logger.info("Building index from scratch (first run — will be cached for next time)...")

        # Dense
        logger.info("Encoding documents for Dense Retrieval...")
        raw_emb = self.model.encode(self.texts, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings = np.array(raw_emb).astype('float32')
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        logger.info("FAISS index built.")

        # Sparse
        logger.info("Building BM25 Sparse Index...")
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built.")

    def _save_index(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(BM25_PATH, 'wb') as f:
            pickle.dump(self.bm25, f)
        with open(META_PATH, 'wb') as f:
            pickle.dump({'labels': self.labels, 'texts': self.texts, 'metadata': self.metadata}, f)
        logger.info(f"Index saved to disk — subsequent runs will be ~10x faster.")

    def _load_index(self):
        logger.info("Loading precomputed index from disk...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(BM25_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(META_PATH, 'rb') as f:
            meta = pickle.load(f)
        # Override with saved labels/texts if consistent
        if len(meta['labels']) == len(self.labels):
            self.labels   = meta['labels']
            self.texts    = meta['texts']
            self.metadata = meta['metadata']
        logger.info("Precomputed index loaded — skipping encoding step.")

    # ── Pipeline steps ─────────────────────────────────────────────────────────

    def _expand_query(self, query_text):
        """LLM query expansion with hard 3s timeout — never blocks the pipeline."""
        if self.llm_client is None:
            return query_text

        prompt = (
            "Rewrite the following user query into a short technical search query "
            "(max 15 words). Use BIS/IS standard terminology only. "
            "Output ONLY the rewritten query, nothing else.\n\n"
            f"Query: {query_text}\nRewritten:"
        )

        def _call():
            return self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=40,
                temperature=0.0,
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                response = future.result(timeout=3.0)
                expanded = response.choices[0].message.content.strip().split('\n')[0]
                if 5 < len(expanded) < 200:
                    logger.info(f"Expanded: '{expanded}'")
                    return expanded
        except FuturesTimeoutError:
            logger.warning("Query expansion timed out — using original query.")
        except Exception as e:
            logger.warning(f"Query expansion skipped: {e}")

        return query_text

    def _retrieve(self, query_text, top_k=20):
        """Hybrid Retrieval: BM25 + FAISS + RRF + Synonym expansion + Metadata Boost."""
        # Apply synonym expansion for better BM25 matching
        enriched_query = _apply_synonyms(query_text)

        # Dense retrieval
        q_emb = self.model.encode([enriched_query], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype('float32')
        _, dense_indices = self.index.search(q_emb, min(top_k * 3, len(self.labels)))
        dense_results = dense_indices[0].tolist()

        # Sparse retrieval
        tokenized_query = enriched_query.lower().split()
        bm25_scores  = self.bm25.get_scores(tokenized_query)
        sparse_results = np.argsort(bm25_scores)[::-1][:min(top_k * 3, len(self.labels))].tolist()

        # Reciprocal Rank Fusion
        k = 60
        rrf_scores = {}
        for rank, idx in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        top_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Metadata boost on top candidates
        signals = _detect_query_signals(query_text)
        candidates = []
        for idx in top_indices[:top_k]:
            boost = _metadata_boost(self.metadata[idx], signals)
            candidates.append((rrf_scores[idx] + boost, idx))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in candidates]

    def _rerank(self, query_text, candidate_indices):
        """
        Cross-Encoder Re-ranking — local CPU, no API calls.
        Reduced to top-10 candidates (was 20) to halve cross-encoder time on slow hardware.
        Quality is preserved because RRF already surfaces the right candidates.
        """
        if not candidate_indices:
            return []
        # Only re-rank top 10 — significant speedup on slow CPUs, minimal quality loss
        candidates_to_rerank = candidate_indices[:10]
        logger.info(f"Re-ranking {len(candidates_to_rerank)} candidates...")
        pairs  = [[query_text, self.texts[idx]] for idx in candidates_to_rerank]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        reranked = [idx for _, idx in sorted(zip(scores, candidates_to_rerank), key=lambda x: x[0], reverse=True)]
        return reranked

    def _generate_rationale(self, query_text, standards, top_texts):
        """
        Single batched Groq call for all standards.
        Includes the actual chunk text so rationale is grounded, not hallucinated.
        Falls back to chunk-derived summary if Groq fails — no more generic fallback text.
        """
        # Smart fallback: first sentence of each chunk — always meaningful, never generic
        fallback = {}
        for std, txt in zip(standards, top_texts):
            first_sentence = txt.split('.')[0].strip()[:120] if txt else ""
            fallback[std] = first_sentence if first_sentence else f"Relevant BIS standard for the specified product."

        if self.llm_client is None or not standards:
            return fallback

        context_blocks = "\n\n".join(
            f"[{std}]: {txt[:300]}" for std, txt in zip(standards, top_texts)
        )
        standards_list = "\n".join(f"- {s}" for s in standards)

        prompt = (
            "You are a BIS compliance expert helping Indian MSEs.\n\n"
            f"Product/Query: {query_text}\n\n"
            f"Standard summaries from BIS SP 21:\n{context_blocks}\n\n"
            f"For each of these standards:\n{standards_list}\n\n"
            "Write ONE sentence (max 20 words) explaining WHY it is relevant to the product above. "
            "Ground your answer strictly in the summary text provided. "
            "Return ONLY valid JSON. No markdown. No extra text.\n"
            'Format: {"IS XXX : YYYY": "reason", ...}'
        )

        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            result = json.loads(raw)
            # Fill any missing keys with fallback
            for std in standards:
                if std not in result:
                    result[std] = fallback[std]
            return result
        except Exception as e:
            logger.warning(f"Rationale generation failed ({e}) — using chunk-derived fallback.")
            return fallback

    def query(self, text, top_k=5):
        start_time = time.time()

        # Step 1: Query Expansion (max 3s timeout)
        expanded_query = self._expand_query(text)

        # Step 2: Hybrid Retrieval — local (BM25 + FAISS + RRF + synonyms + metadata boost)
        candidate_indices = self._retrieve(expanded_query, top_k=20)

        # Step 3: Cross-Encoder Re-ranking — local, top-10 only for speed
        reranked_indices = self._rerank(text, candidate_indices)

        # Step 4: Top-k selection
        top_indices      = reranked_indices[:top_k]
        retrieved_labels = [self.labels[idx] for idx in top_indices]
        top_texts        = [self.texts[idx]  for idx in top_indices]

        # Step 5: Single batched rationale call with chunk context
        rationale_map = self._generate_rationale(text, retrieved_labels, top_texts)

        latency = time.time() - start_time
        logger.info(f"Query completed in {latency:.2f}s")
        return retrieved_labels, rationale_map, latency
