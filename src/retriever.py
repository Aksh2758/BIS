"""
retriever.py — Hybrid Retrieval Module

Handles:
- FAISS dense retrieval (all-MiniLM-L6-v2)
- BM25 sparse retrieval with synonym expansion
- Reciprocal Rank Fusion (RRF)
- Metadata boosting
- Precomputed index persistence (load/save)
"""
import os
import re
import pickle
import logging
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("BIS-RAG")

# Paths for precomputed index
_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_DIR, '..', 'data')
FAISS_INDEX_PATH = os.path.join(_DATA_DIR, 'faiss.index')
BM25_PATH        = os.path.join(_DATA_DIR, 'bm25.pkl')
META_PATH        = os.path.join(_DATA_DIR, 'meta.pkl')

# Synonym expansion — abbreviations → full BIS terminology
SYNONYMS = {
    "OPC":      "Ordinary Portland Cement",
    "PPC":      "Portland Pozzolana Cement",
    "PSC":      "Portland Slag Cement",
    "SRC":      "Sulphate Resisting Cement",
    "HAC":      "High Alumina Cement",
    "AAC":      "Autoclaved Aerated Concrete",
    "RCC":      "Reinforced Cement Concrete",
    "TMT":      "Thermo Mechanically Treated steel bars",
    "HSD":      "High Strength Deformed bars",
    "ISI":      "Indian Standards Institution",
    "33 GRADE": "33 Grade Ordinary Portland Cement",
    "43 GRADE": "43 Grade Ordinary Portland Cement",
    "53 GRADE": "53 Grade Ordinary Portland Cement",
}


def apply_synonyms(text):
    """Expand abbreviations before retrieval for better BM25 matching."""
    upper = text.upper()
    for abbr, full in SYNONYMS.items():
        if abbr in upper:
            text = text + " " + full
    return text


def detect_query_signals(query_text):
    """Detect domain-specific signals in query for metadata boosting."""
    q = query_text.upper()
    signals = {
        'is_lightweight':     any(k in q for k in ['LIGHTWEIGHT', 'LIGHT WEIGHT', 'AERATED', 'CELLULAR', 'AUTOCLAVED']),
        'is_slag_cement':     any(k in q for k in ['SLAG', 'PORTLAND SLAG', 'PSC']),
        'is_pozzolana':       any(k in q for k in ['POZZOLANA', 'POZZOLANIC', 'FLY ASH', 'PPC']),
        'is_rapid_hardening': any(k in q for k in ['RAPID', 'RAPID HARDENING']),
        'is_white_cement':    'WHITE' in q and 'CEMENT' in q,
        'part_hint': None,
    }
    part_match = re.search(r'PART\s*(\d+)', q)
    if part_match:
        signals['part_hint'] = int(part_match.group(1))
    return signals


def metadata_boost(doc_meta, signals):
    """Apply score boost based on metadata flag matches."""
    boost = 0.0
    for flag in ['is_lightweight', 'is_slag_cement', 'is_pozzolana', 'is_rapid_hardening', 'is_white_cement']:
        if signals.get(flag) and doc_meta.get(flag):
            boost += 0.15
    if signals.get('part_hint') is not None and doc_meta.get('part') == signals['part_hint']:
        boost += 0.15
    return min(boost, 0.3)


class HybridRetriever:
    def __init__(self, labels, texts, metadata, embed_model: SentenceTransformer):
        self.labels   = labels
        self.texts    = texts
        self.metadata = metadata
        self.model    = embed_model

        if self._index_exists():
            self._load_index()
        else:
            self._build_index()
            self._save_index()

    # ── Index persistence ──────────────────────────────────────────

    def _index_exists(self):
        return all(os.path.exists(p) for p in [FAISS_INDEX_PATH, BM25_PATH, META_PATH])

    def _build_index(self):
        logger.info("Building index from scratch (first run — will be cached)...")
        logger.info("Encoding documents for Dense Retrieval...")
        raw_emb = self.model.encode(self.texts, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings = np.array(raw_emb).astype('float32')
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)
        logger.info("FAISS index built.")

        logger.info("Building BM25 Sparse Index...")
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built.")

    def _save_index(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
        with open(BM25_PATH, 'wb') as f:
            pickle.dump(self.bm25, f)
        with open(META_PATH, 'wb') as f:
            pickle.dump({'labels': self.labels, 'texts': self.texts, 'metadata': self.metadata}, f)
        logger.info("Index saved to disk — subsequent runs will be ~10x faster.")

    def _load_index(self):
        logger.info("Loading precomputed index from disk...")
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(BM25_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(META_PATH, 'rb') as f:
            meta = pickle.load(f)
        if len(meta['labels']) == len(self.labels):
            self.labels   = meta['labels']
            self.texts    = meta['texts']
            self.metadata = meta['metadata']
        logger.info("Precomputed index loaded — skipping encoding step.")

    # ── Retrieval ──────────────────────────────────────────────────

    def retrieve(self, query_text, top_k=20):
        """
        Hybrid retrieval: FAISS dense + BM25 sparse, fused with RRF.
        Applies synonym expansion and metadata boosting.
        Returns list of candidate indices ranked by combined score.
        """
        enriched_query = apply_synonyms(query_text)

        # Dense retrieval
        q_emb = self.model.encode([enriched_query], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype('float32')
        _, dense_indices = self.faiss_index.search(q_emb, min(top_k * 3, len(self.labels)))
        dense_results = dense_indices[0].tolist()

        # Sparse retrieval
        tokenized_query = enriched_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_results = np.argsort(bm25_scores)[::-1][:min(top_k * 3, len(self.labels))].tolist()

        # Reciprocal Rank Fusion (k=60)
        k = 60
        rrf_scores = {}
        for rank, idx in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        top_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Metadata boost on top candidates
        signals = detect_query_signals(query_text)
        candidates = []
        for idx in top_indices[:top_k]:
            boost = metadata_boost(self.metadata[idx], signals)
            candidates.append((rrf_scores[idx] + boost, idx))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in candidates]
