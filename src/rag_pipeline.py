import time
import os
import re
import json
import logging
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
        logger.info(f"Initializing BISRagPipeline with {len(documents)} documents.")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.labels = [doc['label'] for doc in documents]
        self.texts = [doc['text'] for doc in documents]
        self.metadata = [doc.get('metadata', {}) for doc in documents]

        # Dense Index (FAISS)
        logger.info("Encoding documents for Dense Retrieval...")
        raw_embeddings = self.model.encode(self.texts, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings = np.array(raw_embeddings).astype('float32')
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        logger.info("FAISS index built.")

        # Sparse Index (BM25)
        logger.info("Building BM25 Sparse Index...")
        tokenized_corpus = [doc.lower().split() for doc in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built.")

        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            self.llm_client = Groq(api_key=api_key)
            logger.info("Groq LLM client initialized.")
        else:
            self.llm_client = None
            logger.warning("GROQ_API_KEY not set. Falling back to retrieval-only mode.")

    def _expand_query(self, query_text):
        """
        Query Expansion with a hard 3-second timeout.
        If Groq is slow or rate-limited, we skip expansion and use the original query.
        This ensures this step NEVER causes latency spikes.
        """
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
            logger.warning("Query expansion timed out (>3s) — using original query.")
        except Exception as e:
            logger.warning(f"Query expansion skipped: {e}")

        return query_text

    def _retrieve(self, query_text, top_k=20):
        """Hybrid Retrieval: BM25 + FAISS + RRF + Metadata Boost. 100% local, no API."""
        # Dense
        q_emb = self.model.encode([query_text], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype('float32')
        _, dense_indices = self.index.search(q_emb, min(top_k * 3, len(self.labels)))
        dense_results = dense_indices[0].tolist()

        # Sparse
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_results = np.argsort(bm25_scores)[::-1][:min(top_k * 3, len(self.labels))].tolist()

        # RRF
        k = 60
        rrf_scores = {}
        for rank, idx in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        top_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Metadata boost
        signals = _detect_query_signals(query_text)
        candidates = []
        for idx in top_indices[:top_k]:
            boost = _metadata_boost(self.metadata[idx], signals)
            candidates.append((rrf_scores[idx] + boost, idx))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in candidates]

    def _rerank(self, query_text, candidate_indices):
        """
        Cross-Encoder Re-ranking — runs locally on CPU.
        No API calls, no rate limits, deterministic quality.
        Replaces the LLM verification step entirely.
        """
        if not candidate_indices:
            return []
        logger.info(f"Re-ranking {len(candidate_indices)} candidates...")
        pairs = [[query_text, self.texts[idx]] for idx in candidate_indices]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        reranked = [idx for _, idx in sorted(zip(scores, candidate_indices), key=lambda x: x[0], reverse=True)]
        return reranked

    def _generate_rationale(self, query_text, standards):
        """
        Single batched LLM call for all standards.
        KEY FIX: This is now exactly 1 API call instead of 5+.
        The verification step has been removed — the cross-encoder already
        handles quality filtering far better than per-standard LLM checks.
        """
        if self.llm_client is None or not standards:
            return {s: "Matched via hybrid retrieval (BM25+FAISS) and cross-encoder re-ranking." for s in standards}

        standards_list = "\n".join(f"- {s}" for s in standards)
        prompt = (
            "You are a BIS compliance expert helping Indian MSEs.\n\n"
            f"Product/Query: {query_text}\n\n"
            f"Retrieved BIS standards:\n{standards_list}\n\n"
            "For each standard, write ONE sentence (max 20 words) explaining its relevance.\n"
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
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Rationale generation failed ({e}) — using fallback.")
            return {s: "Matched via hybrid retrieval and cross-encoder re-ranking." for s in standards}

    def query(self, text, top_k=5):
        start_time = time.time()

        # Step 1: Query Expansion (max 3s, skipped gracefully if slow)
        expanded_query = self._expand_query(text)

        # Step 2: Hybrid Retrieval — local only (BM25 + FAISS + RRF + metadata boost)
        candidate_indices = self._retrieve(expanded_query, top_k=20)

        # Step 3: Cross-Encoder Re-ranking — local only, replaces LLM verification
        reranked_indices = self._rerank(text, candidate_indices)

        # Step 4: Top-k selection
        top_indices = reranked_indices[:top_k]
        retrieved_labels = [self.labels[idx] for idx in top_indices]

        # Step 5: Rationale — single batched call (was 5+ separate calls before)
        rationale_map = self._generate_rationale(text, retrieved_labels)

        latency = time.time() - start_time
        logger.info(f"Query completed in {latency:.2f}s (max 2 LLM calls per query)")
        return retrieved_labels, rationale_map, latency
