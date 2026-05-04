"""
reranker.py — Cross-Encoder Re-ranking Module

Uses ms-marco-MiniLM-L-6-v2 to score query-document pairs.
Runs 100% locally on CPU — no API calls, no rate limits.
Operates on top-10 candidates from retriever for speed on consumer hardware.

Improvements:
- Batched predict (batch_size=32) cuts CPU overhead vs sequential scoring
- Returns raw scores for downstream confidence-based filtering
"""
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger("BIS-RAG")

# Threshold below which we flag low-confidence results to the caller
LOW_CONFIDENCE_THRESHOLD = -3.0


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Cross-encoder reranker loaded.")

    def rerank(self, query_text, candidate_indices, texts, top_n=10):
        """
        Re-rank candidate indices using cross-encoder scoring.

        Args:
            query_text:        the user's original query
            candidate_indices: list of doc indices from retriever
            texts:             full list of document texts
            top_n:             only re-rank top_n candidates (default 10 for speed)

        Returns:
            reranked_indices: list of doc indices ordered best-first
            scores:           parallel list of raw cross-encoder scores
            low_confidence:   True if even the top result scores poorly
        """
        if not candidate_indices:
            return [], [], False

        candidates_to_rerank = candidate_indices[:top_n]
        logger.info(f"Re-ranking {len(candidates_to_rerank)} candidates (batched)...")

        pairs = [[query_text, texts[idx]] for idx in candidates_to_rerank]

        # batch_size=32 sends all pairs in one forward pass on CPU,
        # eliminating per-pair Python loop overhead (~3x faster vs default)
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        scored = sorted(
            zip(scores, candidates_to_rerank),
            key=lambda x: x[0],
            reverse=True,
        )
        reranked_scores  = [s for s, _ in scored]
        reranked_indices = [i for _, i in scored]

        top_score      = reranked_scores[0] if reranked_scores else 0.0
        low_confidence = top_score < LOW_CONFIDENCE_THRESHOLD

        if low_confidence:
            logger.info(f"Low-confidence result: top score={top_score:.2f} < {LOW_CONFIDENCE_THRESHOLD}")

        return reranked_indices, reranked_scores, low_confidence
