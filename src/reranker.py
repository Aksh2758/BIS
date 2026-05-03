"""
reranker.py — Cross-Encoder Re-ranking Module

Uses ms-marco-MiniLM-L-6-v2 to score query-document pairs.
Runs 100% locally on CPU — no API calls, no rate limits.
Operates on top-10 candidates from retriever for speed on consumer hardware.
"""
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger("BIS-RAG")


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
            Reranked list of indices, best first.
        """
        if not candidate_indices:
            return []

        candidates_to_rerank = candidate_indices[:top_n]
        logger.info(f"Re-ranking {len(candidates_to_rerank)} candidates...")

        pairs  = [[query_text, texts[idx]] for idx in candidates_to_rerank]
        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked = [
            idx for _, idx in sorted(
                zip(scores, candidates_to_rerank),
                key=lambda x: x[0],
                reverse=True
            )
        ]
        return reranked
