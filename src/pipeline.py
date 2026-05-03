"""
pipeline.py — BIS RAG Pipeline Orchestrator

Wires together all modules:
  Retriever (hybrid BM25+FAISS) → Reranker (cross-encoder) → Generator (rationale)

This is the single entry point used by inference.py and app.py.
"""
import os
import time
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .retriever import HybridRetriever
from .reranker  import CrossEncoderReranker
from .generator import RationaleGenerator

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BIS-RAG")


class BISRagPipeline:
    def __init__(self, documents):
        logger.info(f"Initializing BISRagPipeline with {len(documents)} documents.")

        labels   = [doc['label']           for doc in documents]
        texts    = [doc['text']            for doc in documents]
        metadata = [doc.get('metadata', {}) for doc in documents]

        # Shared embedding model — loaded once, used by retriever
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize modules
        self.retriever  = HybridRetriever(labels, texts, metadata, embed_model)
        self.reranker   = CrossEncoderReranker()
        self.generator  = RationaleGenerator(api_key=os.environ.get("GROQ_API_KEY"))

        # Keep references for label/text lookup
        self.labels = self.retriever.labels
        self.texts  = self.retriever.texts

    def query(self, text, top_k=5):
        """
        Full RAG pipeline:
          1. Input validation
          2. Query expansion (LLM, optional, 3s timeout)
          3. Hybrid retrieval (FAISS + BM25 + RRF + metadata boost)
          4. Cross-encoder reranking (local)
          5. Rationale generation (LLM or chunk-text fallback)

        Returns:
            retrieved_labels: list of IS standard strings
            rationale_map:    dict {label: explanation}
            latency:          float seconds
        """
        start_time = time.time()

        # Input validation
        if not text or not str(text).strip():
            logger.warning("Empty or null query — returning empty result.")
            return [], {}, time.time() - start_time

        text = str(text).strip()[:512]

        # Step 1: Query Expansion
        expanded_query = self.generator.expand_query(text)

        # Step 2: Hybrid Retrieval
        candidate_indices = self.retriever.retrieve(expanded_query, top_k=20)

        # Step 3: Cross-Encoder Reranking
        reranked_indices = self.reranker.rerank(text, candidate_indices, self.texts)

        # Step 4: Top-k selection
        top_indices      = reranked_indices[:top_k]
        retrieved_labels = [self.labels[i] for i in top_indices]
        top_texts        = [self.texts[i]  for i in top_indices]

        # Step 5: Rationale Generation
        rationale_map = self.generator.generate(text, retrieved_labels, top_texts)

        latency = time.time() - start_time
        logger.info(f"Query completed in {latency:.2f}s")
        return retrieved_labels, rationale_map, latency
