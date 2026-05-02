import unittest
import json
import os
from src.rag_pipeline import BISRagPipeline

class TestBISRagPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small dummy corpus for testing
        cls.dummy_docs = [
            {"label": "IS 269 : 1989", "text": "Ordinary Portland Cement, 33 Grade specification. Used for general construction.", "metadata": {"is_pozzolana": False}},
            {"label": "IS 455 : 1989", "text": "Portland Slag Cement specification. Contains blast furnace slag.", "metadata": {"is_slag_cement": True}},
            {"label": "IS 1489 : 1991", "text": "Portland Pozzolana Cement specification. Uses fly ash or calcined clay.", "metadata": {"is_pozzolana": True}},
            {"label": "IS 8041 : 1990", "text": "Rapid Hardening Portland Cement specification. High early strength development.", "metadata": {"is_rapid_hardening": True}},
            {"label": "IS 8042 : 1989", "text": "White Portland Cement specification. Used for architectural finishes.", "metadata": {"is_white_cement": True}}
        ]
        cls.pipeline = BISRagPipeline(cls.dummy_docs)

    def test_basic_retrieval(self):
        query = "cement for fast hardening"
        retrieved, _, _ = self.pipeline.query(query, top_k=1)
        self.assertIn("IS 8041 : 1990", retrieved)

    def test_bm25_dense_hybrid(self):
        # Query that might be tricky for pure dense but clear for BM25
        query = "slag furnace cement"
        retrieved, _, _ = self.pipeline.query(query, top_k=1)
        self.assertIn("IS 455 : 1989", retrieved)

    def test_metadata_boosting(self):
        query = "fly ash pozzolana cement"
        retrieved, _, _ = self.pipeline.query(query, top_k=1)
        self.assertIn("IS 1489 : 1991", retrieved)

    def test_llm_rationales(self):
        # This tests if rationale generation (or fallback) works
        query = "standard for white cement"
        retrieved, rationales, _ = self.pipeline.query(query, top_k=1)
        self.assertTrue(len(retrieved) > 0)
        self.assertIn(retrieved[0], rationales)

if __name__ == "__main__":
    unittest.main()
