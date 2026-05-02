import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class SimpleRAG:
    def __init__(self, documents):
        """
        documents: list of dicts with keys 'label' and 'text'
        - label: the IS standard identifier e.g. "IS 269 : 1989"
        - text:  the full summary text used for embedding/search
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.labels = [doc['label'] for doc in documents]
        texts = [doc['text'] for doc in documents]

        print(f"Encoding {len(texts)} documents...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings).astype('float32'))
        print("FAISS index built.")

    def query(self, text, top_k=5):
        start_time = time.time()

        query_embedding = self.model.encode([text])
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), top_k
        )

        # Return IS standard labels (e.g. "IS 269 : 1989"), not raw text
        results = [self.labels[i] for i in indices[0]]

        latency = time.time() - start_time
        return results, latency
