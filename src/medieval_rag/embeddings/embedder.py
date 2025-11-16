from typing import List
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model: SentenceTransformer, max_batch_size: int = 32):
        self.model = model
        self.max_batch_size = max_batch_size

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embedding par batch pour éviter les erreurs du type :
        ValueError: Batch size X > max batch size Y
        """
        embeddings = []
        n = len(texts)

        for i in range(0, n, self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            batch_emb = self.model.encode(
                batch,
                batch_size=self.max_batch_size,
                show_progress_bar=False,
                convert_to_numpy=False,
                normalize_embeddings=True
            )
            embeddings.extend([emb.tolist() for emb in batch_emb])

        return embeddings

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
