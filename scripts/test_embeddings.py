from pathlib import Path
import sys

# Ajouter src/ au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from medieval_rag.embeddings.model_loader import load_embedding_model
from medieval_rag.embeddings.embedder import Embedder

def main():
    print("🚀 Test Embeddings")

    model, device = load_embedding_model(
        "intfloat/multilingual-e5-large",
        device="auto"
    )

    embedder = Embedder(model, max_batch_size=8)

    text = "Gerbert d'Aurillac est élu pape sous le nom de Sylvestre II en 999."

    emb = embedder.embed_text(text)

    print(f"Embedding OK : {len(emb)} dimensions")
    print(f"Premiers éléments : {emb[:5]}")

if __name__ == "__main__":
    main()
