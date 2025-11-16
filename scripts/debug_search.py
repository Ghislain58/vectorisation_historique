from pathlib import Path
import sys
import json

import numpy as np
import faiss

# Ajouter src/ au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from medieval_rag.embeddings.model_loader import load_embedding_model
from medieval_rag.embeddings.embedder import Embedder


def load_index(index_path: Path, ids_path: Path):
    index = faiss.read_index(str(index_path))
    with ids_path.open("r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    return index, chunk_ids


def load_corpus_records(jsonl_path: Path):
    """
    Charge toutes les lignes du JSONL dans un dict :
      chunk_id -> record complet
    """
    records = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            if cid:
                records[cid] = rec
    return records


def main():
    data_root = Path("data")
    corpus_jsonl = data_root / "chunks" / "standard" / "corpus_chunks.jsonl"
    index_path = data_root / "embeddings" / "e5_large" / "index.faiss"
    ids_path = data_root / "embeddings" / "e5_large" / "index_ids.json"

    if not index_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Index FAISS ou fichier d'IDs introuvable. Lance d'abord build_faiss_index.py")

    print("🚀 Chargement du modèle d'embedding...")
    model, device = load_embedding_model(
        "intfloat/multilingual-e5-large",
        device="auto"
    )
    embedder = Embedder(model, max_batch_size=8)

    print("📥 Chargement de l'index FAISS et des IDs...")
    index, chunk_ids = load_index(index_path, ids_path)

    print("📥 Chargement du corpus JSONL (pour retrouver les textes)...")
    records = load_corpus_records(corpus_jsonl)

    print("✅ Prêt pour la recherche sémantique.")
    print("Tape une question historique (ou juste Enter pour quitter).")

    while True:
        query = input("\n❓ Question : ").strip()
        if not query:
            print("🔚 Fin.")
            break

        q_emb = np.array([embedder.embed_text(query)], dtype="float32")
        top_k = 5

        scores, indices = index.search(q_emb, top_k)
        scores = scores[0]
        indices = indices[0]

        print(f"\n🔎 Top {top_k} résultats :")
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0:
                continue
            cid = chunk_ids[idx]
            rec = records.get(cid)
            if not rec:
                continue

            title = rec.get("title") or rec.get("doc_id")
            pages = f"{rec.get('page_start')}–{rec.get('page_end')}"
            text = rec.get("text", "")
            preview = text[:400].replace("\n", " ")

            print(f"\n#{rank} — score={score:.3f}")
            print(f"📘 {title}  (pages {pages})")
            print(f"🧩 Chunk ID : {cid}")
            print(f"📝 Aperçu : {preview}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
