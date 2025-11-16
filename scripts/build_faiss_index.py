from pathlib import Path
import sys
import json

import numpy as np
import faiss

# Ajouter src/ au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))


def load_corpus_embeddings(jsonl_path: Path):
    """
    Charge tous les embeddings + chunk_id depuis le JSONL.
    Retourne:
      - np.ndarray shape (N, D)
      - liste des chunk_ids dans le même ordre
    """
    vectors = []
    chunk_ids = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            emb = rec.get("embedding")
            cid = rec.get("chunk_id")

            if emb is None or cid is None:
                continue

            vectors.append(emb)
            chunk_ids.append(cid)

    if not vectors:
        raise RuntimeError("Aucun embedding trouvé dans le JSONL.")

    mat = np.array(vectors, dtype="float32")
    return mat, chunk_ids


def main():
    data_root = Path("data")
    corpus_jsonl = data_root / "chunks" / "standard" / "corpus_chunks.jsonl"

    if not corpus_jsonl.exists():
        raise FileNotFoundError(f"JSONL introuvable : {corpus_jsonl}")

    out_dir = data_root / "embeddings" / "e5_large"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.faiss"
    ids_path = out_dir / "index_ids.json"

    print("🚀 Construction de l'index FAISS")
    print(f"📥 Lecture du corpus : {corpus_jsonl}")

    mat, chunk_ids = load_corpus_embeddings(corpus_jsonl)
    n, d = mat.shape
    print(f"   → {n} embeddings, dimension {d}")

    # Index pour similarité cosinus (embeddings normalisés) : produit scalaire
    index = faiss.IndexFlatIP(d)
    index.add(mat)

    print(f"💾 Sauvegarde de l'index FAISS → {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"💾 Sauvegarde des IDs → {ids_path}")
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False, indent=2)

    print("✅ Index FAISS construit et sauvegardé.")


if __name__ == "__main__":
    main()
