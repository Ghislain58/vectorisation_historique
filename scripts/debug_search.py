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


def load_corpus(jsonl_path: Path):
    """
    Charge corpus_chunks.jsonl en mémoire.
    Retourne :
      - records_by_id: dict chunk_id -> record
    """
    records_by_id = {}

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Fichier JSONL introuvable : {jsonl_path}")

    print(f"📂 Chargement du corpus : {jsonl_path}")
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            if not cid:
                continue
            records_by_id[cid] = rec

    print(f"   → {len(records_by_id)} chunks chargés")
    return records_by_id


def load_faiss_index(index_path: Path, ids_path: Path):
    """
    Charge l'index FAISS et la liste des chunk_ids correspondants.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Fichier d'IDs introuvable : {ids_path}")

    print(f"📦 Chargement index FAISS : {index_path}")
    index = faiss.read_index(str(index_path))

    print(f"📄 Chargement IDs : {ids_path}")
    with ids_path.open("r", encoding="utf-8") as f:
        ids_list = json.load(f)

    print(f"   → {len(ids_list)} IDs chargés, {index.ntotal} vecteurs dans l'index.")
    if len(ids_list) != index.ntotal:
        print("⚠️  Mismatch entre nombre d'IDs et nombre de vecteurs FAISS !")

    return index, ids_list


def pretty_print_result(rank, score, rec):
    """
    Affiche un chunk de manière lisible, avec entités si présentes.
    """
    doc_id = rec.get("doc_id")
    source = rec.get("source")
    title = rec.get("title")
    page_start = rec.get("page_start")
    page_end = rec.get("page_end")
    entities = rec.get("entities") or {}
    year_min = rec.get("year_min")
    year_max = rec.get("year_max")

    text_preview = (rec.get("text") or "").replace("\n", " ")
    if len(text_preview) > 300:
        text_preview = text_preview[:300] + "..."

    print(f"\n=== Résultat #{rank} — score={score:.4f} ===")
    print(f"📚 Doc   : {doc_id} ({source})")
    if title:
        print(f"📝 Titre : {title}")
    print(f"📄 Pages : {page_start}–{page_end}")

    # Entités
    persons = entities.get("persons") or []
    places = entities.get("places") or []
    years = entities.get("years") or []

    if persons or places or years or (year_min is not None or year_max is not None):
        print("🔎 Entités détectées :")
        if persons:
            print(f"   👤 Persons : {', '.join(persons)}")
        if places:
            print(f"   📍 Places  : {', '.join(places)}")
        if years:
            print(f"   📅 Years   : {', '.join(map(str, years))}")
        if year_min is not None or year_max is not None:
            print(f"   ⏳ Interval : {year_min}–{year_max}")

    print(f"📑 Texte : {text_preview}")


def main():
    data_root = Path("data")
    jsonl_path = data_root / "chunks" / "standard" / "corpus_chunks.jsonl"
    index_dir = data_root / "embeddings" / "e5_large"
    index_path = index_dir / "index.faiss"
    ids_path = index_dir / "index_ids.json"

    # Chargement corpus + index
    records_by_id = load_corpus(jsonl_path)
    index, ids_list = load_faiss_index(index_path, ids_path)

    # Chargement modèle d'embedding
    print("🧠 Chargement du modèle d'embedding pour les requêtes...")
    model, device = load_embedding_model(
        "intfloat/multilingual-e5-large",
        device="auto"
    )
    embedder = Embedder(model, max_batch_size=16)

    print("\n✅ Prêt pour la recherche sémantique.")
    print("   Tape une question en français (ou latin), ou 'q' pour quitter.")

    while True:
        try:
            query = input("\n❓ Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Fin de session.")
            break

        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            print("👋 Fin de session.")
            break

        # Embedding de la requête
        q_emb = embedder.embed_texts([query])[0]
        q_vec = np.array([q_emb], dtype="float32")

        # Recherche FAISS
        k = 5
        distances, indices = index.search(q_vec, k)

        print(f"\n🔍 Top {k} résultats pour : {query!r}")
        print(f"   indices bruts  : {indices[0].tolist()}")
        print(f"   distances brutes : {distances[0].tolist()}")

        shown = 0

        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0 or idx >= len(ids_list):
                print(f"   ⚠️ Index FAISS {idx} hors plage (0..{len(ids_list)-1})")
                continue

            chunk_id = ids_list[idx]
            rec = records_by_id.get(chunk_id)
            if not rec:
                print(f"   ⚠️ chunk_id {chunk_id} introuvable dans corpus.")
                continue

            score = 1.0 / (1.0 + float(dist))
            pretty_print_result(rank, score, rec)
            shown += 1

        if shown == 0:
            print("   ⚠️ Aucun résultat affichable (problème de mapping index ↔ corpus ?)")


if __name__ == "__main__":
    main()
