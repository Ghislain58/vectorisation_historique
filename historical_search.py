import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def build_filters(doc_lang: Optional[str], doc_type: Optional[str]) -> Dict[str, Any]:
    where: Dict[str, Any] = {}
    if doc_lang:
        where["doc_lang"] = doc_lang

    if doc_type:
        if doc_type == "academic":
            where["is_academic"] = True
        elif doc_type == "charter":
            where["is_charter"] = True
        elif doc_type == "cartulary":
            where["is_cartulary"] = True
        elif doc_type == "autres":
            where["is_entity_index"] = True

    return where


def tokenize(text: str) -> List[str]:
    """Tokenisation simple pour BM25."""
    import re

    return re.findall(r"\w+", text.lower())


def min_max_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < 1e-9:
        return np.ones_like(x) * 0.5
    return (x - xmin) / (xmax - xmin)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recherche hybride (E5 + BM25) dans vector_db_historique."
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Texte de la requête (FR, latin, etc.).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Nombre de résultats à retourner après re-ranking.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Filtre sur doc_lang (ex: FR, EN, DE).",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["academic", "charter", "cartulary", "autres"],
        default=None,
        help="Filtre sur le type de document.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=200,
        help="Nombre de candidats récupérés par embeddings avant re-ranking BM25.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Poids de BM25 dans le score final (0 = embeddings seuls, 1 = BM25 seul).",
    )

    args = parser.parse_args()

    base_path = Path(".").resolve()
    db_path = base_path / "vector_db_historique"

    print("🔗 Connexion à ChromaDB…")
    client = PersistentClient(path=str(db_path))
    collection = client.get_collection("documents_historiques")

    print("📥 Chargement du modèle de requête : intfloat/multilingual-e5-large")
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    query_text = args.query.strip()
    if not query_text:
        print("❌ Requête vide.")
        return

    # E5 : préfixe spécial pour les requêtes
    model_input = f"query: {query_text}"
    query_emb = model.encode([model_input], convert_to_numpy=True)

    where = build_filters(args.lang, args.type)
    if where:
        print(f"🔎 Filtres appliqués : {where}")

    n_candidates = max(args.k, args.candidates)
    print(f"🔎 Récupération de {n_candidates} candidats par similarité d'embedding…")

    res = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=n_candidates,
        where=where if where else None,
    )

    ids: List[str] = res.get("ids", [[]])[0]
    docs: List[str] = res.get("documents", [[]])[0]
    metadatas: List[Dict[str, Any]] = res.get("metadatas", [[]])[0]
    distances: List[float] = res.get("distances", [[]])[0]

    if not ids:
        print("⚠️ Aucun résultat trouvé.")
        return

    # ============================
    #  BM25 sur les candidats
    # ============================
    print("📊 Re-ranking BM25 sur les candidats…")

    tokenized_docs = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = tokenize(query_text)
    bm25_scores = bm25.get_scores(tokenized_query)  # shape (n_candidates,)

    # Embeddings → distance (plus petit = mieux) → similarité
    dist_arr = np.array(distances, dtype=np.float32)
    sim_emb = 1.0 / (1.0 + dist_arr)  # transformation monotone

    bm25_arr = np.array(bm25_scores, dtype=np.float32)

    bm25_norm = min_max_norm(bm25_arr)
    emb_norm = min_max_norm(sim_emb)

    alpha = float(args.alpha)
    alpha = max(0.0, min(1.0, alpha))

    final_scores = alpha * bm25_norm + (1.0 - alpha) * emb_norm

    # Tri décroissant par score final
    order = np.argsort(-final_scores)
    top_k = min(args.k, len(order))
    print(
        f"✅ Re-ranking terminé. Affichage des {top_k} meilleurs résultats "
        f"(alpha={alpha:.2f}, candidats={n_candidates})."
    )

    for rank_idx in range(top_k):
        idx = int(order[rank_idx])
        id_ = ids[idx]
        doc = docs[idx]
        md = metadatas[idx]
        dist = distances[idx]
        score_bm25 = bm25_arr[idx]
        score_final = final_scores[idx]

        print("\n" + "=" * 80)
        print(f"🔹 Résultat #{rank_idx + 1}")
        print(f"ID        : {id_}")
        print(f"Score emb : {dist:.4f} (distance, plus petit = mieux)")
        print(f"Score BM25: {score_bm25:.4f}")
        print(f"Score mix : {score_final:.4f}")
        print(f"Fichier   : {md.get('filename')}")
        print(f"Dossier   : {md.get('folder')}")
        print(f"Langue    : {md.get('doc_lang')}")

        doc_type = (
            "charter" if md.get("is_charter") else
            "cartulary" if md.get("is_cartulary") else
            "academic" if md.get("is_academic") else
            "autres" if md.get("is_entity_index") else
            "other"
        )
        print(f"Type      : {doc_type}")

        ents_lex = (
            md.get("entities_persons"),
            md.get("entities_places"),
            md.get("entities_years"),
        )
        if any(ents_lex):
            print(f"Entités (lexique) : {ents_lex}")

        ents_spacy = (
            md.get("spacy_persons"),
            md.get("spacy_places"),
        )
        if any(ents_spacy):
            print(f"Entités spaCy FR  : {ents_spacy}")

        snippet = doc[:600].replace("\n", " ")
        print(f"Texte     : {snippet}...")
    print("\n✅ Recherche hybride terminée.")


if __name__ == "__main__":
    main()
