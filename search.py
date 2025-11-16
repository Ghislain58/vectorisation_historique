#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recherche sémantique locale sur la base Chroma créée par historical_vectorizer.py

- Modèle : intfloat/multilingual-e5-large
- Base   : ./vector_db_historique
- Collection : documents_historiques

Filtres disponibles :
  --k N                  : nombre de résultats à afficher (par défaut 5)
  --folder NOM           : filtrage a posteriori sur le chemin (cartulaires, articles, theses, etc.)
  --cartulary-only       : limite aux chunks avec is_cartulary = True
  --institution NOM      : filtre sur metadata "institution"
  --lang {latin,french,mixed}
                         : filtre sur la langue dominante/mixte

"""

import argparse
import re
import sys
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient


def highlight(text: str, query: str, width: int = 280) -> str:
    """Mise en évidence simple des mots de la requête (case-insensitive)."""
    terms = [re.escape(t) for t in query.split() if t.strip()]
    snippet = (text[:width] + "…") if len(text) > width else text
    if not terms:
        return snippet
    pat = re.compile(r"(" + "|".join(terms) + r")", re.IGNORECASE)
    return pat.sub(lambda m: f"\x1b[1m{m.group(1)}\x1b[0m", snippet)


def build_where(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Construit un filtre Chroma basé UNIQUEMENT sur des opérateurs supportés ($eq / $and)."""
    clauses: List[Dict[str, Any]] = []

    if args.cartulary_only:
        clauses.append({"is_cartulary": {"$eq": True}})

    if args.institution:
        clauses.append({"institution": {"$eq": args.institution}})

    if args.lang:
        if args.lang == "mixed":
            clauses.append({"is_mixed": {"$eq": True}})
        else:
            clauses.append({"dominant_lang": {"$eq": args.lang}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="Nombre de résultats à afficher")
    ap.add_argument("--folder", type=str, default="", help="Filtre sur un sous-dossier logique (articles, cartulaires, theses...)")
    ap.add_argument("--cartulary-only", action="store_true", help="Limiter la recherche aux cartulaires (is_cartulary=True)")
    ap.add_argument("--institution", type=str, default="", help="Filtrer sur la métadonnée 'institution'")
    ap.add_argument("--lang", type=str, choices=["latin", "french", "mixed"], default="",
                    help="Filtre sur la langue dominante ou mixte")
    args = ap.parse_args()

    print("🔎 Initialisation du moteur (modèle + base)…")
    model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
    client = PersistentClient(path="./vector_db_historique")
    col = client.get_or_create_collection("documents_historiques")

    # Boucle interactive
    while True:
        try:
            query = input("\n🧠 Entrez votre requête (ou 'exit' pour quitter) : ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        q_emb = model.encode([query], convert_to_numpy=True)[0]
        where = build_where(args)

        # On demande plus de résultats que k si on a un filtre folder (post-filtrage)
        raw_n_results = args.k
        if args.folder:
            raw_n_results = max(args.k * 5, 50)

        res = col.query(
            query_embeddings=[q_emb.tolist()],
            n_results=raw_n_results,
            where=where,
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        if not docs:
            print("\n⚠️  Aucun résultat trouvé (base vide ou filtres trop restrictifs).")
            continue

        # Post-filtrage sur le dossier logique, basé sur le chemin "source"
        rows = []
        for doc, meta, dist in zip(docs, metas, dists):
            path = meta.get("source", "") or ""
            if args.folder:
                pattern = f"/mes_documents_historiques/{args.folder}/"
                if pattern not in path:
                    continue
            rows.append((doc, meta, dist))
            if len(rows) >= args.k:
                break

        if not rows:
            print("\n⚠️  Aucun résultat trouvé avec ces filtres (après filtrage folder).")
            continue

        print("\n===== RÉSULTATS =====")
        for i, (doc, meta, dist) in enumerate(rows, 1):
            path = meta.get("source", "inconnu")
            filename = meta.get("filename", "inconnu")

            try:
                score = 1.0 / (1.0 + float(dist))
                score_txt = f"{score:.3f}"
            except Exception:
                score_txt = "N/A"

            print(f"\n[{i}] ---------")
            print(f"📄 Fichier : {filename}")
            print(f"📍 Path    : {path}")
            print(f"📊 Score   : {score_txt}")

            flags = []
            if meta.get("is_cartulary"):
                flags.append("cartulaire")
            if meta.get("is_charter"):
                flags.append("charte")
            if meta.get("is_academic"):
                flags.append("académique")
            if flags:
                print(f"🏷️  Type   : {', '.join(flags)}")

            dom = meta.get("dominant_lang")
            if dom:
                print(f"🌐 Langue : {dom} (latin_ratio={meta.get('latin_ratio')}, french_ratio={meta.get('french_ratio')})")

            if meta.get("has_footnotes"):
                print(f"📝 Notes  : {meta.get('footnote_count', 0)} note(s) dans ce chunk")

            print(f"💬 Extrait : {highlight(doc, query)}")

        print()

    print("👋 Au revoir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

