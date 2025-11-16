#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyse_themes.py
-----------------

Analyse thématique massive sur le corpus indexé dans Chroma.

- On fournit une liste de termes (latin, français, etc.).
- Le script parcourt tous les chunks de la collection "documents_historiques".
- Il compte les occurrences de chaque terme :
    - total global
    - par type de document (cartulaire / charte / académique / autres)
    - par "dossier logique" (articles, cartulaires, theses, ...)

- Il affiche aussi quelques exemples de chunks où chaque terme apparaît.

Usage :

    python analyse_themes.py
        -> utilise une liste de termes par défaut (villa, mansus, ecclesia...)

    python analyse_themes.py --terms "villa,mansus,feodum,ecclesia"
        -> analyse ces termes-là

    python analyse_themes.py --terms "decima,terra castrum"
        -> gère aussi des expressions avec espaces (simple recherche textuelle)

"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Dict, List, Tuple

from chromadb import PersistentClient


def guess_folder_from_source(source: str) -> str:
    """
    Essaie de déduire le “dossier logique” à partir du chemin complet.

    /home/.../mes_documents_historiques/cartulaires/MonPDF.pdf
    -> "cartulaires"
    """
    try:
        p = PurePosixPath(source)
    except Exception:
        return "UNKNOWN"

    parts = list(p.parts)
    if "mes_documents_historiques" in parts:
        idx = parts.index("mes_documents_historiques")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "UNKNOWN"


def iter_batches(collection, batch_size: int = 500):
    """
    Itère sur la collection par paquets (documents + metadatas),
    pour éviter de tout charger en mémoire d'un coup.
    """
    offset = 0
    while True:
        res = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])

        if not docs:
            break

        # docs et metas sont des listes parallèles
        for doc, meta in zip(docs, metas):
            yield doc, meta

        offset += len(docs)


def compile_patterns(terms: List[str]) -> Dict[str, re.Pattern]:
    """
    Prépare des regex insensibles à la casse pour chaque terme.
    On utilise une recherche simple sur la chaîne, sans gestion
    fine de lemmatisation, parce qu'on travaille sur du brut.
    """
    patterns = {}
    for term in terms:
        t = term.strip()
        if not t:
            continue
        # On échappe le terme pour éviter les surprises regex
        pat = re.compile(re.escape(t), re.IGNORECASE)
        patterns[term] = pat
    return patterns


def highlight_snippet(text: str, term: str, width: int = 200) -> str:
    """
    Renvoie un petit extrait du chunk où le terme apparaît.
    On coupe autour de la première occurrence.
    """
    if not text:
        return ""
    lower = text.lower()
    idx = lower.find(term.lower())
    if idx == -1:
        # pas trouvé, on coupe juste le début
        return (text[:width] + "…") if len(text) > width else text
    start = max(0, idx - width // 2)
    end = min(len(text), idx + width // 2)
    snippet = text[start:end]
    return snippet.replace("\n", " ").strip() + ("…" if end < len(text) else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--terms",
        type=str,
        default="villa,mansus,feodum,ecclesia,decima,castrum,pagus",
        help="Liste de termes séparés par des virgules (latin ou autre)",
    )
    args = parser.parse_args()

    terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    if not terms:
        print("⚠️  Aucun terme valide fourni.")
        return

    print("🔗 Connexion à la base Chroma…")
    client = PersistentClient(path="./vector_db_historique")
    col = client.get_or_create_collection("documents_historiques")

    print(f"🔍 Analyse des thèmes : {', '.join(terms)}")
    patterns = compile_patterns(terms)

    # Compteurs globaux
    total_chunks = 0
    counts_global = Counter()
    counts_by_type: Dict[str, Counter] = defaultdict(Counter)
    counts_by_folder: Dict[str, Counter] = defaultdict(Counter)

    # On garde quelques exemples (top chunks) pour chaque terme
    examples: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    # format : (nombre_occurrences_dans_chunk, fichier, extrait)

    print("📥 Parcours de tous les chunks…")
    for doc, meta in iter_batches(col, batch_size=500):
        total_chunks += 1
        if not doc:
            continue

        text = doc
        source = meta.get("source", "") or ""
        filename = meta.get("filename", "") or source or "inconnu"

        folder = guess_folder_from_source(source)

        if meta.get("is_cartulary"):
            doc_type = "cartulary"
        elif meta.get("is_charter"):
            doc_type = "charter"
        elif meta.get("is_academic"):
            doc_type = "academic"
        else:
            doc_type = "other"

        # Comptage pour chaque terme
        for term, pat in patterns.items():
            matches = pat.findall(text)
            n = len(matches)
            if n <= 0:
                continue

            # Mise à jour des compteurs
            counts_global[term] += n
            counts_by_type[term][doc_type] += n
            counts_by_folder[term][folder] += n

            # On stocke quelques exemples (max ~5 par terme)
            if len(examples[term]) < 5:
                snippet = highlight_snippet(text, term)
                examples[term].append((n, filename, snippet))

    print(f"\n✅ Analyse terminée sur {total_chunks} chunks.\n")

    # === Résumé par terme ===
    for term in terms:
        print("=" * 70)
        print(f"🧾 Terme : {term}")
        total = counts_global.get(term, 0)
        print(f"  ➜ Occurrences totales : {total}")

        # Par type de document
        ctype = counts_by_type.get(term, {})
        if ctype:
            print("  📂 Par type de document :")
            for t, c in ctype.most_common():
                print(f"    - {t:10s} : {c}")
        else:
            print("  📂 Par type de document : (aucune occurrence)")

        # Par dossier logique
        cfold = counts_by_folder.get(term, {})
        if cfold:
            print("  🗂️  Par dossier (mes_documents_historiques/…):")
            for f, c in cfold.most_common():
                print(f"    - {f:15s} : {c}")
        else:
            print("  🗂️  Par dossier : (aucune occurrence)")

        # Exemples
        ex_list = examples.get(term, [])
        if ex_list:
            print("  🔍 Exemples de chunks où le terme apparaît :")
            for n, fname, snippet in ex_list:
                print(f"    • [{n}×] {fname}")
                print(f"      « {snippet} »")
        else:
            print("  🔍 Aucun exemple stocké (aucune occurrence trouvée).")

        print()

    print("🏁 Fin de l'analyse thématique.")


if __name__ == "__main__":
    main()
