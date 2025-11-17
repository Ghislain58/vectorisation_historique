#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyse_corpus.py
-----------------
Outil d'inspection du corpus indexé dans Chroma.

Il se connecte à la collection "documents_historiques" créée par
historical_vectorizer.py et affiche :

- le nombre total de chunks ;
- la répartition par type de document :
    - cartulaire (is_cartulary)
    - charte (is_charter)
    - académique (is_academic)
    - autres
- une approximation du "dossier logique" à partir du chemin :
    mes_documents_historiques/articles, cartulaires, theses, etc.
- la liste des institutions détectées pour les cartulaires ;
- la répartition par langue dominante (dominant_lang).
"""

from collections import Counter
from pathlib import PurePosixPath

from chromadb import PersistentClient


def fetch_all_metadatas(collection, batch_size: int = 1000):
    """
    Récupère toutes les métadonnées de la collection par paquets.
    On ne charge PAS les documents ni les embeddings, uniquement metadatas.
    """
    all_metas = []
    offset = 0

    while True:
        res = collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        metas = res.get("metadatas", [])
        if not metas:
            break

        all_metas.extend(metas)
        offset += len(metas)

    return all_metas


def guess_folder_from_source(source: str) -> str:
    """
    Essaie de déduire le “dossier logique” à partir du chemin complet.

    Exemple :
        /home/.../mes_documents_historiques/cartulaires/MonPDF.pdf
        -> folder = "cartulaires"

    Si rien n'est trouvé, renvoie "UNKNOWN".
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


def main():
    print("🔗 Connexion à la base Chroma…")
    client = PersistentClient(path="./vector_db_historique")
    col = client.get_or_create_collection("documents_historiques")

    print("📥 Récupération des métadonnées (pagination)…")
    metadatas = fetch_all_metadatas(col, batch_size=2000)
    total = len(metadatas)
    print(f"✅ {total} chunks trouvés dans la collection.\n")

    by_type = Counter()
    by_folder = Counter()
    by_institution = Counter()
    by_lang = Counter()

    for m in metadatas:
        # Type de document
        if m.get("is_cartulary"):
            by_type["cartulary"] += 1
        elif m.get("is_charter"):
            by_type["charter"] += 1
        elif m.get("is_academic"):
            by_type["academic"] += 1
        else:
            by_type["other"] += 1

        # Dossier logique (déduit de "source")
        source = m.get("source", "") or ""
        folder = guess_folder_from_source(source)
        by_folder[folder] += 1

        # Institutions (cartulaires)
        inst = m.get("institution")
        if inst:
            by_institution[inst] += 1

        # Langue dominante
        dom = m.get("dominant_lang")
        if dom:
            by_lang[dom] += 1

    # === Affichages ===
    print("📊 Chunks par type de document :")
    if not by_type:
        print("  (aucun type détecté)")
    else:
        for t, c in by_type.most_common():
            print(f"  - {t:10s} : {c}")
    print()

    print("📂 Chunks par dossier logique (mes_documents_historiques/…):")
    if not by_folder:
        print("  (aucun chemin exploitable)")
    else:
        for f, c in by_folder.most_common():
            print(f"  - {f:20s} : {c}")
    print()

    print("🏛️ Institutions détectées (cartulaires) :")
    if not by_institution:
        print("  (aucune institution renseignée)")
    else:
        for inst, c in by_institution.most_common():
            print(f"  - {inst:25s} : {c}")
    print()

    print("🌐 Langue dominante :")
    if not by_lang:
        print("  (aucune info de langue dominante)")
    else:
        for lang, c in by_lang.most_common():
            print(f"  - {lang:10s} : {c}")
    print()

    print("✅ Analyse simple terminée.")


if __name__ == "__main__":
    main()
