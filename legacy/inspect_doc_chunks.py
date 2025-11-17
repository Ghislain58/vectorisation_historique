#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_doc_chunks.py
---------------------

Affiche quelques chunks d'un document donné (par nom de fichier)
pour vérifier manuellement la qualité du découpage.
"""

from chromadb import PersistentClient
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filename_substr", type=str,
                    help="Fragment du nom de fichier (ex: 'BERGER_JEAN' ou 'LAuvergne_et_ses_marges')")
    ap.add_argument("--limit", type=int, default=5,
                    help="Nombre de chunks à afficher (par défaut 5)")
    args = ap.parse_args()

    client = PersistentClient(path="./vector_db_historique")
    col = client.get_or_create_collection("documents_historiques")

    # On récupère tout, puis on filtre par filename
    res = col.get(include=["documents", "metadatas"], limit=1000000)
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    target = args.filename_substr.lower()
    selected = []

    for doc, m in zip(docs, metas):
        fname = (m.get("filename") or "").lower()
        if target in fname:
            selected.append((fname, m.get("chunk_id"), doc))

    if not selected:
        print(f"❌ Aucun chunk trouvé pour un fichier contenant : {args.filename_substr}")
        return

    # Tri par chunk_id
    selected.sort(key=lambda x: (x[0], x[1] if x[1] is not None else 0))
    print(f"✅ {len(selected)} chunks trouvés pour ce fichier. Affichage des {args.limit} premiers:\n")

    for fname, cid, doc in selected[:args.limit]:
        print("=" * 70)
        print(f"📄 {fname}  |  chunk #{cid}")
        print("-" * 70)
        print(doc.strip())
        print()

if __name__ == "__main__":
    main()
