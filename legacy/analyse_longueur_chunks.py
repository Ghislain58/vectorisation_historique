#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyse_longueur_chunks.py
--------------------------

Statistiques globales sur la longueur des chunks de la collection
"documents_historiques" (min, max, moyenne, percentiles).
"""

from chromadb import PersistentClient
import numpy as np

def main():
    client = PersistentClient(path="./vector_db_historique")
    col = client.get_or_create_collection("documents_historiques")

    print("📥 Récupération des chunks…")
    all_docs = []
    offset = 0
    batch = 1000

    while True:
        res = col.get(include=["documents"], limit=batch, offset=offset)
        docs = res.get("documents", [])
        if not docs:
            break
        all_docs.extend(docs)
        offset += len(docs)

    print(f"✅ {len(all_docs)} chunks chargés.")

    lengths = [len(d) for d in all_docs]
    arr = np.array(lengths)

    print("\n📊 Longueur des chunks (en caractères) :")
    print(f"  - min     : {arr.min()}")
    print(f"  - max     : {arr.max()}")
    print(f"  - moyenne : {arr.mean():.1f}")
    print(f"  - médiane : {np.median(arr):.1f}")
    print(f"  - p10     : {np.percentile(arr, 10):.1f}")
    print(f"  - p90     : {np.percentile(arr, 90):.1f}")

if __name__ == "__main__":
    main()
