# historical_query.py
# Moteur de requêtes sur la base vectorielle historique

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


class HistoricalQueryEngine:
    """
    Moteur de requêtes :
    - encode la question avec intfloat/multilingual-e5-large
    - interroge la collection Chroma "documents_historiques_v2"
    - applique ensuite des filtres souples sur les métadonnées :
        * --person → entities_persons
        * --place  → entities_places
        * --year   → entities_years
    """

    def __init__(
        self,
        db_path: Path,
        collection_name: str = "documents_historiques_v2",
    ) -> None:
        # Device
        if torch.cuda.is_available():
            self.device = "cuda"
            print("✅ GPU détecté :", torch.cuda.get_device_name(0))
            print(
                "📥 Chargement du modèle : intfloat/multilingual-e5-large"
            )
        else:
            self.device = "cpu"
            print("⚠️  GPU non détecté, utilisation du CPU.")
            print(
                "📥 Chargement du modèle : intfloat/multilingual-e5-large"
            )

        # Modèle d'embedding
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            device=self.device,
        )

        # Client Chroma
        self.client = PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(collection_name)

    # ============================
    #   OUTILS SUR LES MÉTADONNÉES
    # ============================

    def _parse_meta_list(self, value: Any) -> List[str]:
        """
        Normalise une métadonnée en liste de strings :
        - liste → liste (avec str() sur les éléments)
        - string JSON de liste → liste
        - autre → [str(value)]
        - None → []
        """
        if value is None:
            return []

        if isinstance(value, list):
            return [str(x) for x in value]

        if isinstance(value, str):
            # Tentative de décodage JSON
            try:
                data = json.loads(value)
                if isinstance(data, list):
                    return [str(x) for x in data]
                else:
                    return [value]
            except Exception:
                return [value]

        return [str(value)]

    def _matches_filters(
        self,
        md: Dict[str, Any],
        person: Optional[str] = None,
        place: Optional[str] = None,
        year: Optional[str] = None,
    ) -> bool:
        """
        Applique les filtres de manière souple sur les métadonnées d'un chunk.
        - person : substring insensible à la casse dans entities_persons
        - place  : substring insensible à la casse dans entities_places
        - year   : substring dans entities_years
        Tous les filtres fournis sont en AND.
        """

        def match_one(key: str, needle: Optional[str]) -> bool:
            if not needle:
                return True  # pas de filtre sur ce champ
            values = self._parse_meta_list(md.get(key))
            if not values:
                return False
            n = str(needle).lower()
            for item in values:
                if isinstance(item, str) and n in item.lower():
                    return True
            return False

        ok_person = match_one("entities_persons", person)
        ok_place = match_one("entities_places", place)
        ok_year = match_one("entities_years", year)

        return ok_person and ok_place and ok_year

    # ============================
    #   REQUÊTE PRINCIPALE
    # ============================

    def ask(
        self,
        query: str,
        person: Optional[str] = None,
        place: Optional[str] = None,
        year: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Interroge la base vectorielle :
        1) Chroma : recherche sémantique large (sans where sur les entités)
        2) Filtrage Python sur les métadonnées (person/place/year)
        3) Retourne les top_k meilleurs résultats filtrés
        """
        # 1) Embedding de la requête
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
        )[0]

        # 2) Requête Chroma SANS filtre d'entités
        raw = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=max(top_k, 100),
            include=["documents", "metadatas", "distances"],
        )

        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        # Sécurité si Chroma renvoie moins d'entrées que prévu
        n = min(len(docs), len(metas), len(dists))
        docs = docs[:n]
        metas = metas[:n]
        dists = dists[:n]

        # 3) Filtrage Python sur entités
        results: List[Dict[str, Any]] = []
        for doc, md, dist in zip(docs, metas, dists):
            if self._matches_filters(md, person=person, place=place, year=year):
                results.append(
                    {
                        "document": doc,
                        "metadata": md,
                        "distance": float(dist),
                    }
                )

        # Tri par distance (au cas où)
        results.sort(key=lambda r: r["distance"])
        return results[:top_k]

    # ============================
    #   AFFICHAGE
    # ============================

    def _format_entity_list(self, values: Any) -> str:
        lst = self._parse_meta_list(values)
        if not lst:
            return "—"
        # on limite l'affichage pour éviter le bruit monstrueux
        if len(lst) > 10:
            return ", ".join(lst[:10]) + " ..."
        return ", ".join(lst)

    def pretty_print_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        person: Optional[str],
        place: Optional[str],
        year: Optional[str],
    ) -> None:
        print()
        print(f"🔎 Requête : {query}")
        if person:
            print(f"👤 Filtre personne : {person}")
        if place:
            print(f"📍 Filtre lieu     : {place}")
        if year:
            print(f"🗓  Filtre année   : {year}")
        print()

        if not results:
            print("❌ Aucun résultat trouvé (après filtrage).")
            return

        for idx, res in enumerate(results, 1):
            md = res["metadata"]
            doc = res["document"]
            dist = res["distance"]

            filename = md.get("filename", "—")
            folder = md.get("folder", "—")
            chunk_id = md.get("chunk_id", "—")

            print(f"-------------------- [{idx}] --------------------")
            print(
                f"📄 {filename}  (dossier: {folder}, chunk: {chunk_id}, distance: {dist:.4f})"
            )
            print(f"📂 Source : {md.get('source', '—')}")
            print(f"📝 Extrait : {doc[:400].replace('\n', ' ')}[...]")
            print()

            persons = self._format_entity_list(md.get("entities_persons"))
            places = self._format_entity_list(md.get("entities_places"))
            years = self._format_entity_list(md.get("entities_years"))

            if persons != "—":
                print(f"👥 Personnes: {persons}")
            if places != "—":
                print(f"📍 Lieux    : {places}")
            if years != "—":
                print(f"🗓  Années   : {years}")

            print()

        print("✅ Terminé.")


# ============================
#   CLI
# ============================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Moteur de requêtes sur la base vectorielle historique."
    )
    parser.add_argument(
        "query",
        type=str,
        help="Question ou requête textuelle (ex: 'organisation du territoire autour de Brioude au XIe siècle')",
    )
    parser.add_argument(
        "--person",
        type=str,
        default=None,
        help="Filtre personne (ex: 'Géraud', 'Montmorin')",
    )
    parser.add_argument(
        "--place",
        type=str,
        default=None,
        help="Filtre lieu (ex: 'Brioude')",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Filtre année (ex: '1050' ou 'XIe siècle')",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Nombre maximal de résultats à afficher (défaut: 10)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    engine = HistoricalQueryEngine(
        db_path=Path("./vector_db_historique"),
        collection_name="documents_historiques_v2",
    )

    results = engine.ask(
        query=args.query,
        person=args.person,
        place=args.place,
        year=args.year,
        top_k=args.top_k,
    )

    engine.pretty_print_results(
        query=args.query,
        results=results,
        person=args.person,
        place=args.place,
        year=args.year,
    )


if __name__ == "__main__":
    main()
