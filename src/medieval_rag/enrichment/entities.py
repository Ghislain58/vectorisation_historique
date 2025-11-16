from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re


def _clean_lexicon_entries(entries: List[str],
                           min_len: int = 3,
                           drop_if_numeric: bool = True) -> List[str]:
    """
    Nettoie une liste brute venant du JSON :
      - strip espaces, ;, - en début/fin
      - supprime les entrées vides
      - optionnel : supprime les entrées purement numériques (ex: "1199)")
      - filtre les entrées trop courtes (min_len)
    """
    cleaned: List[str] = []

    for raw in entries:
        if raw is None:
            continue

        s = str(raw).strip()

        # enlève ponctuation/pollution typique en début/fin
        s = re.sub(r'^[\s;,\-]+', '', s)
        s = re.sub(r'[\s;,\-]+$', '', s)
        s = s.strip()

        if not s:
            continue

        # supprime les ")"
        s = s.replace(')', '').strip()

        # si c'est purement numérique → on jette (ça ira dans "years", pas ici)
        if drop_if_numeric and re.fullmatch(r'\d+', s):
            continue

        # filtre les trucs trop courts
        if len(s) < min_len:
            continue

        cleaned.append(s)

    return cleaned


def load_entity_lexicon(base_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Charge entity_lexicon_v2.json si présent, sinon entity_lexicon.json.
    Applique un nettoyage simple sur persons / places pour réduire le bruit.
    Retourne un dict avec :
      - persons: liste de labels propres
      - places : liste de labels propres
      - years  : liste de chaînes pour les années
    """
    if base_path is None:
        base_path = Path(".")

    lex_dir = base_path / "mes_documents_historiques"
    lex_path_v2 = lex_dir / "entity_lexicon_v2.json"
    lex_path_v1 = lex_dir / "entity_lexicon.json"

    persons: List[str] = []
    places: List[str] = []
    years: List[str] = []

    path_used = None

    if lex_path_v2.exists():
        path_used = lex_path_v2
    elif lex_path_v1.exists():
        path_used = lex_path_v1

    if path_used is None:
        print("⚠️  Aucun lexique d'entités trouvé, enrichissement désactivé.")
        return {"persons": [], "places": [], "years": []}

    try:
        with path_used.open("r", encoding="utf-8") as f:
            data = json.load(f)

        raw_persons = data.get("persons", []) or []
        raw_places = data.get("places", []) or []
        raw_years = data.get("years", []) or []

        # Nettoyage persons / places
        persons = _clean_lexicon_entries(raw_persons, min_len=3, drop_if_numeric=True)
        places = _clean_lexicon_entries(raw_places, min_len=3, drop_if_numeric=True)

        # Années : juste normalisation en string
        years = [str(y).strip() for y in raw_years if str(y).strip()]

        print(
            f"🧾 Lexique d'entités chargé ({path_used.name}) après nettoyage : "
            f"{len(persons)} persons, {len(places)} places, {len(years)} years."
        )

    except Exception as e:
        print(f"⚠️  Impossible de charger le lexique d'entités : {e}")
        persons, places, years = [], [], []

    return {
        "persons": persons,
        "places": places,
        "years": years,
    }


def _compile_lexicon_patterns(
    lexicon: Dict[str, List[str]]
) -> Dict[str, List[Tuple[str, re.Pattern]]]:
    """
    Prépare des couples (label, regex) pour chaque entrée du lexique.
    On renvoie par ex. pour 'Bonitus' :
      ("Bonitus", re.compile(r"\bBonitus\b", re.I))
    Afin de stocker le label lisible dans le JSONL.
    """
    compiled: Dict[str, List[Tuple[str, re.Pattern]]] = {
        "persons": [],
        "places": [],
        "years": [],
    }

    for key in ("persons", "places", "years"):
        entries = lexicon.get(key) or []
        for entry in entries:
            label = str(entry).strip()
            if not label:
                continue
            # échappe les caractères spéciaux
            pattern = r"\b" + re.escape(label) + r"\b"
            compiled[key].append((label, re.compile(pattern, flags=re.IGNORECASE)))

    return compiled


def extract_entities_from_text(
    text: str,
    lexicon_patterns: Dict[str, List[Tuple[str, re.Pattern]]]
) -> Dict[str, List[str]]:
    """
    Retourne les entités trouvées dans le texte sous forme de labels lisibles, sans regex.
    {
      "persons": ["Géraud d'Aurillac", ...],
      "places": ["Brioude", ...],
      "years": ["999", "1010", ...]
    }
    """
    found_persons: List[str] = []
    found_places: List[str] = []
    found_years: List[str] = []

    for label, pat in lexicon_patterns.get("persons", []):
        if pat.search(text):
            found_persons.append(label)

    for label, pat in lexicon_patterns.get("places", []):
        if pat.search(text):
            found_places.append(label)

    for label, pat in lexicon_patterns.get("years", []):
        if pat.search(text):
            found_years.append(label)

    def _dedup(values: List[str]) -> List[str]:
        uniq: List[str] = []
        for v in values:
            if v not in uniq:
                uniq.append(v)
        return uniq

    return {
        "persons": _dedup(found_persons),
        "places": _dedup(found_places),
        "years": _dedup(found_years),
    }


def enrich_chunk_record_with_entities(
    record: Dict,
    lexicon_patterns: Dict[str, List[Tuple[str, re.Pattern]]]
) -> Dict:
    """
    Prend un record de chunk (JSONL) et ajoute/actualise :
      - "entities" (persons, places, years)
      - "year_min"
      - "year_max"
    """
    text = record.get("text", "") or ""
    entities = extract_entities_from_text(text, lexicon_patterns)

    years_int: List[int] = []
    for y in entities.get("years", []):
        try:
            years_int.append(int(y))
        except ValueError:
            continue

    year_min = min(years_int) if years_int else None
    year_max = max(years_int) if years_int else None

    record["entities"] = entities
    record["year_min"] = year_min
    record["year_max"] = year_max

    return record
