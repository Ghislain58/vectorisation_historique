"""
clean_entity_lexicon.py

Nettoie et enrichit le lexique d'entités :
- découpe les entrées contenant plusieurs entités séparées par des virgules / points-virgules
- supprime le bruit (tokens trop courts, génériques, etc.)
- normalise les espaces
- re-filtre les années

Entrées possibles :
- mes_documents_historiques/entity_lexicon_v2.json
- à défaut : mes_documents_historiques/entity_lexicon.json

Sortie :
- mes_documents_historiques/entity_lexicon_v3.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


# -----------------------------
# Heuristiques de nettoyage
# -----------------------------


STOP_TOKENS = {
    "com", "cant", "cant.", "cantons", "cant.", "b.", "r.", "n°",
    "cf", "op", "cit", "vol", "t", "p", "ibid", "idem"
}

ROMAN_NUMERALS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"
}


def split_entry(entry: str) -> List[str]:
    """
    Découpe une entrée potentiellement multiplet (ex: 'Bo, Bou, I., Pro')
    en sous-entrées.
    """
    # On coupe sur virgule, point-virgule et /, mais on laisse les points internes.
    parts = re.split(r"[;,/]", entry)
    cleaned = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        cleaned.append(s)
    return cleaned


def normalize_entry(entry: str) -> str:
    """
    Normalise légèrement une entrée :
    - supprime les espaces multiples
    - conserve la casse d'origine (pour ne pas massacrer le latin)
    """
    entry = re.sub(r"\s+", " ", entry.strip())
    return entry


def looks_like_year(s: str) -> bool:
    """
    True si s ressemble à une année.
    """
    if not re.fullmatch(r"\d{3,4}", s):
        return False
    year = int(s)
    # Fourchette large mais réaliste pour ton corpus
    return 500 <= year <= 2100


def is_bad_token(tok: str) -> bool:
    """
    True si l'entrée est manifestement du bruit.
    """
    s = tok.strip()

    # Très court
    if len(s) <= 2:
        return True

    s_lower = s.lower()

    # Tokens connus comme bruit
    if s_lower in STOP_TOKENS:
        return True

    # Roman numerals (I, II, III…)
    if s_lower in ROMAN_NUMERALS:
        return True

    # Une seule lettre ou lettre + point (B, B., R., etc.)
    if re.fullmatch(r"[A-Za-z]\.?", s):
        return True

    # Mot tout en minuscules très court et banal
    if len(s) <= 4 and s.islower():
        return True

    return False


def clean_list(entries: List[str], treat_years: bool = False) -> List[str]:
    """
    Nettoie une liste brute d'entrées (persons / places / years).
    """
    out: List[str] = []

    # 1) On découpe les entrées multiplets
    expanded: List[str] = []
    for e in entries:
        if not isinstance(e, str):
            continue
        expanded.extend(split_entry(e))

    for raw in expanded:
        s = normalize_entry(raw)
        if not s:
            continue

        if treat_years:
            # Pour les années, on garde uniquement ce qui ressemble vraiment à une année
            if looks_like_year(s):
                out.append(s)
            continue

        # Pour persons / places : filtrage
        if is_bad_token(s):
            continue

        out.append(s)

    # Dédupe + tri
    # On garde l’ordre par valeur triée (pratique pour lecture humaine)
    out = sorted(set(out), key=lambda x: x.lower())
    return out


def load_lexicon(base_path: Path) -> Tuple[Dict[str, List[str]], Path]:
    """
    Charge entity_lexicon_v2.json ou entity_lexicon.json.
    Retourne (lexique, chemin_source).
    """
    lex_v2 = base_path / "mes_documents_historiques" / "entity_lexicon_v2.json"
    lex_v1 = base_path / "mes_documents_historiques" / "entity_lexicon.json"

    if lex_v2.exists():
        with lex_v2.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data, lex_v2

    if lex_v1.exists():
        with lex_v1.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data, lex_v1

    raise FileNotFoundError("Aucun lexique trouvé (entity_lexicon_v2.json ou entity_lexicon.json).")


def main() -> None:
    base_path = Path(".").resolve()
    lex_raw, source_path = load_lexicon(base_path)

    persons_raw = lex_raw.get("persons", [])
    places_raw = lex_raw.get("places", [])
    years_raw = lex_raw.get("years", [])

    print(f"📥 Lexique brut chargé depuis : {source_path}")
    print(f"   Persons : {len(persons_raw)} entrées")
    print(f"   Places  : {len(places_raw)} entrées")
    print(f"   Years   : {len(years_raw)} entrées")

    persons_clean = clean_list(persons_raw, treat_years=False)
    places_clean = clean_list(places_raw, treat_years=False)
    years_clean = clean_list(years_raw, treat_years=True)

    print("\n✅ Lexique nettoyé :")
    print(f"   Persons : {len(persons_clean)} entrées (après nettoyage)")
    print(f"   Places  : {len(places_clean)} entrées (après nettoyage)")
    print(f"   Years   : {len(years_clean)} entrées (après nettoyage)")

    # Aperçu
    print("\n🧪 Exemples persons :", persons_clean[:20])
    print("🧪 Exemples places  :", places_clean[:20])
    print("🧪 Exemples years   :", years_clean[:20])

    out_path = base_path / "mes_documents_historiques" / "entity_lexicon_v3.json"
    out_data = {
        "persons": persons_clean,
        "places": places_clean,
        "years": years_clean,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Lexique nettoyé sauvegardé dans : {out_path}")


if __name__ == "__main__":
    main()
