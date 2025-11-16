import json
import re
from pathlib import Path

# Stopwords et termes trop génériques qu'on ne veut PAS comme entités
STOPWORDS_COMMON = {
    "nord",
    "sud",
    "dit",
    "dei",
    "domini",
    "conventus",
    "comte",
    "duc",
    "roi",
    "abbé",
    "abbas",
    "episcopus",
    "dominus",
}


def clean_entries(entries, min_len=3, drop_if_numeric=True):
    cleaned = []

    for raw in entries:
        if raw is None:
            continue

        s = str(raw).strip()

        # enlève la pollution en début/fin
        s = re.sub(r'^[\s;,\-]+', '', s)
        s = re.sub(r'[\s;,\-]+$', '', s)
        s = s.strip()
        if not s:
            continue

        # enlève parenthèses fermantes
        s = s.replace(')', '').strip()

        # normalise "d'Auvergne" / "d’Auvergne" -> "Auvergne"
        s = re.sub(r"^d['’]\s*", "", s, flags=re.IGNORECASE).strip()
        if not s:
            continue

        # supprime les entrées purement numériques
        if drop_if_numeric and re.fullmatch(r'\d+', s):
            continue

        # filtre les entrées trop courtes
        if len(s) < min_len:
            continue

        # stopwords génériques (en minuscules ou capitalisés)
        if s.lower() in STOPWORDS_COMMON:
            continue

        cleaned.append(s)

    # déduplication
    uniq = []
    for s in cleaned:
        if s not in uniq:
            uniq.append(s)

    return uniq


def main():
    base = Path("mes_documents_historiques")
    src = base / "entity_lexicon_v2.json"
    dst = base / "entity_lexicon_v2_clean.json"

    if not src.exists():
        print(f"❌ Fichier source introuvable : {src}")
        return

    print(f"📂 Chargement du lexique brut : {src}")
    data = json.load(src.open("r", encoding="utf-8"))

    raw_persons = data.get("persons", []) or []
    raw_places = data.get("places", []) or []
    raw_years = data.get("years", []) or []

    persons_clean = clean_entries(raw_persons, min_len=3, drop_if_numeric=True)
    places_clean = clean_entries(raw_places, min_len=3, drop_if_numeric=True)

    # les années on les garde telles quelles, en les normalisant juste en strings
    years_clean = [str(y).strip() for y in raw_years if str(y).strip()]

    cleaned = {
        "persons": persons_clean,
        "places": places_clean,
        "years": years_clean,
    }

    print(f"✅ Persons : {len(raw_persons)} -> {len(persons_clean)}")
    print(f"✅ Places  : {len(raw_places)} -> {len(places_clean)}")
    print(f"✅ Years   : {len(raw_years)} -> {len(years_clean)}")

    dst.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Lexique nettoyé écrit dans : {dst}")


if __name__ == "__main__":
    main()
