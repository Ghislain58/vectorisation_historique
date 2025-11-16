import json
from pathlib import Path

def main():
    jsonl_path = Path("data/chunks/standard/corpus_chunks.jsonl")

    if not jsonl_path.exists():
        print(f"❌ Fichier introuvable : {jsonl_path}")
        return

    print(f"🔍 Lecture des 10 premiers chunks enrichis : {jsonl_path}\n")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Erreur JSON ligne {i}: {e}")
                continue

            print(f"--- Chunk {i} ---")
            print(f"Doc     : {rec.get('doc_id')} ({rec.get('source')})")
            print(f"Pages   : {rec.get('page_start')}–{rec.get('page_end')}")
            print("Entities:")

            entities = rec.get("entities") or {}

            print(f"  Persons : {entities.get('persons')}")
            print(f"  Places  : {entities.get('places')}")
            print(f"  Years   : {entities.get('years')}")

            print(f"  year_min : {rec.get('year_min')}")
            print(f"  year_max : {rec.get('year_max')}")

            # Pour ne pas spammer, on affiche seulement 200 caractères de texte
            text_preview = rec.get("text", "")[:200].replace("\n"," ")
            print(f"Text    : {text_preview}...")
            print()

if __name__ == "__main__":
    main()
