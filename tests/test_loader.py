from pathlib import Path
import sys

# Ajoute le dossier "src" au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]  # dossier du projet
SRC = ROOT / "src"
sys.path.append(str(SRC))

from medieval_rag.ingestion.loaders import iter_documents

def main():
    local_pdf_dir = Path("data/sources/local/pdf")
    gallica_pdf_dir = Path("data/sources/bnf_gallica/pdf")

    for doc in iter_documents(local_pdf_dir, gallica_pdf_dir):
        print(f"📚 Document : {doc['doc_id']} ({doc['source']})")
        print(f"  Pages extraites : {len(doc['pages_text'])}")
        if doc["pages_text"]:
            print("  Extrait page 1 :")
            print(doc["pages_text"][0][:300])
        print("-" * 80)

if __name__ == "__main__":
    main()
