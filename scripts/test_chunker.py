from pathlib import Path
import sys

# Ajouter src/ au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from medieval_rag.ingestion.loaders import iter_documents
from medieval_rag.ingestion.chunker import chunk_document


def main():
    local_pdf_dir = Path("data/sources/local/pdf")
    gallica_pdf_dir = Path("data/sources/bnf_gallica/pdf")

    for doc in iter_documents(local_pdf_dir, gallica_pdf_dir):
        print(f"📚 Document : {doc['doc_id']} ({doc['source']})")

        chunks = chunk_document(doc, max_chars=1800, overlap_chars=200)
        print(f"  → {len(chunks)} chunks générés")

        # Montrer un exemple de chunk
        if chunks:
            print("  Exemple de chunk :")
            ch = chunks[0]
            print(f"    Pages : {ch['page_start']}–{ch['page_end']}")
            print(f"    Longueur texte : {len(ch['text'])} caractères")
            print(f"    Aperçu : {ch['text'][:300]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
