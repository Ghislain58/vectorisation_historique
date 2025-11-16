from pathlib import Path
import sys
import json
import uuid

# Ajouter src/ au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from medieval_rag.ingestion.loaders import iter_documents
from medieval_rag.ingestion.chunker import chunk_document
from medieval_rag.embeddings.model_loader import load_embedding_model
from medieval_rag.embeddings.embedder import Embedder


def main():
    data_root = Path("data")
    local_pdf_dir = data_root / "sources" / "local" / "pdf"
    gallica_pdf_dir = data_root / "sources" / "bnf_gallica" / "pdf"

    output_dir = data_root / "chunks" / "standard"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "corpus_chunks.jsonl"

    print("🚀 Construction du corpus JSONL (RAG minimal)")
    print(f"📂 Dossiers PDF :")
    print(f"   - Local   : {local_pdf_dir}")
    print(f"   - Gallica : {gallica_pdf_dir}")
    print(f"📝 Fichier de sortie : {output_jsonl}")

    # Charger le modèle d'embedding
    model, device = load_embedding_model(
        "intfloat/multilingual-e5-large",
        device="auto"
    )
    embedder = Embedder(model, max_batch_size=16)

    nb_docs = 0
    nb_chunks_total = 0

    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for doc in iter_documents(local_pdf_dir, gallica_pdf_dir):
            nb_docs += 1
            print(f"\n📚 Document : {doc['doc_id']} ({doc['source']})")

            chunks = chunk_document(
                doc,
                max_chars=1800,
                overlap_chars=200
            )
            if not chunks:
                print("   ⚠️ Aucun chunk généré, document ignoré.")
                continue

            print(f"   → {len(chunks)} chunks")

            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_texts(texts)

            for chunk, emb in zip(chunks, embeddings):
                record = {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "ark": doc.get("ark"),
                    "title": doc.get("title"),
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                    # pas encore d'entités → on les ajoutera plus tard
                    "entities": None,
                    "year_min": None,
                    "year_max": None,
                    "embedding": emb,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            nb_chunks_total += len(chunks)

    print("\n✅ Terminé.")
    print(f"   Documents traités : {nb_docs}")
    print(f"   Chunks totaux     : {nb_chunks_total}")
    print(f"   Fichier créé      : {output_jsonl}")


if __name__ == "__main__":
    main()
