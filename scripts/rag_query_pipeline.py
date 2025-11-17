#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------
# Ajouter automatiquement le repo_root dans sys.path
# ---------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# On importe directement la classe RAGPipeline
from src.medieval_rag.rag.rag_pipeline import RAGPipeline  # type: ignore[import]


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def print_sources_from_chunks(chunks: List[Any]) -> None:
    if not chunks:
        print("Aucune source renvoyée.")
        return

    print_header("SOURCES UTILISÉES")
    for i, ch in enumerate(chunks, start=1):
        meta = ch.metadata or {}
        filename = (
            meta.get("title")
            or meta.get("doc_id")
            or meta.get("filename")
            or meta.get("source")
            or "document inconnu"
        )
        page = (
            meta.get("page")
            or meta.get("page_number")
            or meta.get("page_start")
        )
        print(f"[{i}] {filename}")
        print(f"    - chunk  : {ch.chunk_id}")
        print(f"    - score  : {ch.score:.4f}")
        if page is not None:
            print(f"    - page   : {page}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interroger le pipeline RAG médiéval (classe RAGPipeline)."
    )
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)

    args = parser.parse_args()
    question = args.query or input("Question : ").strip()

    if not question:
        print("❌ Aucune question fournie.")
        sys.exit(1)

    print("🚀 Initialisation du pipeline RAG...")
    pipeline = RAGPipeline(
        base_path=repo_root,
        index_path=repo_root / "data" / "embeddings" / "e5_large" / "index.faiss",
        ids_path=repo_root / "data" / "embeddings" / "e5_large" / "index_ids.json",
        chunks_path=repo_root / "data" / "chunks" / "standard" / "corpus_chunks.jsonl",
    )

    print(f"\n❓ Question : {question}\n")

    try:
        answer_text, retrieved_chunks = pipeline.answer(
            question=question,
            top_k=args.top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        print(f"❌ Erreur lors de l'interrogation du pipeline : {e}")
        sys.exit(1)

    print_header("RÉPONSE DU LLM")
    print((answer_text or "").strip() or "[Réponse vide]")

    print_sources_from_chunks(retrieved_chunks)


if __name__ == "__main__":
    main()
