from __future__ import annotations

import sys
import argparse
from pathlib import Path

# --- Ajout du dossier src/ au PYTHONPATH ---
ROOT_DIR = Path(__file__).resolve().parents[1]  # racine du projet
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medieval_rag.rag.rag_pipeline import RAGPipeline
from medieval_rag.rag.llm_client import OllamaLLMClient, LLMConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interroger le moteur RAG médiéval (index FAISS + LLM Ollama)."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Question en langage naturel (ex: 'Quel est le rôle des Avitii à Clermont ?')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de passages à récupérer dans FAISS (par défaut: 5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral:latest",
        help="Nom du modèle Ollama à utiliser (par défaut: mistral:latest).",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=".",
        help="Racine du projet (par défaut: répertoire courant).",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/embeddings/e5_large/index.faiss",
        help="Chemin vers l'index FAISS.",
    )
    parser.add_argument(
        "--ids-path",
        type=str,
        default="data/embeddings/e5_large/index_ids.json",
        help="Chemin vers le fichier JSON des IDs de l'index.",
    )
    parser.add_argument(
        "--chunks-path",
        type=str,
        default="data/chunks/standard/corpus_chunks.jsonl",
        help="Chemin vers le fichier JSONL contenant les chunks.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Température de génération du LLM (par défaut: 0.2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Nombre maximal de tokens générés (approximation, selon le modèle).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    index_path = base_path / args.index_path
    ids_path = base_path / args.ids_path
    chunks_path = base_path / args.chunks_path

    print("🚀 Initialisation du pipeline RAG (FAISS + Ollama)...")
    print(f"   Base path       : {base_path}")
    print(f"   Index FAISS     : {index_path}")
    print(f"   Index IDs       : {ids_path}")
    print(f"   Fichier chunks  : {chunks_path}")
    print(f"   Modèle Ollama   : {args.model}")

    llm_client = OllamaLLMClient(
        LLMConfig(
            model=args.model,
        )
    )

    pipeline = RAGPipeline(
        base_path=base_path,
        index_path=index_path,
        ids_path=ids_path,
        chunks_path=chunks_path,
        llm_client=llm_client,
    )

    answer, retrieved = pipeline.answer(
        question=args.query,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("\n🧠 Réponse du modèle :\n")
    print(answer)
    print("\n📚 Passages utilisés :\n")
    for i, chunk in enumerate(retrieved, start=1):
        source = chunk.metadata.get("source") or chunk.metadata.get("filename") or "source inconnue"
        page = chunk.metadata.get("page") or chunk.metadata.get("page_number")
        print(f"--- Passage {i} ---")
        print(f"id       : {chunk.chunk_id}")
        print(f"score    : {chunk.score:.4f}")
        print(f"source   : {source}")
        if page is not None:
            print(f"page     : {page}")
        print("extrait  :")
        text_preview = chunk.text.strip().replace("\n", " ")
        if len(text_preview) > 400:
            text_preview = text_preview[:400] + " [...]"
        print(text_preview)
        print()


if __name__ == "__main__":
    main()
