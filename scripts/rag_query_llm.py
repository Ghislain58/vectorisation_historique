from pathlib import Path
import sys
import json
from textwrap import indent

import numpy as np
import faiss

# Ajouter src/ au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from medieval_rag.embeddings.model_loader import load_embedding_model
from medieval_rag.embeddings.embedder import Embedder
from medieval_rag.rag.llm_client import LLMClient, LLMClientError


def load_corpus(jsonl_path: Path):
    records_by_id = {}
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL introuvable : {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            if not cid:
                continue
            records_by_id[cid] = rec

    return records_by_id


def load_faiss_index(index_path: Path, ids_path: Path):
    if not index_path.exists():
        raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Fichier d'IDs introuvable : {ids_path}")

    index = faiss.read_index(str(index_path))
    with ids_path.open("r", encoding="utf-8") as f:
        ids_list = json.load(f)

    if len(ids_list) != index.ntotal:
        print(
            f"⚠️  Attention : {len(ids_list)} IDs, "
            f"{index.ntotal} vecteurs dans l'index."
        )

    return index, ids_list


def search_corpus(
    query: str,
    embedder: Embedder,
    index,
    ids_list,
    records_by_id,
    k: int = 5,
):
    q_emb = embedder.embed_texts([query])[0]
    q_vec = np.array([q_emb], dtype="float32")

    distances, indices = index.search(q_vec, k)
    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(ids_list):
            continue
        chunk_id = ids_list[idx]
        rec = records_by_id.get(chunk_id)
        if not rec:
            continue
        score = 1.0 / (1.0 + float(dist))
        results.append((score, rec))

    # tri décroissant par score
    results.sort(key=lambda x: x[0], reverse=True)
    return results


def build_context_block(results, max_chars: int = 6000) -> str:
    """
    Construit un bloc de contexte à injecter dans le prompt LLM.
    On concatène les meilleurs chunks avec des entêtes clairs.
    """
    parts = []
    total_chars = 0

    for i, (score, rec) in enumerate(results, start=1):
        doc_id = rec.get("doc_id")
        title = rec.get("title")
        source = rec.get("source")
        page_start = rec.get("page_start")
        page_end = rec.get("page_end")
        entities = rec.get("entities") or {}
        year_min = rec.get("year_min")
        year_max = rec.get("year_max")
        text = rec.get("text") or ""

        header_lines = [
            f"[DOC {i} | score={score:.3f}]",
            f"doc_id : {doc_id}",
            f"title  : {title}",
            f"source : {source}",
            f"pages  : {page_start}–{page_end}",
        ]

        persons = entities.get("persons") or []
        places = entities.get("places") or []
        years = entities.get("years") or []
        if persons or places or years or (year_min is not None or year_max is not None):
            header_lines.append("entities :")
            if persons:
                header_lines.append(f"  persons : {', '.join(persons)}")
            if places:
                header_lines.append(f"  places  : {', '.join(places)}")
            if years:
                header_lines.append(f"  years   : {', '.join(map(str, years))}")
            if year_min is not None or year_max is not None:
                header_lines.append(f"  span    : {year_min}–{year_max}")

        header = "\n".join(header_lines)
        block = f"{header}\n\n{text.strip()}\n"

        if total_chars + len(block) > max_chars:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n" + ("\n" + "-" * 80 + "\n\n").join(parts)


def build_system_prompt() -> str:
    """
    Prompt système : comportement de l'assistant.
    """
    return (
        "Tu es un assistant historien spécialisé dans le haut Moyen Âge, "
        "l'Auvergne, la Paix de Dieu, les réseaux aristocratiques et les sources "
        "érudites (thèses, articles, corpus Gallica).\n\n"
        "Règles impératives :\n"
        "- Tu NE DOIS répondre qu'à partir des extraits de documents fournis dans le contexte.\n"
        "- Si une information n'est pas présente dans les extraits, tu réponds clairement que "
        "les sources fournies ne permettent pas de conclure.\n"
        "- Tu cites systématiquement les documents utilisés (doc_id, titre, pages) dans ta réponse.\n"
        "- Tu signales les incertitudes, hypothèses et limites des sources.\n"
        "- Tu rédiges en français, dans un style clair, structuré, sans jargon inutile.\n"
    )


def build_user_prompt(question: str, context_block: str) -> str:
    """
    Prompt utilisateur : question + contexte des documents.
    """
    instructions = (
        "Voici une question de l'utilisateur, et un ensemble d'extraits de documents "
        "historiques pertinents (thèses, articles, sources secondaires). "
        "Tu dois t'appuyer exclusivement sur ces extraits pour répondre.\n\n"
        "Tâches :\n"
        "1) Analyser les extraits fournis et identifier ceux qui répondent à la question.\n"
        "2) Produire une réponse structurée (introduction courte, développement, conclusion).\n"
        "3) Citer explicitement les documents utilisés (doc_id, titre, pages), par exemple :\n"
        "   (Doc 3, La_Paix_des_Montagnes_les_origines_auver, p. 4–5).\n"
        "4) Si la réponse est partielle ou incertaine, l'expliquer.\n\n"
        "Question de l'utilisateur :\n"
        f"{question}\n\n"
        "Extraits contextuels :\n"
        f"{context_block}\n"
    )
    return instructions


def main():
    data_root = Path("data")
    jsonl_path = data_root / "chunks" / "standard" / "corpus_chunks.jsonl"
    index_dir = data_root / "embeddings" / "e5_large"
    index_path = index_dir / "index.faiss"
    ids_path = index_dir / "index_ids.json"

    # Chargement corpus + index
    print("📂 Chargement du corpus et de l'index...")
    records_by_id = load_corpus(jsonl_path)
    index, ids_list = load_faiss_index(index_path, ids_path)

    # Embeddings pour les requêtes
    print("🧠 Chargement du modèle d'embedding pour les requêtes...")
    model, device = load_embedding_model(
        "intfloat/multilingual-e5-large",
        device="auto"
    )
    embedder = Embedder(model, max_batch_size=16)

    # Client LLM (par défaut : ollama local)
    print("🤖 Initialisation du client LLM (mode local par défaut : ollama)...")
    try:
        llm = LLMClient(mode="ollama")
    except LLMClientError as e:
        print(f"❌ Erreur LLM : {e}")
        return

    print("\n✅ RAG médiéval prêt.")
    print("   Pose une question en français (ou latin), ou 'q' pour quitter.\n")

    while True:
        try:
            question = input("❓ Question (RAG) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Fin de session.")
            break

        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            print("👋 Fin de session.")
            break

        # 1) Recherche FAISS
        print("🔍 Recherche sémantique dans le corpus...")
        results = search_corpus(
            query=question,
            embedder=embedder,
            index=index,
            ids_list=ids_list,
            records_by_id=records_by_id,
            k=6,
        )

        if not results:
            print("⚠️ Aucun résultat trouvé dans le corpus pour cette question.")
            continue

        # 2) Construction du contexte pour le LLM
        context_block = build_context_block(results, max_chars=6000)

        # Affichage des docs utilisés (pour debug)
        print("\n📄 Documents retenus pour la génération :")
        for i, (score, rec) in enumerate(results, start=1):
            doc_id = rec.get("doc_id")
            title = rec.get("title")
            page_start = rec.get("page_start")
            page_end = rec.get("page_end")
            print(f"  - DOC {i}: {doc_id} | {title} | pages {page_start}–{page_end} | score={score:.3f}")

        # 3) Construction des prompts
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(question, context_block)

        print("\n🧪 Appel au LLM (génération de la réponse)...")
        try:
            answer = llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except LLMClientError as e:
            print(f"❌ Erreur LLM : {e}")
            continue

        print("\n📜 Réponse de l'assistant (RAG médiéval) :\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
