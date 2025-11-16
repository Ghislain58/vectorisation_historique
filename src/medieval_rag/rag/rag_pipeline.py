from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .llm_client import OllamaLLMClient, LLMConfig


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class RAGPipeline:
    """
    Pipeline RAG :
      - encode la requête avec le même modèle que l'index (intfloat/multilingual-e5-large)
      - interroge FAISS
      - reconstruit les chunks depuis le JSONL
      - appelle le LLM (Ollama) avec les passages trouvés
    """

    def __init__(
        self,
        base_path: Path,
        index_path: Path,
        ids_path: Path,
        chunks_path: Path,
        llm_client: Optional[OllamaLLMClient] = None,
        device: Optional[str] = None,
    ) -> None:
        self.base_path = base_path
        self.index_path = index_path
        self.ids_path = ids_path
        self.chunks_path = chunks_path

        # Device pour les embeddings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._load_embedding_model()
        self._load_faiss_index()
        self._load_index_ids()
        self._load_chunks()

        self.llm = llm_client or OllamaLLMClient(LLMConfig())

    # ------------------------------------------------------------------
    # Chargement des composants
    # ------------------------------------------------------------------
    def _load_embedding_model(self) -> None:
        model_name = "intfloat/multilingual-e5-large"
        print(f"📥 Chargement du modèle d'embedding pour le RAG : {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device=self.device)

    def _load_faiss_index(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index FAISS introuvable : {self.index_path}")
        print(f"📂 Chargement de l'index FAISS : {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

    def _load_index_ids(self) -> None:
        if not self.ids_path.exists():
            raise FileNotFoundError(f"Fichier index_ids.json introuvable : {self.ids_path}")
        print(f"📂 Chargement des IDs d'index : {self.ids_path}")
        with self.ids_path.open("r", encoding="utf-8") as f:
            self.index_ids: List[str] = json.load(f)

    def _load_chunks(self) -> None:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Fichier de chunks JSONL introuvable : {self.chunks_path}")
        print(f"📂 Chargement des chunks depuis : {self.chunks_path}")
        self.chunks: Dict[str, Dict[str, Any]] = {}
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunk_id = obj.get("id") or obj.get("chunk_id")
                if chunk_id is None:
                    # Secours si un chunk n'a pas d'ID explicite
                    chunk_id = str(len(self.chunks))
                self.chunks[chunk_id] = obj

        print(f"   → {len(self.chunks)} chunks chargés.")

    # ------------------------------------------------------------------
    # Embedding + recherche FAISS
    # ------------------------------------------------------------------
    def _encode_query(self, query: str) -> np.ndarray:
        """
        E5 : on préfixe la requête par 'query: ' et on normalise le vecteur.
        """
        prepared = f"query: {query}"
        emb = self.embedding_model.encode(
            prepared,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb.astype("float32")

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_vec = self._encode_query(query)
        scores, indices = self.index.search(query_vec, top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if idx >= len(self.index_ids):
                continue
            chunk_id = self.index_ids[idx]
            chunk_obj = self.chunks.get(chunk_id)
            if not chunk_obj:
                continue
            text = (
                chunk_obj.get("text")
                or chunk_obj.get("content")
                or chunk_obj.get("chunk")
                or ""
            )
            metadata = {
                k: v
                for k, v in chunk_obj.items()
                if k not in {"text", "content", "chunk"}
            }
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=text,
                    metadata=metadata,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Construction du prompt et appel LLM
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        return (
            "Tu es un historien médiéviste rigoureux. "
            "Tu dois répondre uniquement à partir des extraits fournis, "
            "qui sont issus d'un corpus scientifique (articles, thèses, sources éditées). "
            "Si l'information n'est pas présente dans ces extraits, dis-le clairement. "
            "Réponds en français, de façon claire et structurée, en citant les passages pertinents."
        )

    def _build_user_prompt(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> str:
        context_parts: List[str] = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.metadata.get("source") or chunk.metadata.get("filename") or "source inconnue"
            page = chunk.metadata.get("page") or chunk.metadata.get("page_number")
            header = f"[Document {i} — id={chunk.chunk_id}, source={source}"
            if page is not None:
                header += f", page={page}"
            header += "]"
            context_parts.append(header)
            context_parts.append(chunk.text.strip())
            context_parts.append("")  # ligne vide

        context_block = "\n".join(context_parts)

        prompt = (
            "Voici des extraits issus de ton corpus médiéval :\n\n"
            f"{context_block}\n\n"
            "Question de l'utilisateur :\n"
            f"{question}\n\n"
            "Consignes :\n"
            "- Appuie-toi uniquement sur les extraits fournis ci-dessus.\n"
            "- Si tu déduis quelque chose, explique le raisonnement à partir des extraits.\n"
            "- Indique clairement les références aux extraits (par exemple : « voir Document 2, page 134 »).\n"
            "- Si tu ne peux pas répondre avec ces extraits, dis-le explicitement.\n"
        )
        return prompt

    def answer(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Tuple[str, List[RetrievedChunk]]:
        retrieved = self.search(question, top_k=top_k)
        if not retrieved:
            raise RuntimeError("Aucun passage pertinent trouvé dans l'index pour cette question.")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, retrieved)

        answer_text = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return answer_text, retrieved
