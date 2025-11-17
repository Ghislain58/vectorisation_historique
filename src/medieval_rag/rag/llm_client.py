# src/medieval_rag/rag/llm_client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests


@dataclass
class LLMConfig:
    """
    Configuration minimale pour le client Ollama LLM.
    Tu peux changer le modèle ici pour que tout le projet utilise le même.
    """
    model: str = "llama3:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: int = 120  # secondes


class OllamaLLMClient:
    """
    Client léger pour envoyer des requêtes à Ollama via /api/chat.
    Compatible avec les prompts système + utilisateur du pipeline RAG.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        print(f"🔍 LLM utilisé par le pipeline : {self.config.model}")

    # ------------------------------------------------------------------
    # Méthode principale : génération d’un texte via Ollama
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Appelle Ollama via /api/chat.
        Retourne uniquement le contenu textuel généré.
        """

        url = f"{self.config.base_url.rstrip('/')}/api/chat"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
            "stream": False,
            "messages": [],
        }

        # Prompt système (optionnel mais essentiel pour le mode "historien strict")
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})

        # Prompt utilisateur
        payload["messages"].append({"role": "user", "content": prompt})

        # Permet de passer d’autres paramètres si besoin
        if extra_params:
            payload.update(extra_params)

        # ------------------ Requête vers Ollama ------------------
        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout)
        except requests.RequestException as e:
            raise RuntimeError(
                f"Erreur de connexion à Ollama ({url}) : {e}"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Erreur HTTP {response.status_code} depuis Ollama : {response.text}"
            )

        # ------------------ Analyse de la réponse ------------------
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Réponse non JSON depuis Ollama : {response.text}") from e

        # Format standard pour /api/chat
        message = data.get("message") or {}
        content = message.get("content")

        if not content:
            raise RuntimeError(f"Aucun contenu généré par Ollama : {data}")

        return content
