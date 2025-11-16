# src/medieval_rag/rag/llm_client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests


@dataclass
class LLMConfig:
    model: str = "llama3:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: int = 120  # secondes


class OllamaLLMClient:
    """
    Client léger pour appeler un modèle Ollama en mode chat.
    Hypothèse : Ollama tourne déjà en local (ollama serve) et le modèle est présent.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Envoie une requête à Ollama et renvoie le texte généré.
        Utilise l'endpoint /api/chat pour bénéficier du format messages.
        """
        url = f"{self.config.base_url.rstrip('/')}/api/chat"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
            "stream": False,
            "messages": [],
        }

        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})

        payload["messages"].append({"role": "user", "content": prompt})

        if extra_params:
            payload.update(extra_params)

        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"Erreur de connexion à Ollama ({url}) : {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Erreur HTTP {response.status_code} depuis Ollama : {response.text}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Réponse non JSON depuis Ollama : {response.text}") from e

        # Format attendu pour /api/chat : {"message": {"role": "...", "content": "..."}}
        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(f"Aucun contenu généré par Ollama : {data}")

        return content
