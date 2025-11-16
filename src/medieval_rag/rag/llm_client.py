import os
from typing import Optional, List, Dict, Any


class LLMClientError(Exception):
    """Erreur générique du client LLM."""
    pass


class LLMClient:
    """
    Client LLM unifié pour deux backends possibles :
      - mode="ollama" : LLM local via serveur Ollama (http://localhost:11434)
      - mode="openai" : LLM via API OpenAI

    Par défaut, on privilégie le mode local (ollama), avec un modèle spécifié
    dans la variable d'environnement OLLAMA_MODEL, sinon une valeur par défaut.
    """

    def __init__(
        self,
        mode: str = "ollama",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        """
        :param mode: "ollama" ou "openai"
        :param model: nom du modèle (ex: "llama3:8b" pour ollama,
                      "gpt-4.1-mini" pour openai)
        :param temperature: créativité (0 = très factuel)
        :param max_tokens: limite approximative de tokens de sortie
        """
        self.mode = mode.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.mode == "ollama":
            self._init_ollama(model)
        elif self.mode == "openai":
            self._init_openai(model)
        else:
            raise ValueError(f"Mode LLM non supporté : {mode}")

    # ------------------------------------------------------------------
    # Initialisation backends
    # ------------------------------------------------------------------

    def _init_ollama(self, model: Optional[str]) -> None:
        """
        Initialisation pour Ollama local.
        On utilise l'API HTTP standard d'Ollama : POST /api/chat
        """
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # modèle local puissant (à adapter à ce que tu as installé)
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

        try:
            import requests  # type: ignore
            self._requests = requests
        except Exception as e:
            raise LLMClientError(
                "Le module 'requests' est requis pour le mode ollama.\n"
                "Installe-le avec : pip install requests"
            ) from e

    def _init_openai(self, model: Optional[str]) -> None:
        """
        Initialisation pour OpenAI.
        On s'appuie sur la variable d'environnement OPENAI_API_KEY.
        """
        try:
            from openai import OpenAI  # type: ignore
            self._openai_client = OpenAI()
        except Exception as e:
            raise LLMClientError(
                "Le module 'openai' est requis pour le mode openai.\n"
                "Installe-le avec : pip install openai"
            ) from e

        # modèle OpenAI par défaut (à ajuster selon ton abonnement)
        self.model = model or os.getenv("OPENAI_RAG_MODEL", "gpt-4.1-mini")

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Envoie un dialogue (system + user + éventuellement messages additionnels)
        au LLM et retourne uniquement le contenu texte de la réponse.

        :param system_prompt: rôle système (instructions globales)
        :param user_prompt: message utilisateur (question RAG + contexte)
        :param extra_messages: liste optionnelle de messages intermédiaires
                               [{"role": "...", "content": "..."}, ...]
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_prompt})

        if self.mode == "ollama":
            return self._generate_ollama(messages)
        elif self.mode == "openai":
            return self._generate_openai(messages)
        else:
            raise LLMClientError(f"Mode LLM non supporté : {self.mode}")

    # ------------------------------------------------------------------
    # Implémentations spécifiques
    # ------------------------------------------------------------------

    def _generate_ollama(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            resp = self._requests.post(url, json=payload, timeout=600)
        except Exception as e:
            raise LLMClientError(
                f"Erreur de connexion à Ollama ({self.base_url}) : {e}"
            ) from e

        if resp.status_code != 200:
            raise LLMClientError(
                f"Réponse HTTP {resp.status_code} d'Ollama : {resp.text}"
            )

        data = resp.json()
        # Format de réponse typique :
        # {
        #   "model": "...",
        #   "created_at": "...",
        #   "message": {"role": "assistant", "content": "...."},
        #   ...
        # }
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        return str(content).strip()

    def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
        try:
            completion = self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise LLMClientError(f"Erreur lors de l'appel OpenAI : {e}") from e

        try:
            content = completion.choices[0].message.content or ""
        except Exception as e:
            raise LLMClientError(
                f"Réponse inattendue d'OpenAI : {completion}"
            ) from e

        return str(content).strip()
