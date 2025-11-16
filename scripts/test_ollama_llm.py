# scripts/test_ollama_llm.py
from __future__ import annotations

import sys
from pathlib import Path

# --- Ajout du dossier src/ au PYTHONPATH ---
ROOT_DIR = Path(__file__).resolve().parents[1]  # remonte à la racine du projet
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medieval_rag.rag.llm_client import OllamaLLMClient, LLMConfig


def main() -> None:
    # Configuration du client Ollama
    config = LLMConfig(
        model="mistral:latest",           # adapte si tu utilises un autre modèle Ollama
        base_url="http://localhost:11434",
        temperature=0.2,
        max_tokens=256,
    )
    client = OllamaLLMClient(config)

    system_prompt = (
        "Tu es un historien médiéviste. "
        "Réponds de façon courte, claire et en français."
    )

    question = "En une phrase, explique ce qu'est le haut Moyen Âge en Europe occidentale."

    print("➡️ Envoi de la requête à Ollama...")
    try:
        answer = client.generate(
            prompt=question,
            system_prompt=system_prompt,
        )
    except Exception as e:
        print(f"❌ Erreur lors de l'appel à Ollama : {e}")
        return

    print("\n✅ Réponse reçue :\n")
    print(answer)


if __name__ == "__main__":
    main()
