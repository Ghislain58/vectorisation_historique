#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="vectorisation_historique"
VENV_DIR="venv"
PYTHON_BIN="python3"
REQ_FILE="requirements.txt"

echo "==============================================="
echo "  🛠 Installation du projet RAG médiéval"
echo "  Projet : ${PROJECT_NAME}"
echo "==============================================="
echo

# 1) Vérification de python3
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "❌ ${PYTHON_BIN} introuvable. Installe Python 3.10+ puis relance ce script."
  exit 1
fi

echo "✅ Python détecté : $(${PYTHON_BIN} --version)"
echo

# 2) Création (ou réutilisation) de l'environnement virtuel
if [ ! -d "${VENV_DIR}" ]; then
  echo "📦 Création de l'environnement virtuel : ${VENV_DIR}"
  ${PYTHON_BIN} -m venv "${VENV_DIR}"
else
  echo "ℹ️ Environnement virtuel déjà présent : ${VENV_DIR}"
fi

# Activation venv
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "✅ Environnement activé : ${VENV_DIR}"
echo

# 3) Installation des dépendances
if [ ! -f "${REQ_FILE}" ]; then
  echo "❌ Fichier ${REQ_FILE} introuvable à la racine du projet."
  exit 1
fi

echo "📥 Installation des dépendances depuis ${REQ_FILE}..."
pip install --upgrade pip
pip install -r "${REQ_FILE}"
echo "✅ Dépendances Python installées."
echo

# 4) Vérification d'Ollama (optionnel mais recommandé)
if command -v ollama >/dev/null 2>&1; then
  echo "✅ Ollama détecté."
  echo "   Modèles disponibles :"
  ollama list || true
  echo
  echo "ℹ️ Si nécessaire, installe un modèle, par exemple :"
  echo "   ollama pull llama3"
else
  echo "⚠️ Ollama n'est pas détecté sur cette machine."
  echo "   Le RAG fonctionnera, mais sans LLM local."
  echo "   Installe Ollama si tu veux des réponses générées : https://ollama.com"
fi

echo
echo "==============================================="
echo "  📂 Vérification des sources PDF"
echo "==============================================="

LOCAL_PDF_DIR="data/sources/local/pdf"
GALLICA_PDF_DIR="data/sources/bnf_gallica/pdf"

mkdir -p "${LOCAL_PDF_DIR}" "${GALLICA_PDF_DIR}"

LOCAL_COUNT=$(find "${LOCAL_PDF_DIR}" -maxdepth 1 -type f -iname '*.pdf' | wc -l | tr -d ' ')
GALLICA_COUNT=$(find "${GALLICA_PDF_DIR}" -maxdepth 1 -type f -iname '*.pdf' | wc -l | tr -d ' ')

echo "📁 ${LOCAL_PDF_DIR} : ${LOCAL_COUNT} PDF"
echo "📁 ${GALLICA_PDF_DIR} : ${GALLICA_COUNT} PDF"
echo

if [ "${LOCAL_COUNT}" = "0" ] && [ "${GALLICA_COUNT}" = "0" ]; then
  echo "⚠️ Aucun PDF détecté pour le moment."
  echo "   Tu peux copier tes PDF dans :"
  echo "   - ${LOCAL_PDF_DIR}"
  echo "   - ${GALLICA_PDF_DIR}"
  echo "   et relancer la partie build plus tard."
  BUILD_CORPUS="n"
else
  echo "✅ Des PDF sont présents."
  read -r -p "➡️ Lancer la construction du corpus maintenant ? [o/N] " BUILD_CORPUS
  BUILD_CORPUS=${BUILD_CORPUS:-n}
fi

echo
if [[ "${BUILD_CORPUS}" =~ ^[oOyY]$ ]]; then
  echo "==============================================="
  echo "  🧱 Construction du corpus JSONL"
  echo "==============================================="
  python scripts/build_corpus_jsonl.py

  echo
  echo "==============================================="
  echo "  🧱 Construction de l'index FAISS"
  echo "==============================================="
  python scripts/build_faiss_index.py

  echo
  echo "✅ Corpus et index FAISS reconstruits."
else
  echo "⏭  Construction du corpus FAISS ignorée pour l’instant."
  echo "   Tu pourras plus tard lancer manuellement :"
  echo "   - python scripts/build_corpus_jsonl.py"
  echo "   - python scripts/build_faiss_index.py"
fi

echo
echo "==============================================="
echo "  ✅ Installation terminée"
echo "==============================================="
echo "Pour tester le LLM :"
echo "  source venv/bin/activate"
echo "  python scripts/test_ollama_llm.py"
echo
echo "Pour faire une requête RAG :"
echo "  source venv/bin/activate"
echo "  python scripts/rag_query_pipeline.py -q \"Question historique\" --top-k 5"
echo
