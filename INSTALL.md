
# 📄 **INSTALL.md** (procédure complète pour réinstaller le projet)
👉 **À mettre à la racine du projet : `vectorisation_historique/INSTALL.md`**

**COLLE CE CONTENU DANS `INSTALL.md` :**

```markdown
# 🛠 INSTALLATION & RECONSTRUCTION DU PIPELINE RAG MÉDIÉVAL

Ce guide explique comment installer et reconstruire le pipeline complet sur n’importe quel ordinateur.

---

# 1️⃣ Prérequis

- Python 3.10+
- WSL (Ubuntu recommandé)
- CUDA (optionnel mais recommandé)
- Git
- Ollama installé

---

# 2️⃣ Cloner le projet


git clone git@github.com:Ghislain58/vectorisation_historique.git
cd vectorisation_historique

3️⃣ Installer l’environnement Python

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


4️⃣ Installer un modèle LLM dans Ollama


ollama pull llama3
Tester :
python scripts/test_ollama_llm.py

5️⃣ Ajouter les PDF
Placer les documents dans :


data/sources/local/pdf/
data/sources/bnf_gallica/pdf/


6️⃣ Générer le corpus JSONL


python scripts/build_corpus_jsonl.py
Output :


data/chunks/standard/corpus_chunks.jsonl


7️⃣ Construire l’index FAISS

python scripts/build_faiss_index.py
Outputs :


data/embeddings/e5_large/index.faiss
data/embeddings/e5_large/index_ids.json


8️⃣ Tester le pipeline RAG

python scripts/rag_query_pipeline.py \
  -q "Quel rôle jouent les évêques de Clermont dans les monastères d’Auvergne ?" \
  --top-k 5


9️⃣ Debug & outils avancés


python scripts/debug_search.py
Vérification des entités

python scripts/inspect_entities.py


Tests unitaires

python tests/test_loader.py
python tests/test_chunker.py
python tests/test_embeddings.py