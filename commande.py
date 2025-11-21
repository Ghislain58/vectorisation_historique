
## ⚡️ Cheat sheet – Commandes principales

### 1. Activer l’environnement

```bash
cd ~/vectorisation_historique
source venv/bin/activate
2. (Re)construire le corpus

python scripts/build_corpus_jsonl.py
python scripts/build_faiss_index.py
3. Tester le LLM local (Ollama)

python scripts/test_ollama_llm.py
4. Faire une requête RAG complète

python scripts/rag_query_pipeline.py \
  -q "Quel rôle jouent les évêques de Clermont dans les fondations monastiques ?" \
  --top-k 5
5. Debug FAISS avancé

python scripts/debug_search.py
