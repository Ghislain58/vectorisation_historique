```markdown
# 📚 GUIDE AVANCÉ — PIPELINE RAG MÉDIÉVAL

Ce document explique en détail le fonctionnement du moteur RAG : architecture interne, extensions possibles et bonnes pratiques.

---

# 1️⃣ Architecture détaillée

## 📥 1. Ingestion des PDF
- Extraction page par page
- Nettoyage léger
- Encodage unifié

Module :  
src/medieval_rag/ingestion/loaders.py



---

## 🧩 2. Chunking
- Fenêtres de ~1800 caractères  
- Overlap de 200  
- Compatible E5-large

Module :  
src/medieval_rag/ingestion/chunker.py



---

## 🧭 3. Détection d’entités médiévales
- Noms propres
- Lieux
- Années
- Filtrage des faux positifs (“dit”, “fils de”, etc.)

Module :  
src/medieval_rag/enrichment/entities.py



---

## 🧠 4. Embeddings E5-Large

- Modèle : `intfloat/multilingual-e5-large`
- Device : GPU si dispo
- Batch auto-adapté selon VRAM

Modules :  
src/medieval_rag/embeddings/model_loader.py
src/medieval_rag/embeddings/embedder.py



---

## 🗂 5. JSONL unifié

Généré par :

```
python scripts/build_corpus_jsonl.py
Contient : texte, pages, doc_id, métadonnées, entités, embeddings.

📡 6. Index FAISS
Créé avec :



python scripts/build_faiss_index.py
Produits :

index.faiss

index_ids.json

🤖 7. Pipeline RAG + LLM
Étapes :

Embedding de la requête

Recherche FAISS

Construction du contexte

Prompt historien strict

Appel au modèle local (Ollama)

Réponse sourcée

Module principal :

src/medieval_rag/rag/rag_pipeline.py
2️⃣ Exemples de requêtes utiles


python scripts/rag_query_pipeline.py \
  -q "Quels évêques d’Auvergne apparaissent dans les actes du IXe siècle ?" \
  --top-k 8


python scripts/rag_query_pipeline.py \
  -q "Quelle place occupe Brioude dans les réseaux aristocratiques ?" \
  --top-k 7
3️⃣ Étendre le pipeline
➕ Ajouter de nouveaux PDF
Mettre les fichiers dans data/sources/

Reconstruire :



python scripts/build_corpus_jsonl.py
python scripts/build_faiss_index.py
🔧 Changer le modèle LLM
Modifier :



src/medieval_rag/rag/llm_client.py
4️⃣ Legacy
Le dossier legacy/ contient l’ancien pipeline basé sur ChromaDB.
Il n’est plus utilisé mais reste utile pour référence ou reprise d’algorithmes.

🔚 Fin du guide RAG