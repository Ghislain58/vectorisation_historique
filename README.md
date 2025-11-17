Vectorisation Historique & Moteur RAG Médiéval

Système complet pour analyser un corpus scientifique médiéval (thèses, articles, actes, PDF Gallica) et fournir des réponses sourcées via un moteur RAG (Retrieval Augmented Generation).

Optimisé pour :

WSL / Linux

Python 3.10+

CUDA

Embeddings E5-large

Index vectoriel FAISS

LLM local via Ollama

🏛️ Objectif du projet

Construire un assistant médiéval autonome, capable de :

extraire et analyser des sources médiévales complexes,

effectuer une recherche sémantique rigoureuse dans un corpus scientifique,

citer précisément les pages et documents,

réduire drastiquement les hallucinations grâce à FAISS + E5 + prompt historien strict,

fonctionner totalement hors-ligne via Ollama.

Le moteur peut s’intégrer dans ton travail artistique / numérique (Symbioware / CogniLink / installations interactives).

🏗 Architecture du projet
vectorisation_historique/
│
├── src/medieval_rag/
│   ├── ingestion/
│   │    ├── loaders.py        # Extraction PDF page par page
│   │    └── chunker.py        # Chunking structuré (1800 + overlap 200)
│   │
│   ├── enrichment/
│   │    └── entities.py       # Détection entités médiévales : personnes, lieux, années
│   │
│   ├── embeddings/
│   │    ├── model_loader.py   # Chargement E5-large (GPU/CPU)
│   │    └── embedder.py       # Embeddings batchés (sécurité VRAM)
│   │
│   └── rag/
│        ├── rag_pipeline.py   # Pipeline RAG complet (FAISS + LLM)
│        └── llm_client.py     # Client Ollama (API chat locale)
│
├── scripts/
│   ├── build_corpus_jsonl.py  # Pipeline ingestion → chunks → embeddings → JSONL
│   ├── build_faiss_index.py   # Construction index FAISS + index_ids.json
│   ├── rag_query_pipeline.py  # Interface principale RAG (FAISS + LLM)
│   ├── rag_query_llm.py       # Variante LLM-only (ancien test)
│   ├── debug_search.py        # Analyse FAISS avancée (expert)
│   ├── inspect_entities.py    # Vérification lexique médiéval
│   └── test_ollama_llm.py     # Test direct du modèle Ollama
│
├── tests/
│   ├── test_loader.py         # Test extraction PDF
│   ├── test_chunker.py        # Test chunking
│   └── test_embeddings.py     # Test embeddings E5
│
├── legacy/                    # ANCIEN PIPELINE (non utilisé)
│   ├── historical_vectorizer.py     # Obsolète (ChromaDB)
│   ├── vectorizer.py
│   ├── analyse_*.py
│   └── autres anciens scripts
│
├── data/                      # Non versionné (voir .gitignore)
│   ├── sources/               # PDF bruts
│   ├── chunks/                # corpus_chunks.jsonl (généré)
│   └── embeddings/            # index.faiss + index_ids.json
│
├── requirements.txt
├── requirements_freeze.txt
├── DEPENDENCIES.md
└── README.md

⚙️ Installation
git clone git@github.com:Ghislain58/vectorisation_historique.git
cd vectorisation_historique

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

📂 Sources à préparer (non versionnées)

Déposer vos PDF dans :

data/sources/local/pdf/
data/sources/bnf_gallica/pdf/

♻️ Reconstruction du corpus (FAISS + JSONL)

Le dépôt ne contient pas les données lourdes suivantes :

PDF

corpus_chunks.jsonl

index.faiss / index_ids.json

Ces fichiers sont générés localement.

1️⃣ Générer le corpus JSONL (chunks + entités + embeddings)
source venv/bin/activate
python scripts/build_corpus_jsonl.py


Sortie :

data/chunks/standard/corpus_chunks.jsonl


Ce fichier contient :

texte chunké

pages

id de document

entités détectées (people, places, years)

embeddings E5-large

2️⃣ Construire l’index FAISS
python scripts/build_faiss_index.py


Sorties :

data/embeddings/e5_large/index.faiss
data/embeddings/e5_large/index_ids.json

🔍 Vérifications et outils de contrôle
🔎 Recherche FAISS sans LLM (debug expert)
python scripts/debug_search.py


Permet d’inspecter :

les distances FAISS

les chunks les plus proches

les pages correspondantes

les entités médiévales détectées

🔎 Vérification du lexique médiéval
python scripts/inspect_entities.py


Utile pour affiner :

noms propres médiévaux

lieux anciens

années

éliminer les faux positifs (“dit”, “fils de”, etc.)

🔎 Tests unitaires
python scripts/test_loader.py
python scripts/test_chunker.py
python scripts/test_embeddings.py
python scripts/test_ollama_llm.py

🤖 Faire une requête RAG (pipeline complet)
python scripts/rag_query_pipeline.py \
    -q "Quel rôle jouent les évêques de Clermont dans les fondations monastiques ?" \
    --top-k 5


Pipeline :

Embedding de la requête (E5-large)

Recherche vectorielle FAISS

Reconstruction des extraits

Prompt historien strict

Appel au modèle local via Ollama

Réponse sourcée et non-hallucinée

🧠 Utiliser un LLM local (Ollama)

Installer un modèle :

ollama pull llama3.1


Configurer dans :

src/medieval_rag/rag/llm_client.py

🛠 Workflow Git
Branches recommandées

main → version stable

dev → développement courant

feat/... → nouvelles fonctionnalités

Cycle standard
git checkout dev
git pull
# travail...
git add .
git commit -m "feat: ..."
git push


Merge vers main via Pull Request sur GitHub.

🟤 Legacy (ancien pipeline)

Le dossier legacy/ contient l’ancien pipeline basé sur ChromaDB.
Il est conservé uniquement :

pour mémoire,

pour la reprise d’algorithmes (OCR, heuristiques),

mais il n’est pas utilisé dans le pipeline FAISS actuel.

🎯 Résultat :

Un moteur RAG médiéval :

propre

modulaire

stable

professionnel

entièrement offline

adapté aux corpus massifs

extensible vers ton projet artistique ou scientifique