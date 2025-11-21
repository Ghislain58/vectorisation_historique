# 📘 Vectorisation Historique & Moteur RAG Médiéval  

Système complet pour analyser un corpus scientifique médiéval (thèses, articles, actes, PDF Gallica) et fournir des réponses sourcées via un moteur **RAG** (Retrieval Augmented Generation).

🏛️ Objectif du projet
Construire un assistant médiéval autonome, capable de :

interroger un corpus scientifique médiéval,

effectuer une recherche sémantique rigoureuse,

citer précisément pages et documents,

limiter les hallucinations (FAISS + E5 + prompt historien strict),

fonctionner hors-ligne avec un LLM local via Ollama.

Le moteur peut ensuite s’intégrer dans des projets numériques / artistiques (Symbioware / CogniLink, installations interactives, etc.).



Optimisé pour :

- WSL / Linux  
- Python 3.10+  
- CUDA  
- Embeddings E5-large  
- Index vectoriel FAISS  
- LLM local via **Ollama**

---


🏗 Architecture du projet

vectorisation_historique/
│
├── src/medieval_rag/
│   ├── ingestion/
│   │    ├── loaders.py        # Extraction PDF page par page
│   │    └── chunker.py        # Chunking structuré (1800 + overlap 200)
│   │
│   ├── enrichment/
│   │    └── entities.py       # Entités médiévales : personnes, lieux, années
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
│   ├── build_corpus_jsonl.py  # Ingestion → chunks → embeddings → JSONL
│   ├── build_faiss_index.py   # Construction index.faiss + index_ids.json
│   ├── rag_query_pipeline.py  # Interface principale RAG (FAISS + LLM)
│   ├── rag_query_llm.py       # Variante plus simple (LLM configurable)
│   ├── debug_search.py        # Analyse FAISS avancée
│   ├── inspect_entities.py    # Vérification lexique médiéval
│   └── test_ollama_llm.py     # Test direct du modèle Ollama
│
├── tests/
│   ├── test_loader.py         # Test extraction PDF
│   ├── test_chunker.py        # Test chunking
│   └── test_embeddings.py     # Test embeddings E5
│
├── legacy/                    # ANCIEN PIPELINE (non utilisé)
│   ├── historical_vectorizer.py
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
📂 Préparer les sources (non versionnées)
Déposer les PDF dans :


data/sources/local/pdf/        # Thèses, articles, PDF perso
data/sources/bnf_gallica/pdf/  # PDF Gallica
♻️ Reconstruction du corpus (FAISS + JSONL)
1️⃣ Générer le corpus JSONL (chunks + entités + embeddings)

source venv/bin/activate
python scripts/build_corpus_jsonl.py
Sortie :
data/chunks/standard/corpus_chunks.jsonl
Ce JSONL contient pour chaque chunk :

texte,

numéro(s) de page,

doc_id,

métadonnées (source, titre…),

entités (personnes / lieux / années),

embedding E5-large.

2️⃣ Construire l’index FAISS

python scripts/build_faiss_index.py
Sorties :


data/embeddings/e5_large/index.faiss
data/embeddings/e5_large/index_ids.json
🔍 Vérifications et debug
🔎 Test extraction PDF

python tests/test_loader.py
🔎 Test chunking

python tests/test_chunker.py
🔎 Test embeddings E5-large

python tests/test_embeddings.py
🔎 Test LLM local (Ollama)

python scripts/test_ollama_llm.py
Affiche le modèle utilisé (ex : mistral:latest) et une réponse de test.

🔎 Debug FAISS avancé

python scripts/debug_search.py
Permet de :

inspecter les résultats FAISS,

contrôler les distances,

vérifier les pages et les textes,

voir les entités détectées.

🤖 Faire une requête RAG complète
Commande principale

python scripts/rag_query_pipeline.py \
    -q "Quel rôle jouent les évêques de Clermont dans l'implantation des monastères d'Auvergne ?" \
    --top-k 5
Paramètres importants :

-q / --query : ta question en langage naturel

--top-k : nombre de chunks les plus pertinents à récupérer

Pipeline exécuté :

Embedding de la requête (intfloat/multilingual-e5-large)

Recherche vectorielle FAISS

Récupération des chunks JSONL correspondants

Construction d’un prompt “historien strict, sourcé”

Appel à Ollama via llm_client.py

Affichage :

RÉPONSE DU LLM

SOURCES UTILISÉES (docs + pages + scores)

🤖 Modèles locaux (Ollama)
Lister les modèles :


ollama list
Installer un modèle (exemples) :


ollama pull llama3
ollama pull mistral
Le modèle utilisé par défaut dans le pipeline RAG est configuré dans :


src/medieval_rag/rag/llm_client.py
Tu peux adapter :

model = "llama3:latest"
ou

model = "mistral:latest"

🛠 Workflow Git
Branches recommandées :

main → version stable

dev → développement courant

feat/... → nouvelles fonctionnalités

Exemple de cycle :


git checkout dev
git pull

# travail...

git add .
git commit -m "feat: amélioration pipeline RAG"
git push origin dev
Merge vers main via Pull Request sur GitHub.

🟤 Legacy
Le dossier legacy/ contient l’ancien pipeline basé sur ChromaDB.
Il est conservé pour mémoire (OCR, heuristiques, explorations),

mais il n’est plus utilisé dans le pipeline FAISS actuel.

