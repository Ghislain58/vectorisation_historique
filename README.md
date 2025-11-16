📘 Vectorisation Historique & Moteur RAG Médiéval

Pipeline complet pour construire un assistant médiéval basé sur un corpus scientifique (thèses, articles, PDF Gallica).
Le système réalise :

extraction de texte (PDF locaux + Gallica)

chunking structuré

embeddings (E5-large)

index vectoriel FAISS

recherche sémantique

RAG (LLM API ou local via Ollama)

Optimisé pour WSL + Python + GPU CUDA.

📁 Architecture du projet
vectorisation_historique/
│
├── src/medieval_rag/
│   ├── ingestion/        # loaders PDF + chunker
│   ├── enrichment/       # extraction d’entités
│   ├── embeddings/       # modèle E5 + embedder
│   └── rag/              # LLM client + pipeline RAG
│
├── scripts/
│   ├── build_corpus_jsonl.py
│   ├── build_faiss_index.py
│   ├── debug_search.py
│   ├── inspect_entities.py
│   └── rag_query_llm.py
│
├── data/
│   ├── sources/          # PDF d'entrée (non versionnés)
│   ├── chunks/           # corpus JSONL généré
│   └── embeddings/       # index FAISS généré
│
├── requirements.in
├── requirements.txt
├── requirements_freeze.txt
├── DEPENDENCIES.md
└── README.md

⚙️ Installation
1. Cloner le projet
git clone git@github.com:Ghislain58/vectorisation_historique.git
cd vectorisation_historique

2. Installer l’environnement Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


(Vous pouvez utiliser requirements_freeze.txt pour une installation reproductible.)

♻️ Reconstruction des données (après un clone)

Le dépôt ne contient pas :

PDF

corpus_chunks.jsonl

index FAISS

Ces fichiers doivent être générés à nouveau localement.

1. Placer les PDF

Déposer vos PDF dans :

data/sources/local/pdf/       # Thèses, articles, sources personnelles
data/sources/bnf_gallica/pdf/ # PDF Gallica (optionnel)

2. Générer le corpus JSONL (chunks + embeddings)
source venv/bin/activate
python scripts/build_corpus_jsonl.py


Sortie :

data/chunks/standard/corpus_chunks.jsonl

3. Construire l’index FAISS
python scripts/build_faiss_index.py


Sorties :

data/embeddings/e5_large/index.faiss
data/embeddings/e5_large/index_ids.json

4. Tester la recherche sémantique
python scripts/debug_search.py


Permet de vérifier :

les résultats de recherche FAISS

la cohérence des textes

la détection des entités

5. Faire une requête RAG (LLM API ou local)
python scripts/rag_query_llm.py


Pipeline :

Embedding de la question

Recherche FAISS

Construction du contexte

Appel à un LLM (OpenAI, Groq, Ollama…)

Réponse historique sourcée et sans hallucinations

🔌 Intégration LLM local (Ollama)

Pour utiliser un modèle local :

ollama pull llama3.1


Puis configurer :

src/medieval_rag/rag/llm_client.py

mode = "ollama"
model = "llama3.1"

🔍 Outils de vérification des entités
python scripts/inspect_entities.py


Permet d’améliorer :

lexique des personnes

lieux médiévaux

années

élimination des faux positifs (“dit”, “comte”, etc.)

🧪 Tests rapides
Générer un mini-corpus de test
python scripts/build_corpus_jsonl.py --debug

Interroger sans LLM
python scripts/debug_search.py

🛠 Workflow Git
Ajouter des fichiers
git add .
git commit -m "feat: nouvelle fonctionnalité"
git push


Branches recommandées :

main  → stable
dev   → développement
feat/... → nouvelles fonctionnalités

🎯 Objectif du projet

Créer un assistant médiéval autonome, capable de :

répondre à des questions historiques complexes

analyser des sources primaires et secondaires

citer pages, documents, titres

fonctionner totalement hors-ligne

enrichir ton projet artistique / numérique Symbioware / CogniLink

✔️ Fin du README