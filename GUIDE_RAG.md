📚 GUIDE RAG — Vectorisation Historique (Version Complète)

Moteur RAG médiéval — FAISS + Embeddings E5-Large + LLM local (Ollama)

🚀 1. Lancer une requête RAG

C’est la commande principale :

source venv/bin/activate
python scripts/rag_query_pipeline.py -q "Votre question ici" --top-k 5


Exemple :

python scripts/rag_query_pipeline.py \
  -q "Quelle place occupe Brioude dans les réseaux aristocratiques au IXᵉ siècle ?" \
  --top-k 8


Cette commande fait :

Embedding de la question (E5-large)

Recherche vectorielle FAISS

Sélection des top-k chunks pertinents

Construction d’un contexte historique sourcé

Appel au LLM (Ollama, modèle : llama3:latest)

Retourne :

une réponse synthétique

les sources, pages, chunks et scores FAISS

un discours historien strict (pas d’hallucinations)

🎛️ 2. Réglages importants et subtilités de recherche
🔥 --top-k (pertinence et largeur de contexte)

5 → précis, idéal pour question ciblée

8–12 → questions de synthèse, thèmes larges

20+ → exploration, comparaison, panoramas

💡 Recommandation historique :
top-k = 8 → équilibre idéal (peu de bruit, assez de contexte)

🔥 Température LLM

Réglée dans llm_client.py :

0.0 → très strict, non-créatif

0.1–0.2 → idéal pour réponse historique fiable

0.4+ → plus interprétatif (éviter)

👉 Pour l’histoire médiévale : 0.1, c’est parfait.

🔥 Modèle LLM (Ollama)

Par défaut : llama3:latest

Changer :

ollama pull llama3


Et dans llm_client.py :

model="llama3:latest"


Autres possibilités :

mistral:latest

llama3.1 (si tu veux tester)

modèles API (OpenAI, Groq) → en modifiant LLMConfig

🔥 Subtilités de recherche historique
✔ Formuler une question précise

FAISS adore :

lieux > personnes > dates

syntagmes nominaux longs

évêques, abbayes, cartulaires, actes précis

Exemples :

❌ “Les monastères en Auvergne ?”
✔ “Quels acteurs ecclésiastiques apparaissent dans l’implantation des monastères d’Auvergne au IXᵉ siècle ?”

✔ Ajouter un acteur clé

Exemple :

“Avit”

“Géraud d’Aurillac”

“Sidoine Apollinaire”

“évêques de Clermont”

➡️ Excellence du matching FAISS sur anthroponymes uniques.

✔ Ajouter une période ou un siècle

Exemple :

“IXᵉ siècle”

“période carolingienne”

➡️ Active les années et améliore le score.

✔ Très important : le RAG est strict

Ton pipeline empêche :

les hallucinations

les inventions de sources

les synthèses hors contexte

Le LLM ne peut répondre que sur ce que FAISS lui donne.

🧱 3. Reconstruction du pipeline (si tu ajoutes des PDF)

Quand tu ajoutes de nouveaux PDF dans :

data/sources/local/pdf/
data/sources/bnf_gallica/pdf/


Il faut reconstruire le corpus + FAISS.

1️⃣ Générer les chunks
python scripts/build_corpus_jsonl.py


Produit :

data/chunks/standard/corpus_chunks.jsonl


Inclut :

texte

pages

doc_id

métadonnées

entités détectées

années min/max

2️⃣ Construire l’index FAISS
python scripts/build_faiss_index.py


Produit :

data/embeddings/e5_large/index.faiss
data/embeddings/e5_large/index_ids.json

🔍 4. Tester FAISS sans LLM

Très utile pour vérifier les résultats :

python scripts/debug_search.py


Affiche :

les meilleurs chunks

les scores de similarité

les métadonnées

les entités détectées

Permet de valider :

cohérence du corpus

qualité des embeddings

efficacité du lexique

🤖 5. Tester le modèle Ollama

Pour vérifier la connectivité :

python scripts/test_ollama_llm.py


Ou via curl :

curl http://localhost:11434/api/generate -d '{
  "model": "llama3:latest",
  "prompt": "Test de connexion."
}'

🧩 6. Fonctionnement interne du pipeline
📥 1. Ingestion

Module :
src/medieval_rag/ingestion/loaders.py

Extraction page par page

Nettoyage léger

OCR non obligatoire (mais possible plus tard)

🧩 2. Chunking (v1 standard)

Module : ingestion/chunker.py

~1800 caractères

overlap ~200

adapté à E5-large

stable et robuste

🧭 3. Détection d’entités

Module : enrichment/entities.py

Détecte :

personnes (lexique JSON local)

lieux (lexique local)

années (regex intelligente)

nettoie : chiffres, ponctuation, tokens faibles

enrichit chaque chunk

Lexique attendu :

mes_documents_historiques/entity_lexicon_v2.json


Structure :

{
  "persons": ["Avit", "Géraud d'Aurillac"],
  "places": ["Brioude", "Clermont"],
  "years": ["700", "800", "900"]
}

🧠 4. Embeddings E5-large

Modules :

embeddings/model_loader.py

embeddings/embedder.py

Caractéristiques :

modèle SOTA pour texte long

optimisé pour similarité sémantique

GPU si disponible

batch automatique selon ta VRAM

📡 5. Index FAISS

Module : build_faiss_index.py

Index :

métrique : dot product / inner product

normalisation des embeddings

IDs synchronisés avec JSONL

🤖 6. Pipeline RAG final

Module : rag/rag_pipeline.py

Séquence :

Embedding de la requête

Recherche FAISS

Assemblage du contexte (pages ordonnées)

Prompt “historien strict”

Envoi à Ollama

Réponse sourcée + chunks + scores

🧪 7. Suite d’évaluation RAG (optionnelle)

Script :

python scripts/rag_eval_suite.py


Permet :

tests automatiques

comparer pipelines

mesurer régressions

valider qualité des réponses

🛠 8. Résolution des problèmes
❌ index.faiss not found

→ reconstruire :

python scripts/build_faiss_index.py

❌ corpus_chunks.jsonl not found

→ reconstruire :

python scripts/build_corpus_jsonl.py

❌ Ollama refuse la requête

→ vérifier :

ollama list
ollama pull llama3

❌ CUDA OOM

→ diminuer batch_size dans embedder.py
→ fermer Chrome / TouchDesigner / Resolve

🏛️ 9. Bonnes pratiques de recherche médiévale
✔ Citer systématiquement un lieu ou un acteur

FAISS adore ça.

✔ Mentionner le IXᵉ ou Xᵉ siècle

Les années min/max améliorent le score.

✔ Éviter les questions vagues

FAISS ne peut rien matcher avec “parle-moi d’Aurillac”.

✔ Préférer les formulations :

“Quels acteurs”

“Quelle place occupe”

“Quels éléments montrent que”

✔ Pour la macro-histoire → top-k élevé

migrations

tendances sociales

réseaux aristocratiques

✔ Pour la micro-histoire → top-k bas

un acte

un évêque

un monastère précis

🏁 Fin du GUIDE RAG