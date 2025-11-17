#!/usr/bin/env python3
"""
Vectorisation de documents historiques
- Extraction texte depuis PDF ou TXT
- Nettoyage minimal
- Encodage avec SentenceTransformers
- Stockage vectoriel local (ChromaDB, nouvelle API)
"""

import os
import PyPDF2
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = os.path.expanduser("~/vectorisation_historique")
DOCS_DIR = os.path.join(BASE_DIR, "mes_documents_historiques")
DB_DIR = os.path.join(BASE_DIR, "vector_db_historique")
MODEL_NAME = "intfloat/multilingual-e5-large"  # bon équilibre qualité/vitesse

# --- Initialisation du modèle et de la base vectorielle ---
print("🚀 Chargement du modèle d'embedding...")
model = SentenceTransformer(MODEL_NAME, device="cuda")

client = PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("historical_docs")

# --- Fonctions utilitaires ---
def extract_text_from_pdf(pdf_path):
    """Extraction du texte brut depuis un PDF."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"⚠️  Erreur lors de la lecture de {pdf_path}: {e}")
    return text.strip()

def get_documents_texts():
    """Retourne la liste de tous les fichiers PDF et TXT dans le corpus."""
    docs = []
    for root, _, files in os.walk(DOCS_DIR):
        for f in files:
            if f.lower().endswith(".pdf") or f.lower().endswith(".txt"):
                docs.append(os.path.join(root, f))
    return docs

# --- Vectorisation principale ---
docs = get_documents_texts()
print(f"📚 {len(docs)} documents détectés dans {DOCS_DIR}")

if not docs:
    print("⚠️  Aucun fichier PDF ou TXT trouvé.")
    print("   Place tes fichiers dans : mes_documents_historiques/")
    exit()

for i, doc_path in enumerate(tqdm(docs, desc="Vectorisation")):
    # Lecture du contenu texte
    if doc_path.endswith(".pdf"):
        text = extract_text_from_pdf(doc_path)
    else:
        try:
            with open(doc_path, "r", encoding="utf-8") as fp:
                text = fp.read()
        except Exception as e:
            print(f"⚠️  Erreur lecture {doc_path}: {e}")
            continue

    if not text.strip():
        print(f"⚠️  Aucun texte détecté dans {doc_path}")
        continue

    # Encodage et ajout à la base
    embedding = model.encode([text], convert_to_numpy=True)[0]
    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"path": doc_path}],
        ids=[f"doc_{i}"]
    )

# Sauvegarde de la base Chroma
print("\n💾 Sauvegarde de la base vectorielle...")
client.persist()

print(f"\n✅ Vectorisation terminée.")
print(f"📂 Base enregistrée dans : {DB_DIR}")
