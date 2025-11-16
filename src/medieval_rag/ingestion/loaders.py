from pathlib import Path
from typing import Dict, List, Iterator
import PyPDF2


def list_pdf_documents(local_pdf_dir: Path, gallica_pdf_dir: Path) -> List[Dict]:
    """
    Liste les PDFs à traiter (local + Gallica).
    """
    documents = []

    if local_pdf_dir.exists():
        for pdf_path in sorted(local_pdf_dir.glob("*.pdf")):
            documents.append({
                "doc_id": pdf_path.stem,
                "title": pdf_path.stem,
                "source": "local",
                "ark": None,
                "path": pdf_path,
            })

    if gallica_pdf_dir.exists():
        for pdf_path in sorted(gallica_pdf_dir.glob("*.pdf")):
            documents.append({
                "doc_id": pdf_path.stem,
                "title": pdf_path.stem,
                "source": "gallica",
                "ark": None,
                "path": pdf_path,
            })

    return documents


def extract_pdf_text_by_page(pdf_path: Path) -> List[str]:
    """
    Extrait le texte d'un PDF, page par page.
    """
    pages_text: List[str] = []
    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages_text.append(text)
    return pages_text


def iter_documents(local_pdf_dir: Path, gallica_pdf_dir: Path) -> Iterator[Dict]:
    """
    Génère :
    {
      "doc_id": ...,
      "title": ...,
      "source": ...,
      "ark": ...,
      "pages_text": [...],
    }
    """
    docs = list_pdf_documents(local_pdf_dir, gallica_pdf_dir)
    for doc in docs:
        pages_text = extract_pdf_text_by_page(doc["path"])
        yield {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "source": doc["source"],
            "ark": doc["ark"],
            "pages_text": pages_text,
        }
