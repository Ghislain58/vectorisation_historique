from typing import Dict, List


def chunk_document(
    doc: Dict,
    max_chars: int = 1800,
    overlap_chars: int = 200
) -> List[Dict]:
    """
    Découpe un document en chunks de texte avec recouvrement.

    doc attendu :
      {
        "doc_id": str,
        "title": str,
        "source": "local" | "gallica",
        "ark": str | None,
        "pages_text": [str, str, ...]
      }

    Retour :
      [
        {
          "text": str,
          "page_start": int,  # 1-based
          "page_end": int,    # 1-based
        },
        ...
      ]
    """
    chunks: List[Dict] = []

    pages_text = doc.get("pages_text", [])
    if not pages_text:
        return chunks

    current_text_parts: List[str] = []
    current_len = 0
    current_start_page = 0  # index 0 => page 1

    for page_idx, page_text in enumerate(pages_text):
        if not page_text:
            continue

        # Nettoyage simple
        page_text = page_text.replace("\r", " ").replace("\n", " ")
        page_text = " ".join(page_text.split())

        if current_len == 0:
            current_start_page = page_idx

        for token in page_text.split(" "):
            token_len = len(token) + 1
            if current_len + token_len > max_chars:
                # On ferme le chunk courant
                chunk_text = " ".join(current_text_parts).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "page_start": current_start_page + 1,
                        "page_end": page_idx + 1,
                    })

                # Recouvrement
                if overlap_chars > 0:
                    overlap_text = chunk_text[-overlap_chars:]
                    current_text_parts = [overlap_text]
                    current_len = len(overlap_text)
                    current_start_page = max(current_start_page, page_idx)
                else:
                    current_text_parts = []
                    current_len = 0
                    current_start_page = page_idx
            else:
                current_text_parts.append(token)
                current_len += token_len

    # Dernier chunk
    if current_text_parts:
        chunk_text = " ".join(current_text_parts).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page_start": current_start_page + 1,
                "page_end": len(pages_text),
            })

    return chunks
