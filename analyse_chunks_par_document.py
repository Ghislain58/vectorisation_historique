# historical_vectorizer.py
# Vectoriseur de documents historiques — robuste & validé pour ChromaDB

from __future__ import annotations

import re
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

try:
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:
    convert_from_path = None
    pytesseract = None


# ====================================
#   CONFIG GÉNÉRALE & STRUCTURES
# ====================================

@dataclass
class CleaningStats:
    original_length: int
    cleaned_length: int
    reduction_percent: float


class HistoricalDocVectorizer:
    """
    Pipeline robuste :
    - Lecture PDF (texte) + fallback OCR (pdf2image + Tesseract FRA/LAT)
    - Nettoyage intelligent (biblio, en-têtes, TDM, erreurs OCR)
    - Splitters spécialisés (académique/charte/cartulaire)
    - Métadonnées enrichies (langue, notes, institution, actes…)
    - Sanitisation & validation stricte des métadonnées (Chroma-safe)
    - Mode incrémental (registre hash chemin)
    """

    _ALLOWED_META_TYPES = (str, int, float, bool)

    def __init__(
        self,
        base_path: Path,
        db_path: Path,
        collection_name: str,
        registry_path: Path,
        process_footnotes: bool = True,
        enable_text_cleaning: bool = True,
        ocr_enable: bool = True,
        ocr_lang_default: str = "fra+lat",
        max_json_meta_len: int = 5_000,
    ) -> None:
        self.base_path = base_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.registry_path = registry_path
        self.process_footnotes = process_footnotes
        self.enable_text_cleaning = enable_text_cleaning
        self.ocr_enable = ocr_enable
        self.ocr_lang_default = ocr_lang_default
        self.max_json_meta_len = max_json_meta_len

        # Device
        if torch.cuda.is_available():
            self.device = "cuda"
            print("✅ GPU détecté :", torch.cuda.get_device_name(0))
            print(
                "   VRAM disponible :",
                torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "GB",
            )
        else:
            self.device = "cpu"
            print("⚠️  GPU non détecté, utilisation du CPU.")

        # Modèle d'embedding
        print("📥 Chargement du modèle : intfloat/multilingual-e5-large")
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-large", device=self.device
        )

        # Client Chroma
        self.client = PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(self.collection_name)

        # Splitters
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", ".", ";", " "],
            length_function=len,
        )
        self.charter_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", ".", ";", " "],
            length_function=len,
        )
        self.cartulary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n\n", "\n\n", "\n", ". ", ".", " "],
            length_function=len,
        )

        # Stats
        self.stats: Dict[str, Any] = {
            "files_processed": 0,
            "total_chunks": 0,
            "failed_files": [],
            "text_pdfs": 0,
            "ocr_processed": 0,
            "cleaning_stats": [],
        }

        # Registre incrémental
        self.registry = self._load_registry(self.registry_path)

    # ====================================
    #   REGISTRE HASH FICHIERS
    # ====================================

    def _load_registry(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_registry(self, path: Path, data: Dict[str, Any]) -> None:
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Impossible d'écrire le registre {path}: {e}")

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ====================================
    #   OCR & PDF
    # ====================================

    def _is_likely_scanned_pdf(self, text: str, num_pages: int) -> bool:
        if num_pages == 0:
            return False
        avg_len = len(text) / num_pages
        if avg_len < 300:
            return True
        printable_chars = sum(c.isprintable() for c in text)
        if len(text) > 0 and (printable_chars / len(text)) < 0.5:
            return True
        return False

    def _ocr_pdf(self, pdf_path: Path, lang_hint: Optional[str] = None) -> str:
        if not self.ocr_enable:
            return ""
        if convert_from_path is None or pytesseract is None:
            print("❌ OCR indisponible (pdf2image/pytesseract manquants).")
            return ""
        lang = lang_hint or self.ocr_lang_default
        print(f"   🧠 OCR en cours ({lang}) sur {pdf_path.name}…")

        try:
            images = convert_from_path(str(pdf_path))
        except Exception as e:
            print(f"   ❌ Erreur pdf2image : {e}")
            return ""

        texts: List[str] = []
        for i, img in enumerate(images, 1):
            try:
                txt = pytesseract.image_to_string(img, lang=lang)
            except Exception as e:
                print(f"   ⚠️ Erreur OCR page {i}: {e}")
                txt = ""
            texts.append(txt)
        full_text = "\n\n".join(texts)
        self.stats["ocr_processed"] += 1
        return full_text

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            reader = PdfReader(str(pdf_path))
            text = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                text.append(t)
            s = "\n".join(text).strip()
            if not s:
                print(f"   ⚠️  Pas de texte natif → OCR")
                return self._ocr_pdf(pdf_path)
            if self._is_likely_scanned_pdf(s, len(reader.pages)):
                print("   ⚠️  PDF scanné détecté → OCR")
                lang_hint = (
                    "lat+fra"
                    if ("cartul" in str(pdf_path).lower() or "charte" in str(pdf_path).lower())
                    else self.ocr_lang_default
                )
                return self._ocr_pdf(pdf_path, lang_hint=lang_hint)
            self.stats["text_pdfs"] += 1
            return s
        except Exception as e:
            print(f"❌ Erreur lecture PDF {pdf_path.name} : {e}")
            self.stats["failed_files"].append(str(pdf_path))
            return ""

    def extract_text_from_txt(self, txt_path: Path) -> str:
        encodings = ["utf-8", "cp1252", "latin-1", "iso-8859-1"]
        for enc in encodings:
            try:
                return txt_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return txt_path.read_text(encoding="utf-8", errors="ignore")

    # ====================================
    #   NETTOYAGE TEXTE
    # ====================================

    def _remove_bibliography_section(self, text: str) -> str:
        patterns = [
            r"\n\s*BIBLIOGRAPHIE\s*\n",
            r"\n\s*BIBLIOGRAPHIE GÉNÉRALE\s*\n",
            r"\n\s*SOURCES ET BIBLIOGRAPHIE\s*\n",
        ]
        upper = text.upper()
        idx = None
        for pat in patterns:
            m = re.search(pat, upper)
            if m:
                idx = m.start()
                break
        if idx is None:
            return text
        return text[:idx]

    def _remove_headers_footers(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        for ln in lines:
            s = ln.strip()
            if re.fullmatch(r"\d{1,4}", s):
                continue
            if len(s) <= 40 and s.isupper():
                continue
            cleaned.append(ln)
        return "\n".join(cleaned)

    def _remove_table_of_contents(self, text: str) -> str:
        markers = [r"\n\s*TABLE DES MATI[ÈE]RES\s*\n", r"\n\s*SOMMAIRE\s*\n"]
        upper = text.upper()
        start = None
        for pat in markers:
            m = re.search(pat, upper)
            if m:
                start = m.start()
                break
        if start is None:
            return text
        intro_m = re.search(r"\n\s*INTRODUCTION\s*\n", upper[start:])
        if intro_m:
            end = start + intro_m.start()
            return text[:start] + text[end:]
        return text

    def _fix_ocr_errors(self, text: str) -> str:
        repl = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "’": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
        }
        for k, v in repl.items():
            text = text.replace(k, v)
        text = re.sub(r"\s+", " ", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def clean_text_intelligently(self, text: str, document_type: str) -> Tuple[str, CleaningStats]:
        original = len(text)
        if document_type == "academic":
            text = self._remove_bibliography_section(text)
            text = self._remove_headers_footers(text)
            text = self._remove_table_of_contents(text)
            text = self._fix_ocr_errors(text)
            text = self._normalize_whitespace(text)
        elif document_type in ("charter", "cartulary"):
            text = self._fix_ocr_errors(text)
            text = self._normalize_whitespace(text)
        cleaned = len(text)
        stats = CleaningStats(
            original_length=original,
            cleaned_length=cleaned,
            reduction_percent=(1 - cleaned / original) * 100 if original > 0 else 0.0,
        )
        return text, stats

    # ===========================
    #   NOTES / LANGUE / CHARTES
    # ===========================

    def extract_and_integrate_footnotes(self, text: str) -> Tuple[str, Dict[str, Any]]:
        notes_markers = [r"\n\s*(NOTES?|Notes de bas de page|Footnotes?|References)\s*\n"]
        pos = None
        for pat in notes_markers:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                pos = m.start()
                break
        if pos is None:
            return text, {"has_footnotes": False, "count": 0}
        return text, {"has_footnotes": True, "count": text[pos:].count("\n")}

    def detect_charter_language(self, text: str) -> Dict[str, Any]:
        latin_markers = ["quod", "ecclesia", "donatio", "terra", "dominus", "episcopus"]
        french_markers = ["seigneur", "paroisse", "donation", "terre", "église", "chapitre"]
        words = re.findall(r"[a-zA-Zéèêàùûôâîüï]+", text.lower())
        if not words:
            return {"latin": 0.0, "french": 0.0, "mixed": False, "dominant": None}
        total = len(words)
        latin = sum(1 for w in words if w in latin_markers)
        french = sum(1 for w in words if w in french_markers)
        lr, fr = latin / total, french / total
        return {
            "latin": lr,
            "french": fr,
            "mixed": (lr > 0.1 and fr > 0.1),
            "dominant": "latin" if lr > fr else "french",
        }

    def preprocess_charter(self, text: str) -> str:
        repl = {
            "ſ": "s",
            "æ": "ae",
            "œ": "oe",
            "ẽ": "em",
            "ã": "an",
            "õ": "on",
            "ꝰ": "us",
            "ꝑ": "per",
            "ꝓ": "pro",
            "ꝙ": "que",
            "&": "et",
        }
        for k, v in repl.items():
            text = text.replace(k, v)
        return text

    def preprocess_cartulary(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        lines = text.splitlines()
        acts = []
        current_num = 0
        for i, ln in enumerate(lines):
            if re.search(r"\b(N[°o]\s*\d+|ACTE\s+\d+)\b", ln, flags=re.IGNORECASE):
                current_num += 1
                acts.append({"position": sum(len(l) + 1 for l in lines[:i]), "number": current_num})
        return text, acts

    def detect_cartulary_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        lower = text.lower()
        name = filename.lower()
        institution = None
        if "brioude" in lower or "brioude" in name:
            institution = "brioude"
        elif "chaise-dieu" in lower or "chaise dieu" in lower or "casa dei" in lower:
            institution = "chaise-dieu"
        elif "clermont" in lower or "arverne" in lower:
            institution = "clermont/arverne"
        elif "velay" in lower:
            institution = "velay"
        elif "savigny" in lower:
            institution = "savigny"

        acts = re.findall(r"\bACTE\s+\d+\b|\bN[°o]\s*\d+\b", text.upper())
        return {
            "institution": institution,
            "total_acts_estimated": len(acts),
            "date_range": None,
        }

    # ====================================
    #   SANITISATION & VALIDATION META
    # ====================================

    def _sanitize_value(self, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, self._ALLOWED_META_TYPES):
            return v
        if isinstance(v, (list, dict)):
            try:
                s = json.dumps(v, ensure_ascii=False)
            except TypeError:
                s = str(v)
            if len(s) > self.max_json_meta_len:
                s = s[: self.max_json_meta_len] + "…"
            return s
        return str(v)

    def _sanitize_metadatas_list(self, metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for md in metadatas:
            clean_md: Dict[str, Any] = {}
            for k, v in md.items():
                sv = self._sanitize_value(v)
                if sv is None:
                    continue  # supprime les None → évite erreurs Chroma
                clean_md[k] = sv
            sanitized.append(clean_md)
        return sanitized

    def _validate_metadatas(self, metadatas: List[Dict[str, Any]]) -> None:
        for md in metadatas:
            for k, v in md.items():
                if v is None:
                    continue
                if not isinstance(v, self._ALLOWED_META_TYPES):
                    raise TypeError(
                        f"Valeur de métadonnée invalide pour '{k}': {type(v)} ({v})"
                    )

    # ====================================
    #   ENCODE PAR PAQUETS + FALLBACK
    # ====================================

    def _encode_chunks_batched(self, chunks: List[str]) -> np.ndarray:
        """
        Encode les chunks par paquets pour éviter l'erreur
        'Batch size > max batch size'. Si l'encode global plante,
        on passe en mode debug chunk par chunk.
        """
        all_embeddings: List[np.ndarray] = []
        max_batch_texts = 2000  # nombre de textes max par appel encode

        for start in range(0, len(chunks), max_batch_texts):
            batch_texts = chunks[start : start + max_batch_texts]
            try:
                emb = self.model.encode(
                    batch_texts,
                    batch_size=16,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as e:
                print(f"   ❌ Erreur encode sur batch [{start}:{start+len(batch_texts)}]: {e}")
                # Fallback : test un par un, on garde seulement ceux qui passent
                good_vecs: List[np.ndarray] = []
                for i, ch in enumerate(batch_texts):
                    try:
                        v = self.model.encode(
                            [ch],
                            batch_size=1,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                        good_vecs.append(v[0])
                    except Exception as e2:
                        print(f"      ⚠️ Chunk global index {start+i} invalide: {e2}")
                if not good_vecs:
                    continue
                emb = np.vstack(good_vecs)

            all_embeddings.append(emb)

        if not all_embeddings:
            raise RuntimeError("Aucun embedding valide produit.")
        return np.vstack(all_embeddings)

    # ====================================
    #   TRAITEMENT D’UN DOCUMENT
    # ====================================

    def process_document(self, file_path: Path) -> int:
        # 1) Extraction
        if file_path.suffix.lower() == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() in (".txt", ".md"):
            text = self.extract_text_from_txt(file_path)
        else:
            print(f"⏭️  Format non supporté : {file_path.suffix}")
            return 0

        if not text or len(text) < 100:
            print(f"⚠️  Document vide ou trop court : {file_path.name}")
            return 0

        # 2) Type de doc & folder
        parts_lower = [p.lower() for p in file_path.parts]
        is_cartulary = any("cartul" in p for p in parts_lower)
        is_charter = any("charte" in p or "charter" in p for p in parts_lower) and not is_cartulary
        is_academic = not (is_charter or is_cartulary)

        folder: Optional[str] = None
        try:
            idx = parts_lower.index("mes_documents_historiques")
            if idx + 1 < len(file_path.parts):
                folder = file_path.parts[idx + 1]
        except ValueError:
            folder = file_path.parent.name

        # 3) Nettoyage intelligent avec garde-fou + bypass
        filename_lower = file_path.name.lower()
        bypass_cleaning_files = {
            "berger_jean_these_compact.pdf",
        }
        bypass_cleaning = filename_lower in bypass_cleaning_files

        if self.enable_text_cleaning and not bypass_cleaning:
            raw_text = text
            if is_academic:
                cleaned_text, cstats = self.clean_text_intelligently(text, "academic")
            elif is_charter:
                cleaned_text, cstats = self.clean_text_intelligently(text, "charter")
            else:
                cleaned_text, cstats = self.clean_text_intelligently(text, "cartulary")

            if cstats.original_length > 10_000 and cstats.reduction_percent > 50.0:
                print(
                    f"   ⚠️ Nettoyage trop agressif sur {file_path.name} "
                    f"({cstats.original_length} → {cstats.cleaned_length} caractères, "
                    f"-{cstats.reduction_percent:.1f}%). Texte brut conservé."
                )
                text = raw_text
            else:
                text = cleaned_text
                self.stats["cleaning_stats"].append(
                    {"file": file_path.name, "stats": cstats.__dict__}
                )

        # 4) Notes (académique)
        foot_meta = {"has_footnotes": False, "count": 0}
        if is_academic and self.process_footnotes:
            text, foot_meta = self.extract_and_integrate_footnotes(text)

        # 5) Prétraitement & splitters
        cart_meta: Optional[Dict[str, Any]] = None
        acts_info: Optional[List[Dict[str, Any]]] = None
        if is_cartulary:
            text, acts_info = self.preprocess_cartulary(text)
            cart_meta = self.detect_cartulary_metadata(text, file_path.name)
            lang_info = self.detect_charter_language(text[:5000])
            splitter = self.cartulary_splitter
        elif is_charter:
            text = self.preprocess_charter(text)
            lang_info = self.detect_charter_language(text)
            splitter = self.charter_splitter
        else:
            lang_info = None
            splitter = self.splitter

        chunks = splitter.split_text(text)
        if not chunks:
            return 0

        # 6) Nettoyage final des chunks (str + non vides)
        cleaned_chunks: List[str] = []
        for idx, ch in enumerate(chunks):
            if not isinstance(ch, str):
                try:
                    ch = str(ch)
                except Exception:
                    print(
                        f"   ⚠️ Chunk {idx} ignoré (type invalide: {type(ch)}) dans {file_path.name}"
                    )
                    continue
            ch = ch.strip()
            if not ch:
                continue
            cleaned_chunks.append(ch)
        chunks = cleaned_chunks
        if not chunks:
            print(f"⚠️  Plus aucun chunk exploitable après nettoyage pour {file_path.name}")
            return 0

        # 7) Embeddings en paquets (avec fallback intégré)
        embeddings = self._encode_chunks_batched(chunks)

        # 8) IDs & métadonnées
        ids = [
            f"{file_path.stem}__{i:06d}__{abs(hash(ch)) % 100000000}"
            for i, ch in enumerate(chunks)
        ]

        metadatas: List[Dict[str, Any]] = []
        for i, ch in enumerate(chunks):
            md: Dict[str, Any] = {
                "source": str(file_path.resolve()),
                "filename": file_path.name,
                "folder": folder,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "file_type": file_path.suffix[1:].lower(),
                "is_charter": is_charter,
                "is_cartulary": is_cartulary,
                "is_academic": is_academic,
            }

            if is_academic and foot_meta.get("has_footnotes"):
                nums = re.findall(r"\[Note (\d+):", ch)
                md.update(
                    {
                        "has_footnotes": bool(nums),
                        "footnote_count": len(nums),
                        "footnote_numbers": nums if nums else None,
                    }
                )

            if (is_charter or is_cartulary) and lang_info:
                md.update(
                    {
                        "latin_ratio": float(lang_info["latin"]),
                        "french_ratio": float(lang_info["french"]),
                        "is_mixed": bool(lang_info["mixed"]),
                        "dominant_lang": str(lang_info["dominant"])
                        if lang_info["dominant"]
                        else None,
                    }
                )

            if is_cartulary and cart_meta:
                md.update(
                    {
                        "institution": cart_meta.get("institution"),
                        "estimated_total_acts": int(
                            cart_meta.get("total_acts_estimated", 0)
                        ),
                        "date_range": cart_meta.get("date_range") or None,
                    }
                )
                if acts_info:
                    chunk_start = sum(len(c) for c in chunks[:i])
                    relevant = [a for a in acts_info if a["position"] <= chunk_start]
                    if relevant:
                        md["act_number"] = str(relevant[-1]["number"])

            metadatas.append(md)

        # 9) SANITIZE -> VALIDATE -> ADD
        metadatas = self._sanitize_metadatas_list(metadatas)
        self._validate_metadatas(metadatas)

        if not (
            len(chunks) == len(embeddings) == len(ids) == len(metadatas)
        ):
            raise ValueError(
                f"Incohérence longueurs: chunks={len(chunks)} embeddings={len(embeddings)} "
                f"ids={len(ids)} metadatas={len(metadatas)}"
            )

        try:
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                ids=[str(x) for x in ids],
                metadatas=metadatas,
            )
        except TypeError:
            print(
                "❌ TypeError à l'insertion Chroma. Exemple de métadonnée envoyée :"
            )
            print(json.dumps(metadatas[0], ensure_ascii=False, indent=2))
            raise

        return len(chunks)

    # ====================================
    #   TRAITEMENT D’UN DOSSIER
    # ====================================

    def process_folder(self) -> None:
        all_files: List[Path] = []
        for ext in ("*.pdf", "*.txt", "*.md"):
            all_files.extend(self.base_path.rglob(ext))

        print(f"📚 {len(all_files)} fichiers trouvés")
        print(f"📂 Extensions: .pdf, .txt, .md")

        for f in tqdm(all_files, desc="Vectorisation", unit="fichier"):
            try:
                file_hash = self._hash_file(f)
            except Exception as e:
                print(f"⚠️  Impossible de hasher {f.name}: {e}")
                self.stats["failed_files"].append(str(f))
                continue

            reg_entry = self.registry.get(str(f))
            if reg_entry and reg_entry.get("hash") == file_hash:
                tqdm.write(f"⏭️  Déjà à jour : {f.name}")
                continue

            try:
                n = self.process_document(f)
            except Exception as e:
                tqdm.write(f"❌ Erreur sur {f.name}: {e}")
                self.stats["failed_files"].append(str(f))
                continue

            if n > 0:
                self.stats["files_processed"] += 1
                self.stats["total_chunks"] += n
                tqdm.write(f"✓ {f.name}: {n} chunks")

                self.registry[str(f)] = {"hash": file_hash, "chunks": n}
                self._save_registry(self.registry_path, self.registry)

        self._print_stats()

    def _print_stats(self) -> None:
        print("\n===== STATISTIQUES =====")
        print("Fichiers traités :", self.stats["files_processed"])
        print("Chunks totaux    :", self.stats["total_chunks"])
        print("PDF texte natifs :", self.stats["text_pdfs"])
        print("PDF passés en OCR:", self.stats["ocr_processed"])
        if self.stats["failed_files"]:
            print("Fichiers en erreur :")
            for f in self.stats["failed_files"]:
                print("  -", f)
        print("========================\n")


def main():
    base = Path("mes_documents_historiques")
    vec = HistoricalDocVectorizer(
        base_path=base,
        db_path=Path("./vector_db_historique"),
        collection_name="documents_historiques",
        registry_path=Path("documents_historiques_registry.json"),
        process_footnotes=True,
        enable_text_cleaning=True,
    )
    vec.process_folder()


if __name__ == "__main__":
    main()
