# historical_vectorizer.py
# Vectoriseur de documents historiques
# - Extraction PDF/TXT
# - Nettoyage (en-têtes, bibliographie, TDM, erreurs OCR)
# - Découpage en chunks
# - Lexique d'entités (autres/) + annotation des chunks
# - Vectorisation avec intfloat/multilingual-e5-large (E5)
# - Stockage dans ChromaDB (PersistentClient)

from __future__ import annotations

import re
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from chromadb import PersistentClient
from lingua import Language, LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

try:
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:  # OCR optionnel
    convert_from_path = None
    pytesseract = None


# ====================================
#   STRUCTURES
# ====================================

@dataclass
class CleaningStats:
    original_length: int
    cleaned_length: int
    reduction_percent: float


class HistoricalDocVectorizer:
    """Pipeline de vectorisation pour corpus historiques."""

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
        debug: bool = False,
        debug_dir: Optional[Path] = None,
    ) -> None:
        self.base_path = base_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.registry_path = registry_path
        self.process_footnotes = process_footnotes
        self.enable_text_cleaning = enable_text_cleaning
        self.ocr_enable = ocr_enable
        self.ocr_lang_default = ocr_lang_default
        self.debug = debug
        self.debug_dir = debug_dir or Path("debug_vectorizer")

        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

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

        # Détection de langue globale (FR/EN/DE/IT) avec lingua
        self.languages = [Language.FRENCH, Language.ENGLISH, Language.GERMAN, Language.ITALIAN]
        self.lang_detector = LanguageDetectorBuilder.from_languages(*self.languages).build()

        # spaCy FR optionnel pour une couche NER supplémentaire
        try:
            import spacy  # type: ignore
            try:
                self.nlp_fr = spacy.load("fr_core_news_lg")
            except OSError:
                self.nlp_fr = None
                print(
                    "⚠️  Modèle spaCy 'fr_core_news_lg' introuvable. "
                    "Installe-le avec : python -m spacy download fr_core_news_lg"
                )
        except ImportError:
            self.nlp_fr = None
            print("⚠️  spaCy non disponible, analyse d'entités FR désactivée.")

        # Modèle d'embedding
        print("📥 Chargement du modèle : intfloat/multilingual-e5-large")
        self.model = SentenceTransformer("intfloat/multilingual-e5-large", device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

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
            "processed_files": 0,
            "skipped_files": 0,
            "failed_files": [],
            "total_chunks": 0,
            "by_folder": {},
            "by_type": {},
            "text_pdfs": 0,
            "ocr_processed": 0,
            "cleaning_stats": [],
        }

        # Registre incrémental
        self.registry = self._load_registry(self.registry_path)

        # Lexique d'entités (v2 si présent, sinon construit depuis "autres")
        self.entity_lexicon = self._load_entity_lexicon()

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
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except Exception as e:
            print(f"⚠️  Impossible de hasher {path}: {e}")
        return h.hexdigest()

    def _is_already_indexed(self, file_path: Path) -> bool:
        h = self._hash_file(file_path)
        entry = self.registry.get(str(file_path))
        return bool(entry and entry.get("hash") == h)

    def _update_registry_for_file(self, file_path: Path, nb_chunks: int) -> None:
        h = self._hash_file(file_path)
        self.registry[str(file_path)] = {
            "hash": h,
            "chunks": nb_chunks,
        }
        self._save_registry(self.registry_path, self.registry)

    # ====================================
    #   EXTRACTION TEXTE
    # ====================================

    def _pdf_is_scanned(self, pdf_path: Path) -> bool:
        """Heuristique simple : peu ou pas de texte, mais des pages."""
        try:
            reader = PdfReader(str(pdf_path))
            text_len = 0
            for p in reader.pages[:5]:
                txt = p.extract_text() or ""
                text_len += len(txt.strip())
            if len(reader.pages) > 0 and text_len < 200:
                return True
        except Exception:
            return False
        return False

    def _ocr_pdf(self, pdf_path: Path, lang_hint: str = "fra+lat") -> str:
        """OCR d'un PDF scanné, si pdf2image/pytesseract disponibles."""
        if convert_from_path is None or pytesseract is None:
            print(f"⚠️  OCR indisponible (pdf2image/pytesseract manquants) pour {pdf_path.name}")
            return ""
        try:
            pages = convert_from_path(str(pdf_path))
        except Exception as e:
            print(f"❌ Erreur pdf2image sur {pdf_path.name}: {e}")
            return ""
        text_parts: List[str] = []
        for img in tqdm(pages, desc=f"OCR {pdf_path.name}"):
            try:
                t = pytesseract.image_to_string(img, lang=lang_hint)
                text_parts.append(t)
            except Exception as e:
                print(f"⚠️  Erreur OCR page: {e}")
        full_text = "\n".join(text_parts)
        self.stats["ocr_processed"] += 1
        return full_text

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extraction texte : si très peu de texte → tentative OCR (si activée)."""
        try:
            reader = PdfReader(str(pdf_path))
            texts = []
            for p in reader.pages:
                t = p.extract_text() or ""
                texts.append(t)
            s = "\n".join(texts)
            if not s or len(s.strip()) < 200:
                if not self.ocr_enable:
                    print(
                        f"⚠️  PDF probablement scanné mais OCR désactivé : {pdf_path.name} "
                        "(texte très court)."
                    )
                    return s
                print(f"   ⚠️  PDF scanné détecté → OCR ({pdf_path.name})")
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
        try:
            return txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # fallback
            try:
                return txt_path.read_text(encoding="latin-1")
            except UnicodeDecodeError:
                return txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                return txt_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""

    # ====================================
    #   NETTOYAGE TEXTE
    # ====================================

    def _remove_bibliography_section(self, text: str) -> str:
        patterns = [
            r"\n\s*BIBLIOGRAPHIE\s*\n",
            r"\n\s*BIBLIOGRAPHIE GÉNÉRALE\s*\n",
            r"\n\s*RÉFÉRENCES BIBLIOGRAPHIQUES\s*\n",
        ]
        lowest_index = None
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                if lowest_index is None or m.start() < lowest_index:
                    lowest_index = m.start()
        if lowest_index is not None and lowest_index > len(text) * 0.4:
            return text[:lowest_index]
        return text

    def _remove_table_of_contents(self, text: str) -> str:
        m = re.search(r"\n\s*TABLE DES MATIÈRES\s*\n", text, flags=re.IGNORECASE)
        if not m:
            return text
        pos = m.start()
        end_m = re.search(
            r"\n\s*(INTRODUCTION|PREMIÈRE PARTIE|CHAPITRE 1)\b",
            text[pos:],
            flags=re.IGNORECASE,
        )
        if not end_m:
            return text
        end_pos = pos + end_m.start()
        return text[:pos] + text[end_pos:]

    def _remove_headers_and_footers(self, text: str) -> str:
        lines = text.splitlines()
        cleaned_lines: List[str] = []
        for ln in lines:
            ln_stripped = ln.strip()
            if re.match(r"^\d+$", ln_stripped):
                continue
            if len(ln_stripped) < 5 and ("[" in ln_stripped or "]" in ln_stripped):
                # en-têtes de notes ou numérotation bricolée
                pass
            cleaned_lines.append(ln)
        return "\n".join(cleaned_lines)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def clean_text_intelligently(self, text: str, doc_type: str) -> Tuple[str, CleaningStats]:
        original = len(text)
        if doc_type == "academic":
            text = self._remove_table_of_contents(text)
            text = self._remove_bibliography_section(text)
        text = self._remove_headers_and_footers(text)
        text = self._normalize_whitespace(text)
        cleaned = len(text)
        stats = CleaningStats(
            original_length=original,
            cleaned_length=cleaned,
            reduction_percent=(1 - cleaned / original) * 100 if original > 0 else 0.0,
        )
        return text, stats

    # ====================================
    #   NETTOYAGE UNICODE (surrogates)
    # ====================================

    def _strip_surrogates(self, text: str) -> str:
        # supprime les code points surrogates (U+D800–U+DFFF)
        return text.translate({cp: None for cp in range(0xD800, 0xE000)})

    # ====================================
    #   NOTES / LANGUE / CARTULAIRES
    # ====================================

    def extract_and_integrate_footnotes(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Extrait les notes de bas de page simples type [Note X: ...] et les réintègre."""
        m = re.search(r"\n\s*NOTES?\s*\n", text, flags=re.IGNORECASE)
        if not m:
            return text, {"has_footnotes": False, "count": 0}
        pos = m.start()
        main_text = text[:pos]
        notes_text = text[pos:]

        notes = re.findall(r"\[Note (\d+):(.+?)\]", notes_text, flags=re.DOTALL)
        if not notes:
            return text, {"has_footnotes": False, "count": 0}

        notes_map = {int(num): content.strip() for num, content in notes}

        def replace_marker(match: re.Match) -> str:
            num_str = match.group(1)
            try:
                num = int(num_str)
            except ValueError:
                return match.group(0)
            note = notes_map.get(num)
            if not note:
                return match.group(0)
            return f"[Note {num}: {note}]"

        main_text = re.sub(r"\[Note (\d+)\]", replace_marker, main_text)
        return main_text, {"has_footnotes": True, "count": len(notes)}

    def detect_language(self, text: str) -> str:
        """Détecte grossièrement la langue dominante d'un texte sur un échantillon."""
        sample = text[:5000]
        try:
            lang = self.lang_detector.detect_language_of(sample)
            return lang.iso_code_639_1.name if lang else "unknown"
        except Exception:
            return "unknown"

    def analyze_chunk_spacy(self, text: str) -> Dict[str, Any]:
        """Analyse optionnelle d'un chunk avec spaCy FR (entités personnes/lieux)."""
        if not getattr(self, "nlp_fr", None):
            return {}
        try:
            doc = self.nlp_fr(text)
        except Exception:
            return {}
        persons = sorted({ent.text for ent in doc.ents if ent.label_ == "PER"})
        places = sorted({ent.text for ent in doc.ents if ent.label_ in ("LOC", "GPE")})
        meta: Dict[str, Any] = {}
        if persons:
            meta["spacy_persons"] = ", ".join(persons)
        if places:
            meta["spacy_places"] = ", ".join(places)
        return meta

    def detect_charter_language(self, text: str) -> Dict[str, Any]:
        latin_markers = ["quod", "ecclesia", "donatio", "terra", "dominus", "episcopus"]
        french_markers = ["seigneur", "paroisse", "donation", "terre", "église", "chapitre"]
        words = re.findall(r"[a-zA-Zéèêàùûôâîüï]+", text.lower())
        if not words:
            return {"latin": 0.0, "french": 0.0, "mixed": False, "dominant": None}
        total = len(words)
        latin = sum(1 for w in words if w in latin_markers)
        french = sum(1 for w in words if w in french_markers)
        return {
            "latin": latin / total,
            "french": french / total,
            "mixed": latin > 0 and french > 0,
            "dominant": "latin" if latin > french else ("french" if french > latin else None),
        }

    def preprocess_charter(self, text: str) -> str:
        text = text.replace("ſ", "s")
        text = re.sub(r"\s+", " ", text)
        return text

    def preprocess_cartulary(self, text: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        acts = []
        pattern = re.compile(r"\n\s*(ACTE|CHARTRE)\s+(\d+)\s*\n", flags=re.IGNORECASE)
        for m in pattern.finditer(text):
            num = int(m.group(2))
            acts.append({"number": num, "position": m.start()})
        return text, acts or None

    def detect_cartulary_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        inst = None
        m = re.search(r"CARTULAIRE\s+DE\s+([A-ZÉÈÎÂÔÛÄËÏÖÜ\s\-]+)", text[:2000], flags=re.IGNORECASE)
        if m:
            inst = m.group(1).strip().title()
        years = re.findall(r"(1[0-9]{3})", text[:5000])
        if years:
            years_int = sorted(int(y) for y in years)
            date_range = f"{years_int[0]}–{years_int[-1]}"
        else:
            date_range = None
        total_acts_estimated = len(
            re.findall(r"\bACTE\s+\d+\b", text, flags=re.IGNORECASE)
        )
        return {
            "institution": inst,
            "date_range": date_range,
            "total_acts_estimated": total_acts_estimated,
        }

    # ====================================
    #   LEXIQUE D'ENTITÉS
    # ====================================

def _load_entity_lexicon(self) -> Dict[str, List[str]]:
    """
    Charge le lexique d'entités avec ordre de priorité :
    1) entity_lexicon_v3.json (lexique nettoyé)
    2) entity_lexicon_v2.json
    3) entity_lexicon.json
    4) reconstruit depuis 'autres' (fallback)
    """
    base_dir = self.base_path / "mes_documents_historiques"
    lex_path_v3 = base_dir / "entity_lexicon_v3.json"
    lex_path_v2 = base_dir / "entity_lexicon_v2.json"
    lex_path_v1 = base_dir / "entity_lexicon.json"

    # ---------------------------
    # 1) Dissertation V3 améliorée
    # ---------------------------
    if lex_path_v3.exists():
        try:
            with lex_path_v3.open("r", encoding="utf-8") as f:
                data = json.load(f)
            persons = data.get("persons", [])
            places = data.get("places", [])
            years = data.get("years", [])
            print(
                f"🧾 Lexique d'entités chargé (entity_lexicon_v3.json) : "
                f"{len(persons)} noms propres, {len(places)} lieux, {len(years)} années."
            )
            return {
                "persons": persons,
                "places": places,
                "years": years,
            }
        except Exception as e:
            print(f"⚠️  Impossible de charger entity_lexicon_v3.json : {e}")

    # ---------------------------
    # 2) Ancienne version : V2
    # ---------------------------
    if lex_path_v2.exists():
        try:
            with lex_path_v2.open("r", encoding="utf-8") as f:
                data = json.load(f)
            persons = data.get("persons", [])
            places = data.get("places", [])
            years = data.get("years", [])
            print(
                f"🧾 Lexique d'entités chargé (entity_lexicon_v2.json) : "
                f"{len(persons)} noms propres, {len(places)} lieux, {len(years)} années."
            )
            return {
                "persons": persons,
                "places": places,
                "years": years,
            }
        except Exception as e:
            print(f"⚠️  Impossible de charger entity_lexicon_v2.json : {e}")

    # ---------------------------
    # 3) Encore plus ancien : V1
    # ---------------------------
    if lex_path_v1.exists():
        try:
            with lex_path_v1.open("r", encoding="utf-8") as f:
                data = json.load(f)
            print(
                f"🧾 Lexique d'entités chargé (entity_lexicon.json) : "
                f"{len(data.get('persons', []))} noms propres, "
                f"{len(data.get('places', []))} lieux, "
                f"{len(data.get('years', []))} années."
            )
            return data
        except Exception as e:
            print(f"⚠️  Impossible de charger entity_lexicon.json : {e}")

    # ---------------------------
    # 4) Aucun lexique → reconstruction automatique
    # ---------------------------
    lexicon = self._build_entity_lexicon_from_autres()
    try:
        out_path = base_dir / "entity_lexicon_v2.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(lexicon, f, ensure_ascii=False, indent=2)
        print(f"🧾 Lexique d'entités construit et sauvegardé vers {out_path}")
    except Exception as e:
        print(f"⚠️  Impossible d'écrire entity_lexicon_v2.json : {e}")

    return lexicon


    def _build_entity_lexicon_from_autres(self) -> Dict[str, List[str]]:
        autres_dir = self.base_path / "mes_documents_historiques" / "autres"
        persons: List[str] = []
        places: List[str] = []
        years: List[str] = []

        if not autres_dir.exists():
            print(f"⚠️  Dossier 'autres' introuvable ({autres_dir}), lexique d'entités minimal.")
            return {"persons": [], "places": [], "years": []}

        for txt in sorted(autres_dir.glob("*.txt")):
            try:
                content = txt.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = txt.read_text(encoding="latin-1", errors="ignore")
            lower_name = txt.name.lower()
            if "nom" in lower_name or "famille" in lower_name:
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        persons.append(line)
            elif "lieux" in lower_name or "toponym" in lower_name:
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        places.append(line)
            elif "annees" in lower_name or "années" in lower_name or "years" in lower_name:
                for line in content.splitlines():
                    line = line.strip()
                    if re.match(r"^\d{3,4}$", line):
                        years.append(line)

        persons = sorted(set(persons))
        places = sorted(set(places))
        years = sorted(set(years))

        print(
            f"🧾 Lexique d'entités construit depuis 'autres' : "
            f"{len(persons)} noms propres, {len(places)} lieux, {len(years)} années."
        )

        return {
            "persons": persons,
            "places": places,
            "years": years,
        }

    def _annotate_chunk_entities(self, text: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if not self.entity_lexicon:
            return meta

        persons = self.entity_lexicon.get("persons", [])
        places = self.entity_lexicon.get("places", [])
        years = self.entity_lexicon.get("years", [])

        found_persons = [p for p in persons if p in text]
        found_places = [p for p in places if p in text]
        found_years = [y for y in years if y in text]

        if found_persons:
            meta["entities_persons"] = ", ".join(sorted(set(found_persons)))
        if found_places:
            meta["entities_places"] = ", ".join(sorted(set(found_places)))
        if found_years:
            meta["entities_years"] = ", ".join(sorted(set(found_years)))

        return meta

    # ====================================
    #   EMBEDDINGS
    # ====================================

    def _shorten_chunk(self, text: str, max_chars: int = 2000) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return (
            text[:half]
            + "\n[... TEXTE TRONQUÉ POUR L'EMBEDDING ...]\n"
            + text[-half:]
        )

    def _encode_chunks(self, chunks: List[str]) -> np.ndarray:
        """Encode une liste de chunks en embeddings, avec protection et préfixe E5."""
        all_vecs: List[np.ndarray] = []
        for idx, ch in enumerate(chunks):
            safe_text = self._shorten_chunk(ch)
            model_input = f"passage: {safe_text}"
            try:
                emb = self.model.encode(
                    [model_input],
                    batch_size=1,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                if isinstance(emb, np.ndarray):
                    if emb.ndim == 2:
                        v = emb[0]
                    else:
                        v = emb
                else:
                    v = np.array(emb, dtype=np.float32)
            except Exception as e:
                print(f"   ⚠️ Erreur encode sur chunk {idx}: {e} → vecteur neutre.")
                v = np.zeros(self.embedding_dim, dtype=np.float32)
            all_vecs.append(v)
        return np.vstack(all_vecs)

    # ====================================
    #   TRAITEMENT D'UN DOCUMENT
    # ====================================

    def process_document(self, file_path: Path) -> int:
        # 1) Extraction brute
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        elif suffix in (".txt", ".md"):
            text = self.extract_text_from_txt(file_path)
        else:
            print(f"⏭️  Format non supporté : {suffix}")
            return 0

        text = self._strip_surrogates(text)

        if self.debug:
            raw_out = self.debug_dir / f"{file_path.stem}__RAW.txt"
            try:
                raw_out.write_text(text[:200_000], encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"⚠️  Impossible d'écrire RAW debug pour {file_path.name}: {e}")

        if not text or len(text) < 100:
            print(f"⚠️  Document vide ou trop court : {file_path.name}")
            return 0

        # 2) Type de document
        parts_lower = [p.lower() for p in file_path.parts]
        is_cartulary = any("cartul" in p for p in parts_lower)
        is_charter = any("charte" in p or "charter" in p for p in parts_lower) and not is_cartulary
        is_entity_index = any(p == "autres" for p in parts_lower)
        is_academic = not (is_charter or is_cartulary or is_entity_index)

        folder: Optional[str] = None
        try:
            idx = parts_lower.index("mes_documents_historiques")
            if idx + 1 < len(file_path.parts):
                folder = file_path.parts[idx + 1]
        except ValueError:
            folder = file_path.parent.name

        # 3) Nettoyage intelligent
        if self.enable_text_cleaning:
            raw_text = text
            if is_entity_index:
                cleaned_text, cstats = self.clean_text_intelligently(text, "cartulary")
            elif is_academic:
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

        if self.debug:
            cleaned_out = self.debug_dir / f"{file_path.stem}__CLEANED.txt"
            try:
                cleaned_out.write_text(text[:200_000], encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"⚠️  Impossible d'écrire CLEANED debug pour {file_path.name}: {e}")

        # Détection de langue globale du document
        doc_lang = self.detect_language(text)

        # 4) Notes
        foot_meta = {"has_footnotes": False, "count": 0}
        if is_academic and self.process_footnotes:
            text, foot_meta = self.extract_and_integrate_footnotes(text)

        # 5) Prétraitements & splitters
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
            print(f"⚠️  Aucun chunk produit pour {file_path.name}")
            return 0

        cleaned_chunks: List[str] = []
        for idx, ch in enumerate(chunks):
            if not isinstance(ch, str):
                try:
                    ch = str(ch)
                except Exception:
                    print(f"   ⚠️ Chunk {idx} ignoré (type invalide: {type(ch)}) dans {file_path.name}")
                    continue
            ch = self._strip_surrogates(ch)
            ch = ch.strip()
            if not ch:
                continue
            cleaned_chunks.append(ch)
        chunks = cleaned_chunks

        if not chunks:
            print(f"⚠️  Plus aucun chunk exploitable après nettoyage pour {file_path.name}")
            return 0

        if self.debug:
            preview_path = self.debug_dir / f"{file_path.stem}__CHUNKS_PREVIEW.txt"
            try:
                with preview_path.open("w", encoding="utf-8", errors="ignore") as f:
                    for i, ch in enumerate(chunks[:5]):
                        f.write(f"\n\n===== CHUNK {i} =====\n\n")
                        f.write(ch)
            except Exception as e:
                print(f"⚠️  Impossible d'écrire CHUNKS_PREVIEW pour {file_path.name}: {e}")

        # 6) Embeddings
        embeddings = self._encode_chunks(chunks)

        # 7) IDs & métadonnées
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
                "file_type": suffix[1:].lower(),
                "is_charter": is_charter,
                "is_cartulary": is_cartulary,
                "is_academic": is_academic,
                "is_entity_index": is_entity_index,
                "doc_lang": doc_lang,
            }

            # Enrichissement optionnel par spaCy FR si disponible
            if getattr(self, "nlp_fr", None) and doc_lang == "FR":
                spacy_meta = self.analyze_chunk_spacy(ch)
                if spacy_meta:
                    md.update(spacy_meta)

            if is_academic and foot_meta.get("has_footnotes"):
                nums = re.findall(r"\[Note (\d+):", ch)
                md.update(
                    {
                        "has_footnotes": bool(nums),
                        "footnote_count": len(nums),
                        "footnote_numbers": ", ".join(nums) if nums else "",
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
                        else "",
                    }
                )

            if is_cartulary and cart_meta:
                md.update(
                    {
                        "institution": cart_meta.get("institution") or "",
                        "estimated_total_acts": int(cart_meta.get("total_acts_estimated", 0)),
                        "date_range": cart_meta.get("date_range") or "",
                    }
                )
                if acts_info:
                    chunk_start = sum(len(c) for c in chunks[:i])
                    relevant = [a for a in acts_info if a["position"] <= chunk_start]
                    if relevant:
                        md["act_number"] = str(relevant[-1]["number"])

            if self.entity_lexicon and not is_entity_index:
                ent_meta = self._annotate_chunk_entities(ch)
                if ent_meta:
                    md.update(ent_meta)

            metadatas.append(md)

        if not (
            len(chunks) == len(embeddings) == len(ids) == len(metadatas)
        ):
            raise ValueError(
                f"Incohérence longueurs: chunks={len(chunks)} "
                f"embeddings={len(embeddings)} ids={len(ids)} "
                f"metadatas={len(metadatas)}"
            )

        max_batch = 1000
        for start in range(0, len(chunks), max_batch):
            end = start + max_batch
            batch_embeddings = embeddings[start:end]
            batch_chunks = chunks[start:end]
            batch_ids = ids[start:end]
            batch_metadatas = metadatas[start:end]

            self.collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=batch_chunks,
                ids=batch_ids,
                metadatas=batch_metadatas,
            )

        nb_chunks = len(chunks)
        self._update_registry_for_file(file_path, nb_chunks)

        # Stats globales
        self.stats["processed_files"] += 1
        self.stats["total_chunks"] += nb_chunks
        self.stats["by_type"].setdefault(suffix[1:].lower(), 0)
        self.stats["by_type"][suffix[1:].lower()] += 1
        if folder:
            self.stats["by_folder"].setdefault(folder, 0)
            self.stats["by_folder"][folder] += 1

        print(f"✅ {file_path.name}: {nb_chunks} chunks indexés.")
        return nb_chunks

    # ====================================
    #   BOUCLE SUR LE CORPUS
    # ====================================

    def process_corpus(self) -> None:
        docs_dir = self.base_path / "mes_documents_historiques"
        if not docs_dir.exists():
            print(f"❌ Dossier mes_documents_historiques introuvable dans {self.base_path}")
            return

        all_files: List[Path] = []
        for sub in ["theses", "cartulaires", "chartes", "articles", "autres"]:
            d = docs_dir / sub
            if d.exists():
                all_files.extend(sorted(d.glob("*.pdf")))
                all_files.extend(sorted(d.glob("*.txt")))
                all_files.extend(sorted(d.glob("*.md")))

        if not all_files:
            print("❌ Aucun document trouvé dans mes_documents_historiques (pdf/txt/md).")
            return

        print(f"📚 {len(all_files)} fichiers trouvés")
        print("📂 Extensions:", sorted(set(f.suffix for f in all_files)))

        for idx, fpath in enumerate(all_files, start=1):
            print(f"\n🧾 [{idx}/{len(all_files)}] Traitement : {fpath.name}")
            if self._is_already_indexed(fpath):
                print(f"⏭️  Déjà indexé (hash identique) : {fpath.name}")
                self.stats["skipped_files"] += 1
                continue
            try:
                self.process_document(fpath)
            except Exception as e:
                print(f"❌ Erreur sur {fpath.name}: {e}")
                self.stats["failed_files"].append(str(fpath))

        print("\n===== RÉSUMÉ =====")
        print("Fichiers traités:", self.stats["processed_files"])
        print("Fichiers ignorés (déjà indexés):", self.stats["skipped_files"])
        print("Total de chunks:", self.stats["total_chunks"])
        print("Répartition par type:", self.stats["by_type"])
        print("Répartition par dossier:", self.stats["by_folder"])
        if self.stats["failed_files"]:
            print("Fichiers en erreur:")
            for ff in self.stats["failed_files"]:
                print("  -", ff)


if __name__ == "__main__":
    base_path = Path(".").resolve()
    db_path = base_path / "vector_db_historique"
    collection_name = "documents_historiques"
    registry_path = base_path / "documents_historiques_registry_v2.json"

    vectorizer = HistoricalDocVectorizer(
        base_path=base_path,
        db_path=db_path,
        collection_name=collection_name,
        registry_path=registry_path,
        process_footnotes=True,
        enable_text_cleaning=True,
        ocr_enable=True,          # mets False ici si tu veux désactiver l'OCR pour tester
        ocr_lang_default="fra+lat",
        debug=False,
    )
    vectorizer.process_corpus()
