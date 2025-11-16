import torch
from sentence_transformers import SentenceTransformer


def detect_device(preferred: str = "auto") -> str:
    """
    Choisit automatiquement CUDA si présent et demandé.
    """
    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding_model(model_name: str = "intfloat/multilingual-e5-large",
                         device: str = "auto"):
    """
    Charge le modèle d'embedding E5-large, affiche infos GPU si dispo.
    """
    device = detect_device(device)
    model = SentenceTransformer(model_name, device=device)

    print(f"📥 Modèle chargé : {model_name} sur {device}")

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / (1024 ** 3)
        print(f"   GPU : {props.name}")
        print(f"   VRAM : {vram:.2f} GB")

    return model, device
