#!/usr/bin/env python3
"""Test complet de la configuration GPU dans WSL2"""
import torch, sys

print("\n" + "="*70)
print("🔍 TEST DE CONFIGURATION GPU")
print("="*70 + "\n")

# Infos système
print(f"🐍 Python: {sys.version.split()[0]}")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"   CUDA disponible: {torch.cuda.is_available()}")
print(f"   CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"🎮 GPU [{i}]: {props.name}")
        print(f"   VRAM: {props.total_memory / 1e9:.2f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")

    # Test allocation mémoire
    print("\n🧪 Test allocation tensor sur GPU...")
    try:
        x = torch.rand((1000, 1000), device='cuda')
        print(f"✅ Tensor créé sur {x.device}")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Erreur allocation: {e}")
else:
    print("\n⚠️  GPU non disponible.")
    print("   Vérifie les drivers NVIDIA pour WSL et redémarre avec:")
    print("   wsl --shutdown")

print("\n" + "="*70)
if torch.cuda.is_available():
    print("✅ CONFIGURATION COMPLÈTE ET FONCTIONNELLE")
else:
    print("⚠️  GPU non détecté ou non disponible")
print("="*70 + "\n")
