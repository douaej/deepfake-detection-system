"""
Script de test standalone - Deepfake Detection
Teste plusieurs images et affiche les probabilités brutes.

Usage:
    python test_detection.py                        # teste avec images factices
    python test_detection.py --images a.jpg b.jpg   # teste tes propres images
    python test_detection.py --folder ./mon_dossier # teste tout un dossier
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from torchvision import transforms


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = "models/pretrained/deepfake_efficientnet.pth"  # chemin par défaut
DEVICE       = "cpu"
INPUT_SIZE   = 224
THRESHOLD    = 0.50

# ImageFolder trie alphabétiquement → fake=0, real=1
CLASS_NAMES  = {0: "FAKE", 1: "REAL"}

TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Chargement modèle ─────────────────────────────────────────────────────────

def load_model(model_path: str) -> torch.nn.Module:
    path = Path(model_path)
    if not path.exists():
        print(f"❌  Modèle introuvable : {path}")
        print("    Vérifiez le chemin avec --model")
        sys.exit(1)

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(
        torch.load(str(path), map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
    print(f"✓  Modèle chargé : {path}\n")
    return model


# ── Inférence sur une image ───────────────────────────────────────────────────

def predict(model: torch.nn.Module, image_path: str) -> dict:
    """Retourne le résultat complet pour une image."""
    path = Path(image_path)

    if not path.exists():
        return {"file": str(path), "error": "fichier introuvable"}

    try:
        t0 = time.time()

        img = Image.open(path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = F.softmax(outputs, dim=1)

        # fake=0, real=1  (ordre alphabétique ImageFolder)
        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()

        # fake=0 → si fake_prob dominant, prediction=0 (FAKE)
        prediction = 0 if fake_prob >= THRESHOLD else 1
        elapsed_ms = (time.time() - t0) * 1000

        return {
            "file":       path.name,
            "verdict":    CLASS_NAMES[prediction],
            "fake_prob":  round(fake_prob, 4),
            "real_prob":  round(real_prob, 4),
            "confidence": round(max(fake_prob, real_prob), 4),
            "time_ms":    round(elapsed_ms, 1),
            "error":      None,
        }

    except Exception as e:
        return {"file": str(path), "error": str(e)}


# ── Affichage ─────────────────────────────────────────────────────────────────

def print_result(r: dict):
    if r.get("error"):
        print(f"  ⚠️  {r['file']} → ERREUR : {r['error']}")
        return

    icon = "🔴 FAKE" if r["verdict"] == "FAKE" else "🟢 REAL"
    bar_fake = "█" * int(r["fake_prob"] * 20)
    bar_real = "█" * int(r["real_prob"] * 20)

    print(f"  Fichier    : {r['file']}")
    print(f"  Verdict    : {icon}  (confiance {r['confidence']:.1%})")
    print(f"  FAKE prob  : {r['fake_prob']:.4f}  |{bar_fake:<20}|")
    print(f"  REAL prob  : {r['real_prob']:.4f}  |{bar_real:<20}|")
    print(f"  Temps      : {r['time_ms']} ms")
    print()


def print_summary(results: list):
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return

    n_fake = sum(1 for r in valid if r["verdict"] == "FAKE")
    n_real = len(valid) - n_fake
    avg_conf = np.mean([r["confidence"] for r in valid])

    print("=" * 50)
    print("  RÉSUMÉ")
    print("=" * 50)
    print(f"  Images testées : {len(valid)}")
    print(f"  FAKE détectés  : {n_fake}")
    print(f"  REAL détectés  : {n_real}")
    print(f"  Confiance moy. : {avg_conf:.1%}")
    print("=" * 50)


# ── Images de test synthétiques ───────────────────────────────────────────────

def make_dummy_images() -> list:
    """Crée 4 images PNG temporaires pour tester sans avoir de photos."""
    import tempfile, os

    tmp = tempfile.mkdtemp()
    paths = []

    specs = [
        ("dummy_face_1.png",  (200, 180, 160)),   # teinte chair
        ("dummy_face_2.png",  (180, 160, 140)),
        ("dummy_noise.png",   None),               # bruit aléatoire
        ("dummy_gradient.png", None),              # dégradé
    ]

    for name, color in specs:
        arr = np.zeros((224, 224, 3), dtype=np.uint8)

        if color:
            arr[:] = color
            # Ajouter un cercle pour simuler un visage
            import cv2
            cv2.circle(arr, (112, 100), 70, (220, 190, 170), -1)
            cv2.circle(arr, (85,  90),  10, (60, 40, 30),    -1)
            cv2.circle(arr, (139, 90),  10, (60, 40, 30),    -1)
        elif "noise" in name:
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            for i in range(224):
                arr[i, :] = [i, 255 - i, 128]

        p = os.path.join(tmp, name)
        Image.fromarray(arr).save(p)
        paths.append(p)

    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test de détection deepfake")
    parser.add_argument("--model",   default=MODEL_PATH, help="Chemin vers le .pth")
    parser.add_argument("--images",  nargs="+",          help="Images à tester")
    parser.add_argument("--folder",  default=None,       help="Dossier d'images")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════╗")
    print("║     DEEPFAKE DETECTION — TEST SCRIPT     ║")
    print("╚══════════════════════════════════════════╝\n")

    # Charger le modèle
    model = load_model(args.model)

    # Collecter les images à tester
    image_paths = []

    if args.images:
        image_paths = args.images

    elif args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"❌  Dossier introuvable : {folder}")
            sys.exit(1)
        image_paths = [
            str(p) for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        print(f"  {len(image_paths)} images trouvées dans {folder}\n")

    else:
        # Aucune image fournie → test avec images synthétiques
        print("  ℹ️  Aucune image fournie → génération d'images de test synthétiques")
        print("  (ces images n'ont pas de vrai visage, les résultats sont indicatifs)\n")
        image_paths = make_dummy_images()

    if not image_paths:
        print("❌  Aucune image à tester.")
        sys.exit(1)

    # Lancer les tests
    results = []
    print("─" * 50)

    for path in image_paths:
        result = predict(model, path)
        results.append(result)
        print_result(result)

    print_summary(results)


if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        pass  # cv2 optionnel pour les images synthétiques

    main()