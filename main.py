#!/usr/bin/env python3
"""
Deepfake Detection System - Point d'entrée principal
Optimisé pour Intel i5-1135G7 / 8GB RAM
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configurer les threads PyTorch AVANT tout import torch
# i5-1135G7 = 4 cœurs physiques / 8 threads logiques
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from config import CONFIG, print_config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(CONFIG.logging.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Commandes ─────────────────────────────────────────────────────────────────

def cmd_setup(args=None) -> int:
    """Télécharge et initialise les modèles pré-entraînés"""
    logger.info("=" * 60)
    logger.info("SETUP: Téléchargement des modèles")
    logger.info("=" * 60)
    try:
        from scripts.download_models import main as download_main
        download_main()
        logger.info("✓ Setup terminé")
        return 0
    except Exception as e:
        logger.error(f"✗ Setup échoué: {e}")
        return 1


def cmd_test(args=None) -> int:
    """Lance la suite de tests"""
    logger.info("=" * 60)
    logger.info("TESTS: Lancement de la suite de tests")
    logger.info("=" * 60)
    try:
        from scripts.test_detection import run_full_test_suite
        results = run_full_test_suite()
        logger.info("✓ Tests terminés")
        return 0
    except Exception as e:
        logger.error(f"✗ Tests échoués: {e}")
        return 1


def cmd_detect_image(args=None) -> int:
    """Détecte deepfake dans une image"""
    from src.detector import DeepfakeDetector

    logger.info(f"Analyse image: {args.image}")

    if not Path(args.image).exists():
        logger.error(f"Fichier introuvable: {args.image}")
        return 1

    try:
        detector = DeepfakeDetector(device=CONFIG.device)
        detector.warm_up()  # Chauffe les modèles

        result = detector.detect_image(
            args.image,
            return_features=args.features,
            return_heatmap=args.heatmap
        )

        _print_image_result(result)
        return 0

    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=args.verbose)
        return 1


def cmd_detect_video(args=None) -> int:
    """Détecte deepfake dans une vidéo"""
    from src.detector import DeepfakeDetector

    logger.info(f"Analyse vidéo: {args.video}")

    if not Path(args.video).exists():
        logger.error(f"Fichier introuvable: {args.video}")
        return 1

    try:
        detector = DeepfakeDetector(device=CONFIG.device)
        result = detector.detect_video(args.video, return_frame_details=args.details)

        _print_video_result(result)
        return 0

    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=args.verbose)
        return 1


def cmd_api(args=None) -> int:
    """Démarre le serveur API Flask"""
    logger.info("=" * 60)
    logger.info("API: Démarrage du serveur")
    logger.info("=" * 60)

    try:
        from backend.app import app

        host = args.host or CONFIG.api.HOST
        port = args.port or CONFIG.api.PORT
        debug = args.debug

        logger.info(f"  Host:   {host}")
        logger.info(f"  Port:   {port}")
        logger.info(f"  Debug:  {debug}")
        logger.info(f"  Device: {CONFIG.device}")
        logger.info(f"\n  Endpoints:")
        logger.info(f"    GET  /api/health")
        logger.info(f"    POST /api/detect/image")
        logger.info(f"    POST /api/detect/video")
        logger.info(f"    POST /api/detect/batch")
        logger.info(f"\n  URL: http://localhost:{port}")

        app.run(host=host, port=port, debug=debug, use_reloader=False)
        return 0

    except Exception as e:
        logger.error(f"Erreur démarrage API: {e}")
        return 1


def cmd_interactive(args=None) -> None:
    """Mode interactif en ligne de commande"""
    from src.detector import DeepfakeDetector

    logger.info("MODE INTERACTIF - Ctrl+C pour quitter")
    detector = DeepfakeDetector(device=CONFIG.device)
    detector.warm_up()

    while True:
        print("\n" + "─" * 40)
        print("  1. Analyser une image")
        print("  2. Analyser une vidéo")
        print("  3. Afficher la configuration")
        print("  4. Quitter")
        print("─" * 40)

        try:
            choice = input("Choix (1-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir!")
            break

        if choice == "1":
            path = input("Chemin de l'image: ").strip()
            if Path(path).exists():
                result = detector.detect_image(path)
                _print_image_result(result)
            else:
                print("❌ Fichier introuvable")

        elif choice == "2":
            path = input("Chemin de la vidéo: ").strip()
            if Path(path).exists():
                result = detector.detect_video(path)
                _print_video_result(result)
            else:
                print("❌ Fichier introuvable")

        elif choice == "3":
            print_config()

        elif choice == "4":
            print("Au revoir!")
            break


# ── Formatage résultats ───────────────────────────────────────────────────────

def _print_image_result(result: dict):
    """Affiche résultat image en console"""
    r = result.get("result", "UNKNOWN")
    conf = result.get("confidence", 0)
    time_ms = result.get("processing_time_ms", 0)
    evidence = result.get("evidence", {})

    icon = "🔴 FAKE" if r == "FAKE" else ("🟢 REAL" if r == "REAL" else "⚠️  INCONNU")

    print("\n" + "=" * 60)
    print("RÉSULTAT - IMAGE")
    print("=" * 60)
    print(f"  Verdict:          {icon}")
    print(f"  Confiance:        {conf:.1%}")
    print(f"  Probabilité FAKE: {result.get('fake_probability', 0):.1%}")
    print(f"  Temps:            {time_ms:.0f}ms")

    if evidence.get("num_faces"):
        print(f"  Visages:          {evidence['num_faces']}")

    if evidence.get("texture"):
        t = evidence["texture"]
        print(f"  Texture homog.:   {t.get('homogeneity', 'N/A'):.4f}")

    print("=" * 60)


def _print_video_result(result: dict):
    """Affiche résultat vidéo en console"""
    r = result.get("overall_result", "UNKNOWN")
    conf = result.get("avg_confidence", 0)
    time_ms = result.get("processing_time_ms", 0)
    stats = result.get("stats", {})

    icon = "🔴 FAKE" if r == "FAKE" else ("🟢 REAL" if r == "REAL" else "⚠️  INCONNU")

    print("\n" + "=" * 60)
    print("RÉSULTAT - VIDÉO")
    print("=" * 60)
    print(f"  Verdict:          {icon}")
    print(f"  Confiance moy.:   {conf:.1%}")
    print(f"  Ratio FAKE:       {stats.get('fake_ratio', 0):.1%}")
    print(f"  Frames analysées: {stats.get('num_frames', 'N/A')}")
    print(f"  Temps total:      {time_ms/1000:.1f}s")
    print("=" * 60)


# ── Parser CLI ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deepfake Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py setup                              # 1ère installation
  python main.py test                               # Vérification
  python main.py detect-image --image photo.jpg     # Analyser image
  python main.py detect-video --video video.mp4     # Analyser vidéo
  python main.py api --port 5000                    # Démarrer l'API
  python main.py interactive                        # Mode interactif
  python main.py config                             # Voir configuration
        """
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose / traceback complet")

    sub = parser.add_subparsers(dest="command")

    # setup
    sub.add_parser("setup", help="Télécharger et initialiser les modèles")

    # test
    sub.add_parser("test", help="Lancer les tests")

    # detect-image
    p_img = sub.add_parser("detect-image", help="Analyser une image")
    p_img.add_argument("--image", required=True, help="Chemin de l'image")
    p_img.add_argument("--features", action="store_true", help="Inclure les features")
    p_img.add_argument("--heatmap", action="store_true", help="Générer une heatmap")

    # detect-video
    p_vid = sub.add_parser("detect-video", help="Analyser une vidéo")
    p_vid.add_argument("--video", required=True, help="Chemin de la vidéo")
    p_vid.add_argument("--details", action="store_true", help="Détails par frame")

    # api
    p_api = sub.add_parser("api", help="Démarrer le serveur API")
    p_api.add_argument("--host", default=CONFIG.api.HOST)
    p_api.add_argument("--port", type=int, default=CONFIG.api.PORT)
    p_api.add_argument("--debug", action="store_true")

    # interactive
    sub.add_parser("interactive", help="Mode interactif")

    # config
    sub.add_parser("config", help="Afficher la configuration")

    args = parser.parse_args()

    # Banner
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         DEEPFAKE DETECTION SYSTEM v1.1               ║")
    print(f"║  Device: {CONFIG.device:<10}  Threads: {CONFIG.optimization.NUM_THREADS:<2}                ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Dispatch
    dispatch = {
        "setup": cmd_setup,
        "test": cmd_test,
        "detect-image": cmd_detect_image,
        "detect-video": cmd_detect_video,
        "api": cmd_api,
        "interactive": lambda a: (cmd_interactive(a), 0)[1],
        "config": lambda a: (print_config(), 0)[1],
    }

    if args.command in dispatch:
        return dispatch[args.command](args)

    parser.print_help()
    print("\nDémarrage rapide:")
    print("  1. python main.py setup")
    print("  2. python main.py test")
    print("  3. python main.py api")
    return 0


if __name__ == "__main__":
    sys.exit(main())