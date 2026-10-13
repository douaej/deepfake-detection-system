"""
Exemples d'utilisation du détecteur de deepfakes
Démontre les différents modes d'utilisation
"""

from src.detector import DeepfakeDetector, quick_detect_image, quick_detect_video
from config import CONFIG
import json


def example_1_basic_image_detection():
    """Exemple 1: Détection image simple"""
    print("\n" + "="*60)
    print("EXEMPLE 1: Détection image simple")
    print("="*60)
    
    # Détection rapide
    result = quick_detect_image('data/sample.jpg')
    
    print(f"\nRésultat: {result['result']}")
    print(f"Confiance: {result['confidence']:.2%}")
    print(f"Temps: {result['processing_time_ms']:.1f}ms")
    
    if result['evidence']:
        print(f"Evidence: {json.dumps(result['evidence'], indent=2)}")


def example_2_detector_class():
    """Exemple 2: Utiliser classe DeepfakeDetector"""
    print("\n" + "="*60)
    print("EXEMPLE 2: Utiliser classe DeepfakeDetector")
    print("="*60)
    
    # Initialiser détecteur
    detector = DeepfakeDetector(device='cpu')
    
    # Option 1: Chemin fichier
    result = detector.detect_image('image.jpg')
    
    # Option 2: Bytes (upload)
    with open('image.jpg', 'rb') as f:
        result = detector.detect_image(f.read())
    
    # Option 3: NumPy array
    import numpy as np
    image_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.detect_image(image_array)
    
    print(f"Result: {result['result']}")
    print(f"Confidence: {result['confidence']:.2%}")


def example_3_video_detection():
    """Exemple 3: Détection vidéo"""
    print("\n" + "="*60)
    print("EXEMPLE 3: Détection vidéo")
    print("="*60)
    
    detector = DeepfakeDetector(device='cpu')
    
    # Détection vidéo
    result = detector.detect_video('video.mp4')
    
    print(f"Résultat global: {result['overall_result']}")
    print(f"Confiance moyenne: {result['avg_confidence']:.2%}")
    print(f"Frames analysés: {result['stats']['num_frames']}")
    print(f"Ratio faux: {result['stats']['fake_ratio']:.2%}")
    print(f"Temps total: {result['processing_time_ms']:.1f}ms")


def example_4_batch_processing():
    """Exemple 4: Traitement batch"""
    print("\n" + "="*60)
    print("EXEMPLE 4: Traitement batch d'images")
    print("="*60)
    
    detector = DeepfakeDetector(device='cpu', batch_size=4)
    
    image_paths = [
        'data/image1.jpg',
        'data/image2.jpg',
        'data/image3.jpg',
    ]
    
    results = detector.detect_batch(image_paths)
    
    for img_path, result in zip(image_paths, results):
        print(f"{img_path}: {result['result']} ({result['confidence']:.2%})")


def example_5_with_features():
    """Exemple 5: Détection avec features d'analyse"""
    print("\n" + "="*60)
    print("EXEMPLE 5: Détection avec features d'analyse")
    print("="*60)
    
    detector = DeepfakeDetector(device='cpu')
    
    # Activer features d'analyse
    result = detector.detect_image(
        'image.jpg',
        return_features=True,
        return_heatmap=True
    )
    
    print(f"Résultat: {result['result']}")
    print(f"Confiance: {result['confidence']:.2%}")
    
    # Evidence contient texture, frequency, geometry
    if 'texture' in result['evidence']:
        print(f"Texture features: {result['evidence']['texture']}")
    
    if 'frequency' in result['evidence']:
        print(f"Frequency features: {result['evidence']['frequency']}")


def example_6_api_usage():
    """Exemple 6: Utiliser l'API Backend"""
    print("\n" + "="*60)
    print("EXEMPLE 6: Utiliser API Backend")
    print("="*60)
    
    print("\n1. Démarrer le serveur:")
    print("   python backend/app.py --port 5000 --device cpu")
    
    print("\n2. Détection image (Python):")
    print("""
    import requests
    
    with open('image.jpg', 'rb') as f:
        files = {'file': f}
        response = requests.post(
            'http://localhost:5000/api/detect/image',
            files=files,
            data={'return_heatmap': 'true'}
        )
        result = response.json()
        print(f"Result: {result['result']}")
        print(f"Confidence: {result['confidence']:.2%}")
    """)
    
    print("\n3. Détection image (cURL):")
    print("   curl -X POST -F 'file=@image.jpg' http://localhost:5000/api/detect/image")
    
    print("\n4. Détection vidéo:")
    print("   curl -X POST -F 'file=@video.mp4' http://localhost:5000/api/detect/video")
    
    print("\n5. Health check:")
    print("   curl http://localhost:5000/api/health")


def example_7_configuration():
    """Exemple 7: Configuration personnalisée"""
    print("\n" + "="*60)
    print("EXEMPLE 7: Configuration personnalisée")
    print("="*60)
    
    # Modifier configuration
    CONFIG.batch_size = 2
    CONFIG.device = 'cpu'
    CONFIG.inference.FAKE_THRESHOLD = 0.5
    CONFIG.inference.VIDEO_FPS_TARGET = 5
    
    print("Configuration modifiée:")
    print(f"  Device: {CONFIG.device}")
    print(f"  Batch size: {CONFIG.batch_size}")
    print(f"  Fake threshold: {CONFIG.inference.FAKE_THRESHOLD}")
    print(f"  Video FPS: {CONFIG.inference.VIDEO_FPS_TARGET}")
    
    # Utiliser détecteur avec nouvelle config
    detector = DeepfakeDetector(
        device=CONFIG.device,
        batch_size=CONFIG.batch_size
    )
    
    print(f"\nDétecteur initialisé avec nouvelle config")


def example_8_error_handling():
    """Exemple 8: Gestion d'erreurs"""
    print("\n" + "="*60)
    print("EXEMPLE 8: Gestion d'erreurs")
    print("="*60)
    
    detector = DeepfakeDetector(device='cpu')
    
    # Fichier inexistant
    result = detector.detect_image('nonexistent.jpg')
    print(f"Fichier inexistant: result={result['result']}, confidence={result['confidence']}")
    
    # Image invalide
    result = detector.detect_image(b'invalid data')
    print(f"Data invalide: result={result['result']}")
    
    # Aucun visage détecté
    import numpy as np
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect_image(blank_image)
    print(f"Aucun visage: result={result['result']}")


def example_9_performance_monitoring():
    """Exemple 9: Monitoring performance"""
    print("\n" + "="*60)
    print("EXEMPLE 9: Monitoring performance")
    print("="*60)
    
    detector = DeepfakeDetector(device='cpu')
    
    import time
    import numpy as np
    
    # Créer images test
    test_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(10)
    ]
    
    # Mesurer temps total
    start = time.time()
    results = detector.detect_batch(test_images)
    elapsed = time.time() - start
    
    fps = len(test_images) / elapsed
    avg_time = (elapsed / len(test_images)) * 1000
    
    print(f"Images traitées: {len(results)}")
    print(f"Temps total: {elapsed:.2f}s")
    print(f"Temps moyen par image: {avg_time:.1f}ms")
    print(f"Throughput: {fps:.1f} images/sec")


def main():
    """Exécute les exemples"""
    print("\n")
    print("╔════════════════════════════════════════════════════╗")
    print("║    DEEPFAKE DETECTION - USAGE EXAMPLES             ║")
    print("╚════════════════════════════════════════════════════╝")
    
    examples = [
        ("Détection image simple", example_1_basic_image_detection),
        ("Classe DeepfakeDetector", example_2_detector_class),
        ("Détection vidéo", example_3_video_detection),
        ("Batch processing", example_4_batch_processing),
        ("Features d'analyse", example_5_with_features),
        ("API Backend", example_6_api_usage),
        ("Configuration", example_7_configuration),
        ("Gestion d'erreurs", example_8_error_handling),
        ("Monitoring", example_9_performance_monitoring),
    ]
    
    print("\nExemples disponibles:")
    for i, (title, _) in enumerate(examples, 1):
        print(f"  {i}. {title}")
    
    print("\nPour exécuter un exemple:")
    print("  python examples.py --example 1")
    print("  python examples.py --example all")
    
    # Optionnel: exécuter exemple spécifique
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, help='Example number')
    args = parser.parse_args()
    
    if args.example:
        if args.example == 0:
            for _, func in examples:
                func()
        elif 1 <= args.example <= len(examples):
            examples[args.example - 1][1]()
        else:
            print(f"Exemple {args.example} non trouvé")


if __name__ == '__main__':
    main()
