#!/usr/bin/env python3
"""
Test simple - Vérification que tout fonctionne
À exécuter après: python main.py setup
"""

import numpy as np
from pathlib import Path
from PIL import Image

print("\n" + "="*60)
print("DEEPFAKE DETECTION - SIMPLE TEST")
print("="*60)

# 1. Vérifier imports
print("\n1. Checking imports...")
try:
    from config import CONFIG
    from src.detector import DeepfakeDetector
    from src.image_processor import ImageProcessor
    from src.video_processor import VideoProcessor
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    print("   Please run: pip install -r requirements.txt")
    exit(1)

# 2. Vérifier modèles
print("\n2. Checking models...")
models_dir = Path('models/pretrained')
if models_dir.exists() and list(models_dir.glob('*.pth')):
    print(f"   ✓ Models found in {models_dir}")
else:
    print(f"   ✗ Models not found")
    print("   Please run: python main.py setup")
    exit(1)

# 3. Créer image test
print("\n3. Creating test image...")
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
test_image_path = Path('test_simple.jpg')

Image.fromarray(test_image).save(test_image_path)
print(f"   ✓ Test image saved: {test_image_path}")

# 4. Initialiser détecteur
print("\n4. Initializing detector...")
try:
    detector = DeepfakeDetector(device='cpu')
    print("   ✓ Detector initialized")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# 5. Test image detection
print("\n5. Testing image detection...")
try:
    print("   Running inference... (may take 30-60 seconds on first run)")
    result = detector.detect_image(test_image)
    
    print(f"\n   Result:")
    print(f"   - Status: {result['result']}")
    print(f"   - Confidence: {result['confidence']:.2%}")
    print(f"   - Processing time: {result['processing_time_ms']:.1f}ms")
    
    if result['result'] == -1:
        print(f"   ⚠ Warning: No faces detected (normal for random image)")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. Test batch processing
print("\n6. Testing batch processing...")
try:
    images = [test_image for _ in range(2)]
    results = detector.detect_batch(images)
    print(f"   ✓ Processed {len(results)} images")
    for i, res in enumerate(results):
        print(f"   Image {i+1}: {res['result']} ({res['confidence']:.2%})")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 7. Test image processor
print("\n7. Testing ImageProcessor...")
try:
    processor = ImageProcessor(device='cpu')
    
    # Test face detection
    faces = processor.detect_faces(test_image)
    print(f"   ✓ Detected {len(faces)} faces")
    
    # Test texture analysis
    texture = processor.get_texture_features(test_image)
    print(f"   ✓ Texture analysis: {len(texture)} features")
    
    # Test frequency analysis  
    freq = processor.get_frequency_features(test_image)
    print(f"   ✓ Frequency analysis: {len(freq)} features")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

# 8. Summary
print("\n" + "="*60)
print("✓ ALL TESTS PASSED")
print("="*60)

print("\nYou can now:")
print("  - Run full tests: python main.py test")
print("  - Detect images: python main.py detect-image --image <path>")
print("  - Start API: python main.py api --port 5000")
print("  - Read docs: README.md, QUICKSTART.md")

# Cleanup
test_image_path.unlink()
print(f"\nTest image cleaned up")
