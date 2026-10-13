#!/usr/bin/env python3
"""
Script de vérification de l'installation
À exécuter après: pip install -r requirements.txt
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Vérifie version Python"""
    print("1. Python Version", end=" ... ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✓ {version.major}.{version.minor}")
        return True
    else:
        print(f"✗ {version.major}.{version.minor} (need 3.9+)")
        return False


def check_imports():
    """Vérifie que tous les imports fonctionnent"""
    print("\n2. Required Libraries")
    
    libraries = [
        ('torch', 'PyTorch'),
        ('torchvision', 'Torchvision'),
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('sklearn', 'Scikit-Learn'),
        ('skimage', 'Scikit-Image'),
        ('flask', 'Flask'),
        ('psutil', 'PSUtil'),
    ]
    
    all_ok = True
    for module, name in libraries:
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} (missing)")
            all_ok = False
    
    return all_ok


def check_directories():
    """Vérifie structure répertoires"""
    print("\n3. Directory Structure")
    
    required_dirs = [
        'src',
        'backend',
        'scripts',
        'data',
        'models',
        'logs',
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            print(f"   ✓ {dir_name}/")
        else:
            print(f"   ✗ {dir_name}/ (missing)")
            all_ok = False
    
    return all_ok


def check_files():
    """Vérifie fichiers clés"""
    print("\n4. Key Files")
    
    required_files = [
        'config.py',
        'main.py',
        'requirements.txt',
        'README.md',
        'src/__init__.py',
        'src/detector.py',
        'src/image_processor.py',
        'src/video_processor.py',
        'src/utils.py',
        'backend/app.py',
        'scripts/download_models.py',
        'scripts/test_detection.py',
    ]
    
    all_ok = True
    for file_name in required_files:
        path = Path(file_name)
        if path.exists() and path.is_file():
            size = path.stat().st_size / 1024
            print(f"   ✓ {file_name} ({size:.0f}KB)")
        else:
            print(f"   ✗ {file_name} (missing)")
            all_ok = False
    
    return all_ok


def check_config():
    """Vérifie configuration"""
    print("\n5. Configuration")
    
    try:
        from config import CONFIG
        
        print(f"   ✓ config.py loaded")
        print(f"      Device: {CONFIG.device}")
        print(f"      Batch size: {CONFIG.batch_size}")
        print(f"      Input size: {CONFIG.inference.INPUT_SIZE}")
        print(f"      Confidence threshold: {CONFIG.inference.CONFIDENCE_THRESHOLD}")
        
        return True
    except Exception as e:
        print(f"   ✗ Error loading config: {e}")
        return False


def check_models():
    """Vérifie modèles"""
    print("\n6. Models")
    
    from pathlib import Path
    models_dir = Path('models/pretrained')
    
    if models_dir.exists():
        files = list(models_dir.glob('*.pth'))
        if files:
            print(f"   ✓ Models directory exists")
            for f in files:
                size = f.stat().st_size / (1024 ** 2)
                print(f"      - {f.name} ({size:.1f}MB)")
            return True
        else:
            print(f"   ⚠ Models directory empty")
            print(f"      Run: python main.py setup")
            return False
    else:
        print(f"   ⚠ Models directory missing")
        print(f"      Run: python main.py setup")
        return False


def check_detector():
    """Teste instanciation du détecteur"""
    print("\n7. Detector Initialization")
    
    try:
        from src.detector import DeepfakeDetector
        detector = DeepfakeDetector(device='cpu', load_models=False)
        print(f"   ✓ DeepfakeDetector initialized (models not loaded)")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def check_processors():
    """Teste image et video processors"""
    print("\n8. Image & Video Processors")
    
    try:
        from src.image_processor import ImageProcessor
        from src.video_processor import VideoProcessor
        
        img_proc = ImageProcessor(device='cpu')
        print(f"   ✓ ImageProcessor initialized")
        
        vid_proc = VideoProcessor(device='cpu')
        print(f"   ✓ VideoProcessor initialized")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def main():
    """Main verification"""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - SETUP VERIFICATION")
    print("="*60 + "\n")
    
    checks = [
        check_python_version,
        check_imports,
        check_directories,
        check_files,
        check_config,
        check_models,
        check_detector,
        check_processors,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print("✓ SETUP VERIFIED - ALL CHECKS PASSED")
        print("="*60)
        
        print("\nNext steps:")
        print("  1. python main.py setup          # Download models")
        print("  2. python main.py test           # Run tests")
        print("  3. python main.py api --port 5000  # Start API")
        
        return 0
    else:
        print(f"⚠ SETUP INCOMPLETE - {passed}/{total} checks passed")
        print("="*60)
        
        print("\nTo fix issues:")
        print("  1. pip install -r requirements.txt")
        print("  2. python main.py setup")
        print("  3. Run this script again")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
