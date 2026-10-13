"""
Downloads and initializes pre-trained models
Execute once to prepare the environment
"""

import os
import sys
import logging
from pathlib import Path

# Fix imports: add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
import torchvision.models as models
import timm
import mediapipe as mp

from config import CONFIG
from src.utils import ModelDownloader

# Setup logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
        )
    ]
)
logger = logging.getLogger(__name__)


def download_pytorch_models():
    """Download pre-trained PyTorch models"""
    logger.info("=" * 60)
    logger.info("DOWNLOADING PYTORCH MODELS")
    logger.info("=" * 60)
    
    # Convert to Path object if it's a string
    pretrained_dir = Path(CONFIG.pretrained_dir)
    
    # EfficientNet-B0 via timm
    logger.info("\n1. EfficientNet-B0 (timm)...")
    try:
        model = timm.create_model('efficientnet_b0', pretrained=True)
        # Save
        save_path = pretrained_dir / "efficientnet_b0.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"✓ Saved: {save_path}")
        file_size = os.path.getsize(save_path) / (1024 ** 2)
        logger.info(f"   Size: {file_size:.1f} MB")
    except Exception as e:
        logger.error(f"✗ Error: {e}")
    
    # ResNet18
    logger.info("\n2. ResNet18 (torchvision)...")
    try:
        model = models.resnet18(pretrained=True)
        save_path = pretrained_dir / "resnet18.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"✓ Saved: {save_path}")
        file_size = os.path.getsize(save_path) / (1024 ** 2)
        logger.info(f"   Size: {file_size:.1f} MB")
    except Exception as e:
        logger.error(f"✗ Error: {e}")


def initialize_mediapipe():
    """Initialize MediaPipe (downloads automatically)"""
    logger.info("\n" + "=" * 60)
    logger.info("INITIALIZING MEDIAPIPE")
    logger.info("=" * 60)
    
    logger.info("\n1. MediaPipe Face Detection...")
    try:
        with mp.solutions.face_detection.FaceDetection() as detector:
            logger.info("✓ Face Detection model loaded")
    except Exception as e:
        logger.error(f"✗ Error: {e}")
    
    logger.info("\n2. MediaPipe Face Mesh...")
    try:
        with mp.solutions.face_mesh.FaceMesh() as mesh:
            logger.info("✓ Face Mesh model loaded")
    except Exception as e:
        logger.error(f"✗ Error: {e}")


def verify_installations():
    """Verify that everything is installed correctly"""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFYING INSTALLATIONS")
    logger.info("=" * 60)
    
    checks = {
        'PyTorch': lambda: __import__('torch').__version__,
        'torchvision': lambda: __import__('torchvision').__version__,
        'OpenCV': lambda: __import__('cv2').__version__,
        'MediaPipe': lambda: __import__('mediapipe').__version__,
        'NumPy': lambda: __import__('numpy').__version__,
        'PIL': lambda: __import__('PIL').__version__,
        'scikit-image': lambda: __import__('skimage').__version__,
    }
    
    logger.info("\nLibraries:")
    for name, check in checks.items():
        try:
            version = check()
            logger.info(f"✓ {name}: {version}")
        except Exception as e:
            logger.warning(f"✗ {name}: {e}")
    
    # Verify models directory - convert to Path object
    logger.info("\nDirectories:")
    pretrained_dir = Path(CONFIG.pretrained_dir)
    if pretrained_dir.exists():
        files = list(pretrained_dir.glob('*.pth'))
        logger.info(f"✓ Models directory exists: {pretrained_dir}")
        logger.info(f"   Files: {len(files)}")
        for f in files:
            size = os.path.getsize(f) / (1024 ** 2)
            logger.info(f"   - {f.name} ({size:.1f} MB)")
    else:
        logger.warning(f"✗ Models directory not found: {pretrained_dir}")


def estimate_performance():
    """Estimate expected performance"""
    logger.info("\n" + "=" * 60)
    logger.info("ESTIMATED PERFORMANCE (i5-1135G7, 8GB RAM)")
    logger.info("=" * 60)
    
    logger.info("\nImage Detection:")
    logger.info("  Inference time: 80-150ms per image")
    logger.info("  FPS: 6-12 FPS")
    logger.info("  RAM usage: 800MB - 1.5GB")
    
    logger.info("\nVideo Detection:")
    logger.info("  Frame skip: 6 (analyzes 5 FPS instead of 30)")
    logger.info("  Inference time: 15-25ms per frame")
    logger.info("  Effective FPS: ~5 analyzed FPS")
    logger.info("  5-minute video: ~10-15 minutes processing")
    logger.info("  RAM usage: 1.5GB - 2.5GB")
    
    logger.info("\nModels:")
    logger.info("  EfficientNet-B0: ~20MB")
    logger.info("  ResNet18: ~45MB")
    logger.info("  MediaPipe: ~5MB")
    logger.info("  Total: ~70MB")


def main():
    """Main execution"""
    logger.info("\n")
    logger.info("╔════════════════════════════════════════════════════╗")
    logger.info("║     DEEPFAKE DETECTION - MODELS INITIALIZATION     ║")
    logger.info("╚════════════════════════════════════════════════════╝")
    
    # Create directories - convert to Path object
    logger.info(f"\nCreating directories...")
    pretrained_dir = Path(CONFIG.pretrained_dir)
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Models directory: {pretrained_dir}")
    
    # Download PyTorch models
    download_pytorch_models()
    
    # Initialize MediaPipe
    initialize_mediapipe()
    
    # Verify installations
    verify_installations()
    
    # Estimations
    estimate_performance()
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ SETUP COMPLETE")
    logger.info("=" * 60)
    
    logger.info("\nNext steps:")
    logger.info("1. Test detection: python scripts/test_detection.py")
    logger.info("2. Start API: python backend/app.py")
    logger.info("3. Detect image: python -c \"from src.detector import quick_detect_image; print(quick_detect_image('image.jpg'))\"")
    logger.info("\nFor details, see README.md")


if __name__ == '__main__':
    main()