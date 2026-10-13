"""
Configuration centralisée - Deepfake Detection System
Optimisé pour: Intel i5-1135G7 @ 2.40GHz, 8GB RAM, Intel Iris Xe Graphics
"""

import os
from pathlib import Path
from enum import Enum

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

for dir_path in [MODELS_DIR, PRETRAINED_DIR, DATA_DIR, OUTPUTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ==================== HARDWARE AUTO-DETECTION ====================
def _detect_best_device() -> str:
    """
    Détecte automatiquement le meilleur device disponible.
    Priorité: XPU (Intel) > CUDA > MPS (Apple) > CPU
    """
    try:
        import torch
        # Intel XPU (Intel Arc / Iris Xe avec drivers oneAPI)
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                return "xpu"
        except ImportError:
            pass

        if torch.cuda.is_available():
            return "cuda"

        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


# Forcer via variable d'environnement, sinon auto-détection
DEVICE = os.getenv("DEVICE", _detect_best_device())

# Paramètres mémoire adaptés à 8GB RAM
ENABLE_MEMORY_OPTIMIZATION = True
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))       # Max 2 images simultanées
FRAME_BATCH_SIZE = int(os.getenv("FRAME_BATCH_SIZE", "1"))  # 1 frame à la fois


# ==================== MODEL CONFIG ====================
class ModelConfig:
    """Configuration des modèles pré-entraînés"""

    # EfficientNet-B0 (classificateur principal, ~20MB)
    EFFICIENTNET_B0 = {
        "name": "efficientnet_b0",
        "source": "timm",
        "pretrained": True,
        "num_classes": 2,
        "input_size": 224,
        "local_path": str(PRETRAINED_DIR / "efficientnet_b0.pth"),
    }

    # ResNet18 (extraction de features, ~45MB)
    RESNET18 = {
        "name": "resnet18",
        "source": "torchvision",
        "pretrained": True,
        "num_classes": 1000,
        "input_size": 224,
        "local_path": str(PRETRAINED_DIR / "resnet18.pth"),
    }

    # MediaPipe Face Detection
    MEDIAPIPE_FACE = {
        "name": "mediapipe_face_detection",
        "source": "mediapipe",
        "min_detection_confidence": 0.5,
        "model_selection": 1,  # 0=lite, 1=full
    }

    # MediaPipe Face Mesh (468 landmarks)
    MEDIAPIPE_MESH = {
        "name": "mediapipe_face_mesh",
        "source": "mediapipe",
        "static_image_mode": True,  # True pour images, False pour vidéo temps-réel
        "max_num_faces": 5,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }


# ==================== INFERENCE CONFIG ====================
class InferenceConfig:
    """Paramètres d'inférence optimisés i5-1135G7"""

    # Thresholds de décision
    CONFIDENCE_THRESHOLD = 0.50
    FAKE_THRESHOLD = 0.50

    # Normalisation ImageNet standard
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Processing image
    INPUT_SIZE = 224
    MIN_FACE_SIZE = 50       # pixels minimum détectable
    MAX_FACES = 5            # Limiter pour économiser RAM

    # Processing vidéo (adapté 8GB RAM)
    VIDEO_FPS_TARGET = 5     # Analyser 5fps au lieu de 30fps → économie 6x
    FRAME_SKIP = 6           # Traiter 1 frame sur 6
    MAX_VIDEO_DURATION = 300 # 5 minutes max
    MAX_VIDEO_FRAMES = 1500  # ~300s × 5fps

    # Agrégation résultats vidéo
    AGGREGATION_METHOD = "weighted_mean"  # "mean", "max", "percentile"
    CONFIDENCE_PERCENTILE = 90

    # Outputs
    RETURN_HEATMAP = True
    RETURN_FEATURES = False
    RETURN_INTERMEDIATE = False


# ==================== OPTIMIZATION CONFIG ====================
class OptimizationConfig:
    """Optimisations pour CPU i5-1135G7"""

    # Quantization INT8 → réduit consommation mémoire ~4x
    ENABLE_QUANTIZATION = True
    QUANTIZATION_TYPE = "dynamic"  # "dynamic" est le plus stable sur CPU

    # ONNX Runtime (optionnel, ~2x plus rapide que PyTorch CPU)
    ENABLE_ONNX = False
    ONNX_MODEL_PATH = str(PRETRAINED_DIR / "model.onnx")

    # Optimisations mémoire
    ENABLE_GRADIENT_CHECKPOINTING = False
    ENABLE_MIXED_PRECISION = False  # Pas utile sur CPU pur
    ENABLE_CHANNELS_LAST = True    # Peut améliorer perfs CPU

    # Cache détection visages (évite de re-détecter frames identiques)
    ENABLE_FACE_DETECTION_CACHE = True
    CACHE_MAX_SIZE = 500   # Réduit à 500 pour 8GB RAM

    # Nombre de threads PyTorch (auto = nombre de cœurs logiques)
    # i5-1135G7 = 4 cœurs / 8 threads
    NUM_THREADS = int(os.getenv("PYTORCH_NUM_THREADS", "4"))


# ==================== API CONFIG ====================
class APIConfig:
    """Configuration serveur Flask"""

    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "5000"))
    DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
    UPLOAD_FOLDER = str(OUTPUTS_DIR / "uploads")
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp", "mp4", "avi", "mov", "mkv"}

    # Timeouts adaptés aux performances CPU
    REQUEST_TIMEOUT = 300   # 5 minutes
    INFERENCE_TIMEOUT = 180 # 3 minutes (marge plus large sur CPU)

    # CORS (mettre False en production)
    CORS_ENABLED = True


# ==================== ANALYSIS CONFIG ====================
class AnalysisConfig:
    """Features d'analyse supplémentaires"""

    # Texture (Haralick GLCM)
    ENABLE_TEXTURE_ANALYSIS = True
    GLCM_DISTANCES = [1, 2, 3]

    # Analyse fréquentielle (FFT)
    ENABLE_FREQUENCY_ANALYSIS = True
    FFT_COMPONENTS = 64

    # Landmarks géométriques
    ENABLE_LANDMARK_ANALYSIS = True
    LANDMARK_CONFIDENCE_THRESHOLD = 0.7


# ==================== LOGGING CONFIG ====================
class LoggingConfig:
    """Configuration logging"""

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = str(LOGS_DIR / "deepfake_detection.log")
    ENABLE_PERFORMANCE_LOG = True
    MAX_LOG_SIZE_MB = 50
    BACKUP_COUNT = 3


# ==================== AUDIO CONFIG ====================
class AudioConfig:
    """Configuration analyse audio (future feature)"""
    ENABLE_AUDIO_ANALYSIS = False
    SAMPLE_RATE = 16000
    MFCC_N_MELS = 40


# ==================== AUGMENTATION CONFIG ====================
class AugmentationConfig:
    """Augmentation données (pour fine-tuning uniquement, pas inférence)"""
    ENABLE_AUGMENTATION = False
    AUGMENTATION_PARAMS = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "rotation": 10,
        "zoom": 0.1,
    }


# ==================== CONFIG GLOBALE ====================
class CONFIG:
    """Configuration globale combinée"""

    # Hardware
    device = DEVICE
    enable_memory_optimization = ENABLE_MEMORY_OPTIMIZATION
    batch_size = BATCH_SIZE
    frame_batch_size = FRAME_BATCH_SIZE

    # Paths
    base_dir = str(BASE_DIR)
    models_dir = str(MODELS_DIR)
    pretrained_dir = str(PRETRAINED_DIR)
    data_dir = str(DATA_DIR)
    outputs_dir = str(OUTPUTS_DIR)
    logs_dir = str(LOGS_DIR)

    # Sous-configurations
    model_config = ModelConfig
    inference = InferenceConfig
    api = APIConfig
    optimization = OptimizationConfig
    analysis = AnalysisConfig
    logging = LoggingConfig
    audio = AudioConfig
    augmentation = AugmentationConfig


# ==================== DEBUG ====================
def print_config():
    """Affiche la configuration actuelle"""
    print("=" * 60)
    print("DEEPFAKE DETECTION - CONFIGURATION")
    print("=" * 60)
    print(f"Device détecté:        {CONFIG.device}")
    print(f"Batch Size:            {CONFIG.batch_size}")
    print(f"Dossier modèles:       {CONFIG.pretrained_dir}")
    print(f"Quantization:          {CONFIG.optimization.ENABLE_QUANTIZATION}")
    print(f"Video FPS cible:       {CONFIG.inference.VIDEO_FPS_TARGET}")
    print(f"Confidence Threshold:  {CONFIG.inference.CONFIDENCE_THRESHOLD}")
    print(f"Threads PyTorch:       {CONFIG.optimization.NUM_THREADS}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()