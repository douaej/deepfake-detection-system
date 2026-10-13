"""
Deepfake Detection Module
Main package for deepfake and fake image detection
"""

__version__ = "1.0.0"
__author__ = "Deep Learning Team"

from src.detector import DeepfakeDetector
from src.image_processor import ImageProcessor
from src.video_processor import VideoProcessor

__all__ = ["DeepfakeDetector", "ImageProcessor", "VideoProcessor"]
