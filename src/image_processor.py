"""
Pipeline de traitement d'images pour détection de deepfakes
- Chargement et validation
- Détection de visages
- Normalisation et augmentation
- Extraction de features (texture, fréquence, landmarks)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import mediapipe as mp

from config import CONFIG

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Processus d'images avec détection visages et features"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialise le processeur
        
        Args:
            device: 'cpu', 'cuda', 'xpu'
        """
        self.device = device
        self.input_size = CONFIG.inference.INPUT_SIZE
        
        # Normalization ImageNet
        self.normalize = transforms.Normalize(
            mean=CONFIG.inference.IMAGENET_MEAN,
            std=CONFIG.inference.IMAGENET_STD
        )
        
        # Transformations base
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # MediaPipe Face Detector
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=CONFIG.model_config.MEDIAPIPE_FACE['model_selection'],
            min_detection_confidence=CONFIG.model_config.MEDIAPIPE_FACE['min_detection_confidence']
        )
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=CONFIG.model_config.MEDIAPIPE_MESH['max_num_faces'],
            min_detection_confidence=CONFIG.model_config.MEDIAPIPE_MESH['min_detection_confidence']
        )
        
        logger.info(f"ImageProcessor initialized (device: {device})")
    
    def load_image(self, image_path: str) -> Tuple[np.ndarray, bool]:
        """
        Charge image depuis fichier avec validation
        
        Args:
            image_path: Chemin vers image
            
        Returns:
            (image RGB numpy, succès)
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return None, False
            
            # Charger avec PIL pour meilleure robustesse
            pil_image = Image.open(image_path).convert('RGB')
            image = np.array(pil_image)
            
            logger.info(f"✓ Image loaded: {image_path.name} ({image.shape[0]}x{image.shape[1]})")
            return image, True
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None, False
    
    def load_image_from_bytes(self, image_bytes: bytes) -> Tuple[np.ndarray, bool]:
        """Charge image depuis bytes (upload)"""
        try:
            from io import BytesIO
            pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            image = np.array(pil_image)
            logger.info(f"✓ Image loaded from bytes: {image.shape}")
            return image, True
        except Exception as e:
            logger.error(f"Error loading image from bytes: {e}")
            return None, False
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Prétraite image pour modèle (redimensionne, normalise)
        
        Args:
            image: Image numpy RGB (H, W, 3)
            
        Returns:
            Tensor normalisé (1, 3, 224, 224)
        """
        try:
            # Convertir en PIL
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Appliquer transformations
            tensor = self.transform(pil_image)
            
            # Ajouter batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Détecte visages dans image avec MediaPipe
        
        Args:
            image: Image numpy RGB (H, W, 3)
            
        Returns:
            Liste de dict {bbox, confidence, landmarks}
        """
        try:
            h, w = image.shape[:2]
            
            # MediaPipe utilise RGB
            results = self.mp_face_detection.process(image)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convertir en pixels
                    x1 = max(0, int(bbox.xmin * w))
                    y1 = max(0, int(bbox.ymin * h))
                    x2 = min(w, int((bbox.xmin + bbox.width) * w))
                    y2 = min(h, int((bbox.ymin + bbox.height) * h))
                    
                    # Filtrer par taille minimum
                    face_w, face_h = x2 - x1, y2 - y1
                    if face_w < CONFIG.inference.MIN_FACE_SIZE or face_h < CONFIG.inference.MIN_FACE_SIZE:
                        continue
                    
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': detection.score[0],
                        'keypoints': self._extract_face_keypoints(detection)
                    })
            
            logger.info(f"✓ Detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _extract_face_keypoints(self, detection) -> Dict[str, Tuple[float, float]]:
        """Extrait keypoints d'une détection"""
        keypoints = {}
        if detection.location_data.relative_keypoints:
            for i, kp in enumerate(detection.location_data.relative_keypoints):
                keypoints[f"kp_{i}"] = (kp.x, kp.y)
        return keypoints
    
    def get_face_landmarks(self, image: np.ndarray, face_bbox: Tuple) -> Optional[np.ndarray]:
        """
        Détecte 468 landmarks faciaux avec MediaPipe FaceMesh
        
        Args:
            image: Image RGB
            face_bbox: (x1, y1, x2, y2)
            
        Returns:
            Landmarks (468, 3) ou None
        """
        try:
            results = self.mp_face_mesh.process(image)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                landmarks_array = np.array([
                    [lm.x, lm.y, lm.z] for lm in landmarks.landmark
                ])
                return landmarks_array
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def extract_face_patches(
        self,
        image: np.ndarray,
        faces: List[Dict],
        patch_size: int = 224
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Extrait patchs 224x224 pour chaque visage
        
        Args:
            image: Image RGB
            faces: Liste de visages détectés
            patch_size: Taille des patchs
            
        Returns:
            Liste (patch_RGB, metadata)
        """
        patches = []
        
        for i, face in enumerate(faces):
            try:
                x1, y1, x2, y2 = face['bbox']
                
                # Extraire patch
                patch = image[y1:y2, x1:x2].copy()
                
                # Redimensionner à 224x224
                patch_resized = cv2.resize(patch, (patch_size, patch_size))
                
                metadata = {
                    'face_id': i,
                    'original_bbox': face['bbox'],
                    'original_size': patch.shape[:2],
                    'confidence': face['confidence']
                }
                
                patches.append((patch_resized, metadata))
                
            except Exception as e:
                logger.warning(f"Error extracting patch {i}: {e}")
        
        logger.info(f"✓ Extracted {len(patches)} face patches")
        return patches
    
    def get_texture_features(self, image: np.ndarray, patch_size: int = 32) -> Dict[str, float]:
        """
        Analyse texture avec features Haralick
        - Détecte artefacts de compression, bords non-naturels
        - Compatible avec scikit-image 0.24+
        
        Args:
            image: Image RGB
            patch_size: Taille patchs GLCM
            
        Returns:
            Dict avec features texture
        """
        try:
            # Try new API first, fallback to old one
            try:
                from skimage.feature import graycomatrix, graycoprops
            except ImportError:
                from skimage.feature import greycomatrix as graycomatrix, greycoprops
            
            # Convertir en grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Normaliser 0-255
            gray = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype(np.uint8)
            
            # Calculer GLCM (Haralick features)
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256)
            
            # Features texture Haralick
            features = {
                'contrast': float(graycoprops(glcm, 'contrast')[0, 0]),
                'dissimilarity': float(graycoprops(glcm, 'dissimilarity')[0, 0]),
                'homogeneity': float(graycoprops(glcm, 'homogeneity')[0, 0]),
                'energy': float(graycoprops(glcm, 'energy')[0, 0]),
                'correlation': float(graycoprops(glcm, 'correlation')[0, 0]),
            }
            
            logger.debug(f"Texture features: {list(features.keys())}")
            return features
            
        except ImportError:
            # Fallback: use simple edge detection as texture proxy
            logger.debug("Using edge-based texture features (greycomatrix not available)")
            try:
                edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 100, 200)
                features = {
                    'edge_density': float(np.mean(edges) / 255.0),
                    'edge_variance': float(np.var(edges) / (255.0**2)),
                }
                return features
            except Exception as e:
                logger.warning(f"Texture feature extraction failed: {e}")
                return {}
        
        except Exception as e:
            logger.warning(f"Error computing texture features: {e}")
            return {}
    
    def get_frequency_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyse domaine fréquentiel avec FFT
        - Détecte patterns non-naturels, compression
        
        Args:
            image: Image RGB
            
        Returns:
            Dict avec features fréquentielles
        """
        try:
            # Convertir en grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            
            # FFT 2D
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Log scale
            magnitude_log = np.log1p(magnitude)
            
            # Features
            features = {
                'fft_mean': float(np.mean(magnitude_log)),
                'fft_std': float(np.std(magnitude_log)),
                'fft_max': float(np.max(magnitude_log)),
                'fft_energy': float(np.sum(magnitude ** 2) / magnitude.size),
            }
            
            # Énergie par quadrants (détecte patterns directionnels)
            h, w = magnitude.shape
            q1 = np.sum(magnitude[:h//2, :w//2])
            q2 = np.sum(magnitude[:h//2, w//2:])
            q3 = np.sum(magnitude[h//2:, :w//2])
            q4 = np.sum(magnitude[h//2:, w//2:])
            
            total = q1 + q2 + q3 + q4
            features.update({
                'fft_q1_ratio': float(q1 / total) if total > 0 else 0,
                'fft_q2_ratio': float(q2 / total) if total > 0 else 0,
                'fft_q3_ratio': float(q3 / total) if total > 0 else 0,
                'fft_q4_ratio': float(q4 / total) if total > 0 else 0,
            })
            
            return features
            
        except Exception as e:
            logger.warning(f"Error computing frequency features: {e}")
            return {}
    
    def get_geometry_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Analyse géométrie faciale pour détecter distorsions
        
        Args:
            landmarks: Array (468, 3) de landmarks
            
        Returns:
            Dict avec features géométriques
        """
        try:
            if landmarks is None or len(landmarks) < 10:
                return {}
            
            # Distancer entre points clés
            features = {}
            
            # Symétrie oeil-oeil
            left_eye_center = np.mean(landmarks[33:42], axis=0)  # Points oeil gauche
            right_eye_center = np.mean(landmarks[362:371], axis=0)  # Points oeil droit
            
            eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
            features['eye_distance'] = float(eye_distance)
            
            # Symétrie du visage
            left_face = landmarks[100:136]  # Contour gauche
            right_face = landmarks[365:401]  # Contour droit
            
            if len(left_face) > 0 and len(right_face) > 0:
                symmetry_error = np.mean(np.linalg.norm(left_face - right_face, axis=1))
                features['symmetry_error'] = float(symmetry_error)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error computing geometry features: {e}")
            return {}
    
    def augment_image(self, image: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Augmentation légère (optionnel, pas utilisé pour inférence)
        
        Args:
            image: Image RGB
            strength: Intensité augmentation
            
        Returns:
            Image augmentée
        """
        try:
            # Brightness
            if np.random.rand() > 0.5:
                brightness = 1 + np.random.uniform(-strength, strength)
                image = np.clip(image * brightness, 0, 255).astype(np.uint8)
            
            # Contrast
            if np.random.rand() > 0.5:
                image = cv2.convertScaleAbs(image, alpha=1 + np.random.uniform(-strength, strength), beta=0)
            
            return image
            
        except Exception as e:
            logger.warning(f"Error augmenting image: {e}")
            return image
    
    def get_all_features(self, image: np.ndarray) -> Dict:
        """Extrait tous les features pour analyse complète"""
        return {
            'texture': self.get_texture_features(image),
            'frequency': self.get_frequency_features(image),
        }


def process_image_for_inference(
    image_path: str,
    processor: ImageProcessor
) -> Tuple[List[torch.Tensor], List[Dict], bool]:
    """
    Pipeline complet de préparation image pour inférence
    
    Returns:
        (patches_tensors, face_metadata, succès)
    """
    try:
        # Charger image
        image, success = processor.load_image(image_path)
        if not success:
            return [], [], False
        
        # Détecter visages
        faces = processor.detect_faces(image)
        if not faces:
            logger.warning("No faces detected")
            return [], [], False
        
        # Extraire patchs
        patches = processor.extract_face_patches(image, faces)
        
        # Prétraiter chaque patch
        patch_tensors = []
        patch_metadata = []
        
        for patch, metadata in patches:
            tensor = processor.preprocess(patch)
            if tensor is not None:
                patch_tensors.append(tensor)
                patch_metadata.append(metadata)
        
        return patch_tensors, patch_metadata, True
        
    except Exception as e:
        logger.error(f"Error in image processing pipeline: {e}")
        return [], [], False


if __name__ == "__main__":
    processor = ImageProcessor(device='cpu')
    print("✓ ImageProcessor initialized")
