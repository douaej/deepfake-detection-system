"""
Pipeline de traitement vidéo pour détection de deepfakes
- Extraction frames efficace
- Traitement batch
- Agrégation résultats par frame
- Gestion mémoire pour vidéos longues
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm

from config import CONFIG

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Pipeline de traitement vidéo avec gestion mémoire optimisée"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialise le processeur vidéo
        
        Args:
            device: 'cpu', 'cuda', 'xpu'
        """
        self.device = device
        self.frame_skip = CONFIG.inference.FRAME_SKIP
        self.max_frames = CONFIG.inference.MAX_VIDEO_FRAMES
        
        logger.info(f"VideoProcessor initialized (frame_skip: {self.frame_skip})")
    
    def load_video(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict, bool]:
        """
        Ouvre vidéo et retourne metadata
        
        Args:
            video_path: Chemin vidéo
            
        Returns:
            (cap, metadata, succès)
        """
        try:
            video_path = Path(video_path)
            
            if not video_path.exists():
                logger.error(f"Video not found: {video_path}")
                return None, {}, False
            
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None, {}, False
            
            # Récupérer metadata
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = total_frames / fps if fps > 0 else 0
            
            metadata = {
                'path': str(video_path.name),
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration_seconds': duration,
            }
            
            logger.info(f"✓ Video loaded: {metadata}")
            return cap, metadata, True
            
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return None, {}, False
    
    def extract_frames(
        self,
        cap: cv2.VideoCapture,
        metadata: Dict,
        step: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[int], bool]:
        """
        Extrait frames avec skipping pour efficacité
        
        Args:
            cap: VideoCapture object
            metadata: Video metadata
            step: Frame skip (défaut: CONFIG.FRAME_SKIP)
            
        Returns:
            (frames_list, frame_indices, succès)
        """
        try:
            if step is None:
                step = self.frame_skip
            
            frames = []
            frame_indices = []
            frame_count = 0
            
            total_frames = metadata['total_frames']
            
            logger.info(f"Extracting frames (step={step})...")
            
            with tqdm(total=total_frames // step, desc="Extracting frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Traiter 1 frame tous les 'step' frames
                    if frame_count % step == 0:
                        # Convertir BGR → RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                        frame_indices.append(frame_count)
                        pbar.update(1)
                        
                        # Limiter nombre frames
                        if len(frames) >= self.max_frames:
                            logger.warning(f"Max frames reached: {len(frames)}")
                            break
                    
                    frame_count += 1
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset
            
            logger.info(f"✓ Extracted {len(frames)} frames")
            return frames, frame_indices, True
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return [], [], False
    
    def extract_frames_buffered(
        self,
        cap: cv2.VideoCapture,
        metadata: Dict,
        buffer_size: int = 10,
        step: Optional[int] = None
    ):
        """
        Générateur pour extraire frames par buffer
        Économe en RAM pour vidéos longues
        
        Args:
            cap: VideoCapture
            metadata: Video metadata
            buffer_size: Nombre de frames par buffer
            step: Frame skip
            
        Yields:
            (buffer_frames, buffer_indices)
        """
        if step is None:
            step = self.frame_skip
        
        frames_buffer = []
        indices_buffer = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    if frames_buffer:
                        yield frames_buffer, indices_buffer
                    break
                
                if frame_count % step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_buffer.append(frame_rgb)
                    indices_buffer.append(frame_count)
                    
                    if len(frames_buffer) >= buffer_size:
                        yield frames_buffer, indices_buffer
                        frames_buffer = []
                        indices_buffer = []
                
                frame_count += 1
                
                if len(indices_buffer) >= self.max_frames:
                    if frames_buffer:
                        yield frames_buffer, indices_buffer
                    break
        
        except Exception as e:
            logger.error(f"Error in buffered extraction: {e}")
    
    def detect_faces_in_frames(
        self,
        frames: List[np.ndarray],
        face_detector
    ) -> Dict[int, List[Dict]]:
        """
        Détecte visages dans chaque frame
        
        Args:
            frames: Liste frames RGB
            face_detector: Fonction détection (doit accepter image)
            
        Returns:
            Dict {frame_idx: [faces]}
        """
        try:
            faces_per_frame = {}
            
            logger.info("Detecting faces in frames...")
            
            for i, frame in enumerate(tqdm(frames, desc="Face detection")):
                try:
                    faces = face_detector(frame)
                    faces_per_frame[i] = faces
                except Exception as e:
                    logger.warning(f"Error detecting faces in frame {i}: {e}")
                    faces_per_frame[i] = []
            
            logger.info(f"✓ Face detection complete")
            return faces_per_frame
            
        except Exception as e:
            logger.error(f"Error in face detection pipeline: {e}")
            return {}
    
    def extract_face_patches_from_frames(
        self,
        frames: List[np.ndarray],
        faces_per_frame: Dict[int, List[Dict]],
        patch_size: int = 224
    ) -> List[Dict]:
        """
        Extrait patchs visage de chaque frame
        
        Args:
            frames: Liste frames
            faces_per_frame: Dict {frame_idx: [faces]}
            patch_size: Taille patchs
            
        Returns:
            Liste {frame_idx, face_id, patch, bbox}
        """
        try:
            patches_list = []
            
            for frame_idx, frame in enumerate(frames):
                if frame_idx not in faces_per_frame:
                    continue
                
                for face_id, face in enumerate(faces_per_frame[frame_idx]):
                    try:
                        x1, y1, x2, y2 = face['bbox']
                        patch = frame[y1:y2, x1:x2].copy()
                        
                        # Redimensionner
                        patch_resized = cv2.resize(patch, (patch_size, patch_size))
                        
                        patches_list.append({
                            'frame_idx': frame_idx,
                            'face_id': face_id,
                            'patch': patch_resized,
                            'bbox': (x1, y1, x2, y2),
                            'confidence': face.get('confidence', 0.0)
                        })
                    except Exception as e:
                        logger.warning(f"Error extracting patch frame {frame_idx}: {e}")
            
            logger.info(f"✓ Extracted {len(patches_list)} face patches from video")
            return patches_list
            
        except Exception as e:
            logger.error(f"Error extracting patches: {e}")
            return []
    
    def process_batch(
        self,
        frames: List[np.ndarray],
        preprocess_fn,
        batch_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Traite batch de frames
        
        Args:
            frames: Liste frames
            preprocess_fn: Fonction de prétraitement
            batch_size: Taille batch (défaut: CONFIG.FRAME_BATCH_SIZE)
            
        Returns:
            Liste {frame_idx, tensor}
        """
        try:
            if batch_size is None:
                batch_size = CONFIG.frame_batch_size
            
            preprocessed = []
            
            for i, frame in enumerate(tqdm(frames, desc="Preprocessing")):
                try:
                    tensor = preprocess_fn(frame)
                    if tensor is not None:
                        preprocessed.append({
                            'frame_idx': i,
                            'tensor': tensor,
                            'original_size': frame.shape[:2]
                        })
                except Exception as e:
                    logger.warning(f"Error preprocessing frame {i}: {e}")
            
            logger.info(f"✓ Preprocessed {len(preprocessed)}/{len(frames)} frames")
            return preprocessed
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return []
    
    def aggregate_frame_results(
        self,
        frame_results: List[Dict],
        method: Optional[str] = None
    ) -> Dict:
        """
        Agrège résultats par frame en prédiction vidéo unique
        
        Args:
            frame_results: Liste {frame_idx, prediction, confidence}
            method: "mean", "max", "percentile"
            
        Returns:
            Dict avec prédiction globale et stats
        """
        try:
            if not frame_results:
                return {
                    'overall_prediction': -1,
                    'avg_confidence': 0.0,
                    'stats': {}
                }
            
            if method is None:
                method = CONFIG.inference.AGGREGATION_METHOD
            
            confidences = np.array([r['confidence'] for r in frame_results])
            predictions = np.array([r['prediction'] for r in frame_results])
            
            # Calculer statistiques — fake=0, real=1
            stats = {
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'min_confidence': float(np.min(confidences)),
                'max_confidence': float(np.max(confidences)),
                'num_frames': len(frame_results),
                'fake_count': int(np.sum(predictions == 0)),
                'fake_ratio': float(np.sum(predictions == 0) / len(predictions)) if len(predictions) > 0 else 0.0,
            }
            
            # Agrégation
            if method == "mean":
                avg_confidence = np.mean(confidences)
            elif method == "max":
                avg_confidence = np.max(confidences)
            elif method == "percentile":
                percentile = CONFIG.inference.CONFIDENCE_PERCENTILE
                avg_confidence = np.percentile(confidences, percentile)
            else:
                avg_confidence = np.mean(confidences)
            
            # Prédiction globale — vote majoritaire fake=0
            fake_votes = int(np.sum(predictions == 0))
            overall_prediction = 0 if fake_votes > len(predictions) / 2 else 1
            
            return {
                'overall_prediction': overall_prediction,
                'avg_confidence': float(avg_confidence),
                'stats': stats,
            }
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            return {
                'overall_prediction': -1,
                'avg_confidence': 0.0,
                'stats': {}
            }
    
    def get_frame_difficulty_score(
        self,
        frames: List[np.ndarray]
    ) -> List[float]:
        """
        Estime difficulté de chaque frame
        Frames floues = plus faciles à analyser
        
        Args:
            frames: Liste frames
            
        Returns:
            Liste scores difficulté [0, 1]
        """
        try:
            difficulty_scores = []
            
            for frame in frames:
                # Laplacian variance (mesure flou)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Normaliser [0, 1] (inverse: plus flou = score bas)
                score = min(1.0, laplacian_var / 1000.0)
                difficulty_scores.append(score)
            
            return difficulty_scores
            
        except Exception as e:
            logger.warning(f"Error computing difficulty scores: {e}")
            return [0.5] * len(frames)


def process_video_for_inference(
    video_path: str,
    processor: 'VideoProcessor',
    face_detector,
    image_processor
) -> Tuple[List[Dict], Dict, bool]:
    """
    Pipeline complet de préparation vidéo pour inférence
    
    Returns:
        (patches_list, video_metadata, succès)
    """
    try:
        # Charger vidéo
        cap, metadata, success = processor.load_video(video_path)
        if not success:
            return [], {}, False
        
        # Extraire frames
        frames, frame_indices, success = processor.extract_frames(cap, metadata)
        cap.release()
        
        if not success or not frames:
            return [], metadata, False
        
        # Détecter visages
        faces_per_frame = processor.detect_faces_in_frames(frames, face_detector)
        
        # Extraire patchs
        patches = processor.extract_face_patches_from_frames(frames, faces_per_frame)
        
        metadata['extracted_frames'] = len(frames)
        metadata['total_patches'] = len(patches)
        
        return patches, metadata, True
        
    except Exception as e:
        logger.error(f"Error in video processing pipeline: {e}")
        return [], {}, False


if __name__ == "__main__":
    processor = VideoProcessor(device='cpu')
    print("✓ VideoProcessor initialized")