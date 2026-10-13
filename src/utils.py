"""
Utilitaires pour détection de deepfakes
- Téléchargement de modèles
- Chargement de poids
- Formatage résultats
- Calcul de métriques
- Visualisation heatmaps
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from urllib.request import urlopen
from io import BytesIO

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

from config import CONFIG

# Setup logging with UTF-8 support for Windows
import sys
logging.basicConfig(
    level=logging.INFO,
    format=CONFIG.logging.LOG_FORMAT,
    handlers=[
        logging.FileHandler(CONFIG.logging.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    encoding='utf-8'  # Python 3.9+
)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Gère le téléchargement et la vérification de modèles pré-entraînés"""
    
    @staticmethod
    def download_file(url: str, save_path: str, chunk_size: int = 8192) -> bool:
        """
        Télécharge fichier depuis URL avec barre de progression
        
        Args:
            url: URL du fichier
            save_path: Chemin local de destination
            chunk_size: Taille des chunks
            
        Returns:
            True si succès, False sinon
        """
        try:
            logger.info(f"Téléchargement: {url}")
            response = urlopen(url)
            total_size = int(response.headers.get('content-length', 0))
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            downloaded = 0
            with open(save_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        logger.info(f"Progression: {percent:.1f}%")
            
            logger.info(f"✓ Téléchargement réussi: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Erreur téléchargement: {e}")
            return False
    
    @staticmethod
    def verify_checksum(file_path: str, expected_hash: Optional[str] = None) -> bool:
        """Vérifie intégrité du fichier avec SHA256"""
        try:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            
            actual_hash = sha256.hexdigest()
            
            if expected_hash and actual_hash != expected_hash:
                logger.warning(f"Hash mismatch! Expected: {expected_hash}, Got: {actual_hash}")
                return False
            
            logger.info(f"✓ Checksum OK: {actual_hash[:16]}...")
            return True
            
        except Exception as e:
            logger.error(f"Erreur vérification checksum: {e}")
            return False


class ModelLoader:
    """Charge et initialise les modèles pré-entraînés"""
    
    @staticmethod
    def load_efficientnet_b0(device: str = 'cpu') -> torch.nn.Module:
        import timm
        import torch

        model = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=2
        )

        weights_path = Path(CONFIG.pretrained_dir) / "deepfake_efficientnet.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Modèle fine-tuné introuvable: {weights_path}\n"
                "Lancez d'abord: python main.py setup"
            )

        model.load_state_dict(
            torch.load(str(weights_path), map_location=device, weights_only=True)
        )

        model = model.to(device)
        model.eval()

        logger.info(f"✓ Fine-tuned EfficientNet-B0 chargé depuis {weights_path}")

        return model

    
    @staticmethod
    def load_resnet18(device: str = 'cpu', weights_path: Optional[str] = None) -> torch.nn.Module:
        """Charge ResNet18 pré-entraîné"""
        try:
            from torchvision import models
            model = models.resnet18(pretrained=True)
            
            if weights_path and os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
            
            model = model.to(device)
            model.eval()
            logger.info("✓ ResNet18 chargé")
            return model
            
        except Exception as e:
            logger.error(f"Erreur chargement ResNet18: {e}")
            raise
    
    @staticmethod
    def load_model_weights(model: torch.nn.Module, weights_path: str) -> bool:
        """Charge poids dans modèle existant"""
        try:
            if not os.path.exists(weights_path):
                logger.warning(f"Fichier poids non trouvé: {weights_path}")
                return False
            
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"✓ Poids chargés: {weights_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement poids: {e}")
            return False


def _generate_explanation(prediction: int, confidence: float, evidence: dict) -> str:
    """Génère une explication en langage naturel du verdict."""
    is_fake = prediction == 0
    conf_pct = round(confidence * 100)
    reasons = []

    # Niveau de confiance
    if confidence >= 0.90:
        conf_label = "très haute confiance"
    elif confidence >= 0.75:
        conf_label = "haute confiance"
    elif confidence >= 0.60:
        conf_label = "confiance modérée"
    else:
        conf_label = "faible confiance — résultat incertain"

    # Analyse texture
    texture = evidence.get("texture", {})
    homogeneity = texture.get("homogeneity")
    if homogeneity is not None:
        if is_fake and homogeneity > 0.40:
            reasons.append("texture cutanée anormalement homogène (artefact GAN)")
        elif is_fake and homogeneity < 0.20:
            reasons.append("texture incohérente avec des discontinuités locales")
        elif not is_fake and 0.20 <= homogeneity <= 0.40:
            reasons.append("texture naturelle dans les normes attendues")

    contrast = texture.get("contrast")
    if contrast is not None:
        if is_fake and contrast > 120:
            reasons.append("contraste local élevé typique des rendus synthétiques")
        elif not is_fake and contrast <= 120:
            reasons.append("contraste local cohérent avec une photo authentique")

    # Analyse fréquentielle
    frequency = evidence.get("frequency", {})
    fft_std = frequency.get("fft_std")
    if fft_std is not None:
        if is_fake and fft_std < 1.0:
            reasons.append("spectre fréquentiel trop régulier (signature de génération IA)")
        elif not is_fake and fft_std >= 1.0:
            reasons.append("distribution fréquentielle naturelle")

    # Visages
    num_faces = evidence.get("num_faces", 0)
    if num_faces == 0:
        reasons.append("aucun visage détecté — image entière analysée")
    elif num_faces > 1:
        reasons.append(f"{num_faces} visages analysés simultanément")

    # Construction phrase
    verdict = "synthétique (deepfake)" if is_fake else "authentique"
    base = f"Image classifiée comme {verdict} avec {conf_label} ({conf_pct}%)."

    if reasons:
        base += " Indices : " + ", ".join(reasons) + "."

    return base


class ResultFormatter:
    """Formate les résultats de détection"""
    
    @staticmethod
    def format_image_result(
        prediction: int,
        confidence: float,
        evidence: Optional[Dict] = None,
        processing_time_ms: float = 0.0,
        heatmap: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Formate résultat pour image
        
        Returns:
            Dict avec structure standardisée
        """
        result = {
            "result": "FAKE" if prediction == 0 else "REAL",
            "confidence": float(confidence),
            "fake_probability": float(confidence) if prediction == 0 else float(1 - confidence),
            "processing_time_ms": float(processing_time_ms),
            "evidence": evidence or {},
            "explanation": _generate_explanation(prediction, confidence, evidence or {}),
            "timestamp": str(np.datetime64('now'))
        }
        
        if heatmap is not None:
            result["has_heatmap"] = True
        
        return result
    
    @staticmethod
    def format_video_result(
        frame_results: List[Dict],
        overall_prediction: int,
        avg_confidence: float,
        processing_time_ms: float = 0.0,
        stats: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Formate résultat pour vidéo"""
        return {
            "overall_result": "FAKE" if overall_prediction == 0 else "REAL",
            "avg_confidence": float(avg_confidence),
            "fake_probability": float(avg_confidence) if overall_prediction == 0 else float(1 - avg_confidence),
            "total_frames_analyzed": len(frame_results),
            "processing_time_ms": float(processing_time_ms),
            "stats": stats or {},
            "frame_results": frame_results[:10],  # Retourner que premiers 10 frames
            "timestamp": str(np.datetime64('now'))
        }
    
    @staticmethod
    def to_json(result: Dict, pretty: bool = True) -> str:
        """Convertit résultat en JSON"""
        try:
            return json.dumps(result, indent=2 if pretty else None, default=str)
        except Exception as e:
            logger.error(f"Erreur JSON serialization: {e}")
            return "{}"


class MetricsCalculator:
    """Calcule métriques d'évaluation"""
    
    @staticmethod
    def calculate_metrics(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calcule métriques classiques
        
        Args:
            predictions: Prédictions binaires (0=REAL, 1=FAKE)
            ground_truth: Vraies étiquettes
            probabilities: Probabilités pour AUC (optionnel)
            
        Returns:
            Dict avec Accuracy, Precision, Recall, F1, AUC
        """
        metrics = {
            "accuracy": accuracy_score(ground_truth, predictions),
            "precision": precision_score(ground_truth, predictions, zero_division=0),
            "recall": recall_score(ground_truth, predictions, zero_division=0),
            "f1": f1_score(ground_truth, predictions, zero_division=0),
        }
        
        if probabilities is not None and len(np.unique(ground_truth)) == 2:
            metrics["auc_roc"] = roc_auc_score(ground_truth, probabilities)
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, int]:
        """Retourne confusion matrix sous forme dict"""
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
        return {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }


class VisualizationUtils:
    """Utilitaires de visualisation"""
    
    @staticmethod
    def create_heatmap(
        image: np.ndarray,
        activations: np.ndarray,
        colormap: str = 'jet',
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Crée heatmap des activations
        
        Args:
            image: Image originale (H, W, 3)
            activations: Activations (h, w) normalized [0, 1]
            
        Returns:
            Image avec heatmap overlayée
        """
        try:
            # Redimensionner activations à taille image
            from cv2 import resize, INTER_LINEAR, applyColorMap, COLORMAP_JET, addWeighted
            
            activations = resize(activations, (image.shape[1], image.shape[0]), INTER_LINEAR)
            
            # Normaliser et colormap
            activations_norm = ((activations - activations.min()) / 
                               (activations.max() - activations.min() + 1e-8) * 255).astype(np.uint8)
            heatmap = applyColorMap(activations_norm, COLORMAP_JET)
            
            # Overlay
            result = addWeighted(image, 1 - alpha, heatmap, alpha, 0)
            return result
            
        except Exception as e:
            logger.error(f"Erreur création heatmap: {e}")
            return image
    
    @staticmethod
    def plot_roc_curve(
        ground_truth: np.ndarray,
        probabilities: np.ndarray,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Génère courbe ROC"""
        try:
            fpr, tpr, _ = roc_curve(ground_truth, probabilities)
            auc = roc_auc_score(ground_truth, probabilities)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"✓ ROC curve saved: {save_path}")
            
            plt.close()
            return save_path
            
        except Exception as e:
            logger.error(f"Erreur plot ROC: {e}")
            return None


class PerformanceLogger:
    """Enregistre métriques de performance"""
    
    @staticmethod
    def log_inference_time(
        elapsed_time_ms: float,
        num_frames: int = 1,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """Enregistre temps d'inférence"""
        fps = num_frames / (elapsed_time_ms / 1000.0) if elapsed_time_ms > 0 else 0
        logger.info(f"Inference: {elapsed_time_ms:.1f}ms | FPS: {fps:.1f} | Size: {image_size}")
    
    @staticmethod
    def log_memory_usage(peak_memory_mb: float, process_name: str = "Inference"):
        """Enregistre utilisation mémoire"""
        logger.info(f"{process_name} - Peak Memory: {peak_memory_mb:.1f} MB")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)


    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[:, class_idx]
        loss.backward()

        gradients = self.gradients
        activations = self.activation

        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def batch_format_results(results: List[Dict]) -> List[Dict]:
    """Formate batch de résultats"""
    return [
        ResultFormatter.format_image_result(
            prediction=r.get('prediction', 0),
            confidence=r.get('confidence', 0.0),
            evidence=r.get('evidence'),
            processing_time_ms=r.get('time', 0.0)
        )
        for r in results
    ]


if __name__ == "__main__":
    # Test utils
    print("✓ Utils module loaded")