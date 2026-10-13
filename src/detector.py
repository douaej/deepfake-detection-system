"""
Classe principale de détection de deepfakes
Pipeline complet: chargement → prétraitement → inférence → résultats
Optimisé pour i5-1135G7 / 8GB RAM
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CONFIG
from src.image_processor import ImageProcessor
from src.video_processor import VideoProcessor
from src.utils import ModelLoader, ResultFormatter, GradCAM, VisualizationUtils

logger = logging.getLogger(__name__)

# Configurer le nombre de threads PyTorch pour i5-1135G7
torch.set_num_threads(CONFIG.optimization.NUM_THREADS)
logger.debug(f"PyTorch threads: {torch.get_num_threads()}")


class DeepfakeDetector:
    """
    Détecteur de deepfakes principal.

    Stratégie multi-modèles:
    - EfficientNet-B0 : classification binaire REAL/FAKE
    - ResNet18 : extraction de features pour analyse
    - MediaPipe : détection et tracking de visages

    Optimisations pour 8GB RAM:
    - Quantization INT8 dynamique
    - Batch size réduit (2)
    - Traitement par patch unitaire
    - Libération mémoire explicite
    """

    def __init__(
        self,
        device: str = None,
        batch_size: Optional[int] = None,
        enable_quantization: bool = True,
        load_models: bool = True
    ):
        self.device = device or CONFIG.device
        self.batch_size = batch_size or CONFIG.batch_size
        self.enable_quantization = enable_quantization and CONFIG.optimization.ENABLE_QUANTIZATION
        self._lock = threading.Lock()  # Thread-safety pour l'API

        # Processeurs
        self.image_processor = ImageProcessor(device=self.device)
        self.video_processor = VideoProcessor(device=self.device)

        # Modèles (chargés à la demande ou à l'init)
        self.model_efficientnet = None
        self.model_resnet = None
        self._models_loaded = False

        if load_models:
            self._load_models()

        logger.info(f"✓ DeepfakeDetector initialisé (device={self.device}, "
                    f"quantization={self.enable_quantization})")

    # ─── Chargement des modèles ───────────────────────────────────────────────

    def _load_models(self):
        """Charge modèles avec gestion d'erreur robuste"""
        try:
            logger.info("Chargement des modèles...")

            self.model_efficientnet = ModelLoader.load_efficientnet_b0(device=self.device)
            self.model_resnet = ModelLoader.load_resnet18(device=self.device)

            # GradCAM initialisé sur le modèle AVANT quantization
            # La quantization INT8 casse les gradients -> on garde une copie non-quantizée
            try:
                target_layer = self.model_efficientnet.blocks[-1]
                self.gradcam = GradCAM(self.model_efficientnet, target_layer)
                self._gradcam_model = self.model_efficientnet
                logger.info("✓ GradCAM initialisé")
            except Exception as e:
                self.gradcam = None
                self._gradcam_model = None
                logger.warning(f"GradCAM non disponible: {e}")

            # Quantization sur une COPIE séparée pour l'inférence rapide
            if self.enable_quantization and self.device == "cpu":
                import copy
                self._model_efficientnet_infer = copy.deepcopy(self.model_efficientnet)
                from torch.quantization import quantize_dynamic
                self._model_efficientnet_infer = quantize_dynamic(
                    self._model_efficientnet_infer, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("✓ Modèle quantizé (inférence) + modèle original (GradCAM)")
            else:
                self._model_efficientnet_infer = self.model_efficientnet

            self._models_loaded = True
            logger.info("✓ Modèles chargés")

        except Exception as e:
            logger.error(f"Erreur chargement modèles: {e}")
            raise RuntimeError(
                "Impossible de charger les modèles. Lancez d'abord:\n"
                "  python main.py setup"
            ) from e

    def _ensure_models_loaded(self):
        """Charge les modèles si pas encore fait (lazy loading)"""
        if not self._models_loaded:
            self._load_models()

    def _apply_quantization(self):
        """Quantization INT8 dynamique → réduit RAM ~50%, légère perte de précision"""
        try:
            from torch.quantization import quantize_dynamic

            self.model_efficientnet = quantize_dynamic(
                self.model_efficientnet,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            self.model_resnet = quantize_dynamic(
                self.model_resnet,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("✓ Quantization INT8 appliquée")

        except Exception as e:
            logger.warning(f"Quantization ignorée: {e}")

    # ─── Inférence ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_efficientnet(self, image_tensor: torch.Tensor) -> Tuple[int, float]:
        """
        Inférence EfficientNet-B0 pour classification REAL/FAKE.
        Utilise le modèle quantizé si disponible (plus rapide).

        Returns:
            (prediction: 0=FAKE / 1=REAL, confidence: float 0-1)
        """
        try:
            # Utiliser le modèle quantizé pour l'inférence si disponible
            infer_model = getattr(self, '_model_efficientnet_infer', self.model_efficientnet)
            image_tensor = image_tensor.to(self.device)
            outputs = infer_model(image_tensor)
            probs = F.softmax(outputs, dim=1)

            # ImageFolder trie les classes alphabétiquement :
            # → classe 0 = "fake", classe 1 = "real"
            fake_prob = probs[0, 0].item()
            real_prob = probs[0, 1].item()
            # fake=0 → si fake_prob dominant, prediction=0 (FAKE)
            prediction = 0 if fake_prob >= CONFIG.inference.FAKE_THRESHOLD else 1
            confidence = max(real_prob, fake_prob)

            return prediction, confidence

        except Exception as e:
            logger.error(f"Erreur inférence EfficientNet: {e}")
            return -1, 0.0

    @torch.no_grad()
    def _infer_resnet_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extraction de features via ResNet18 (jusqu'au layer avgpool).

        Returns:
            features: np.ndarray (512,)
        """
        try:
            image_tensor = image_tensor.to(self.device)
            features = None
            x = image_tensor

            for name, module in self.model_resnet.named_children():
                if name == "fc":
                    features = x.view(x.size(0), -1).cpu().numpy().flatten()
                    break
                x = module(x)

            return features if features is not None else np.array([])

        except Exception as e:
            logger.error(f"Erreur extraction features ResNet: {e}")
            return np.array([])

    # ─── Détection image ──────────────────────────────────────────────────────

    def detect_image(
        self,
        image_input: Union[str, bytes, "Image.Image", np.ndarray],
        return_features: bool = False,
        return_heatmap: bool = False
    ) -> Dict:
        """
        Détecte si une image est un deepfake.

        Args:
            image_input: Chemin (str), bytes, PIL Image, ou numpy array RGB
            return_features: Inclure features dans la réponse
            return_heatmap: Inclure heatmap dans la réponse

        Returns:
            {
                result: "REAL" | "FAKE",
                confidence: float,
                fake_probability: float,
                processing_time_ms: float,
                evidence: { num_faces, texture, frequency, ... }
            }
        """
        start_time = time.time()
        self._ensure_models_loaded()

        with self._lock:
            try:
                # Charger image
                image = self._load_image_input(image_input)
                if image is None:
                    logger.warning("Impossible de charger l'image")
                    return ResultFormatter.format_image_result(
                        prediction=-1, confidence=0.0,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

                # Détecter visages
                faces = self.image_processor.detect_faces(image)
                if not faces:
                    logger.warning("Aucun visage détecté dans l'image")
                    return ResultFormatter.format_image_result(
                        prediction=0, confidence=0.0,
                        evidence={"num_faces": 0, "message": "No face detected"},
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

                # Limiter le nombre de visages traités
                faces = faces[:CONFIG.inference.MAX_FACES]

                # Extraire patchs visage
                patches = self.image_processor.extract_face_patches(image, faces)
                if not patches:
                    return ResultFormatter.format_image_result(
                        prediction=0, confidence=0.0,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

                # Inférence sur chaque patch
                predictions, confidences, features_list = [], [], []

                for patch, metadata in patches:
                    tensor = self.image_processor.preprocess(patch)
                    if tensor is None:
                        continue

                    # Inférence sans gradient (optimisé RAM)
                    pred, conf = self._infer_efficientnet(tensor)

                    # GradCAM nécessite les gradients → appelé séparément hors no_grad
                    heatmap_image = None
                    if return_heatmap and self.gradcam is not None:
                        try:
                            tensor_grad = tensor.clone().requires_grad_(True).to(self.device)
                            cam = self.gradcam.generate(tensor_grad)
                            heatmap_image = VisualizationUtils.create_heatmap(patch, cam)
                        except Exception as e:
                            logger.warning(f"GradCAM échoué (ignoré): {e}")

                    predictions.append(pred)
                    confidences.append(conf)

                    if return_features:
                        feats = self._infer_resnet_features(tensor)
                        features_list.append(feats.tolist() if feats.size > 0 else [])

                if not confidences:
                    return ResultFormatter.format_image_result(
                        prediction=0, confidence=0.0,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

                # fake=0, real=1 → vote majoritaire sur les patches
                avg_confidence = float(np.mean(confidences))
                fake_votes = sum(1 for p in predictions if p == 0)
                overall_prediction = 0 if fake_votes > len(predictions) / 2 else 1

                # Evidence
                evidence = {
                    "num_faces": len(faces),
                    "patches_analyzed": len(predictions),
                    "avg_confidence": avg_confidence,
                    "face_confidences": [round(float(c), 4) for c in confidences],
                }

                if CONFIG.analysis.ENABLE_TEXTURE_ANALYSIS:
                    evidence["texture"] = self.image_processor.get_texture_features(image)

                if CONFIG.analysis.ENABLE_FREQUENCY_ANALYSIS:
                    evidence["frequency"] = self.image_processor.get_frequency_features(image)

                if return_features and features_list:
                    evidence["features"] = features_list

                if return_heatmap and heatmap_image is not None:
                    try:
                        import base64
                        from io import BytesIO
                        from PIL import Image as _PILImage
                        import numpy as _np
                        buf = BytesIO()
                        _PILImage.fromarray(heatmap_image.astype(_np.uint8)).save(buf, format="PNG")
                        evidence["heatmap_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
                    except Exception as _e:
                        print(f"[heatmap] encodage échoué: {_e}")


                elapsed = (time.time() - start_time) * 1000
                logger.info(f"✓ Image analysée en {elapsed:.0f}ms → "
                            f"{'FAKE' if overall_prediction == 1 else 'REAL'} "
                            f"({avg_confidence:.1%})")

                return ResultFormatter.format_image_result(
                    prediction=overall_prediction,
                    confidence=avg_confidence,
                    evidence=evidence,
                    processing_time_ms=elapsed
                )

            except Exception as e:
                logger.error(f"Erreur detect_image: {e}", exc_info=True)
                return ResultFormatter.format_image_result(
                    prediction=-1, confidence=0.0,
                    evidence={"error": str(e)},
                    processing_time_ms=(time.time() - start_time) * 1000
                )

    # ─── Détection vidéo ──────────────────────────────────────────────────────

    def detect_video(
        self,
        video_path: str,
        return_frame_details: bool = False
    ) -> Dict:
        """
        Détecte deepfake dans une vidéo en analysant frame par frame.

        Args:
            video_path: Chemin vers le fichier vidéo
            return_frame_details: Inclure résultats par frame

        Returns:
            {
                overall_result: "REAL" | "FAKE",
                avg_confidence: float,
                stats: { num_frames, fake_ratio, ... },
                frame_results: [...] (si return_frame_details=True)
            }
        """
        start_time = time.time()
        self._ensure_models_loaded()

        try:
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Vidéo introuvable: {video_path}")

            # Charger vidéo
            cap, metadata, success = self.video_processor.load_video(video_path)
            if not success:
                return ResultFormatter.format_video_result(
                    frame_results=[], overall_prediction=-1, avg_confidence=0.0,
                    processing_time_ms=0.0
                )

            # Extraire frames avec skip
            frames, frame_indices, success = self.video_processor.extract_frames(cap, metadata)
            cap.release()

            if not success or not frames:
                logger.warning("Aucune frame extraite")
                return ResultFormatter.format_video_result(
                    frame_results=[], overall_prediction=-1, avg_confidence=0.0,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            logger.info(f"Vidéo: {len(frames)} frames à analyser...")

            # Détecter visages dans chaque frame
            faces_per_frame = self.video_processor.detect_faces_in_frames(
                frames, self.image_processor.detect_faces
            )

            # Extraire patchs
            patches = self.video_processor.extract_face_patches_from_frames(
                frames, faces_per_frame
            )

            if not patches:
                logger.warning("Aucun visage trouvé dans la vidéo")
                return ResultFormatter.format_video_result(
                    frame_results=[], overall_prediction=0, avg_confidence=0.0,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # Inférence patch par patch + collecte patches pour analyse temporelle
            frame_results_dict = {}
            total_patches = len(patches)
            face_patches_ordered = []  # pour analyse temporelle

            for i, patch_info in enumerate(patches):
                frame_idx = patch_info["frame_idx"]
                patch = patch_info["patch"]

                tensor = self.image_processor.preprocess(patch)
                if tensor is None:
                    continue

                pred, conf = self._infer_efficientnet(tensor)

                if frame_idx not in frame_results_dict:
                    frame_results_dict[frame_idx] = {"predictions": [], "confidences": []}

                frame_results_dict[frame_idx]["predictions"].append(pred)
                frame_results_dict[frame_idx]["confidences"].append(conf)
                face_patches_ordered.append(patch)

                if (i + 1) % 50 == 0:
                    logger.info(f"  Progression: {i+1}/{total_patches} patches")

            # ── Analyse temporelle robuste ───────────────────────────────────────
            # Métrique : incohérence des contours du visage entre frames normalisées
            # Insensible aux mouvements de caméra car on travaille sur patches 64x64
            temporal_fake_score = 0.0
            if len(face_patches_ordered) >= 3:
                try:
                    import cv2
                    target_size = (64, 64)
                    edge_inconsistencies = []
                    texture_stds = []

                    normalized = [cv2.resize(p, target_size) for p in face_patches_ordered]

                    for j in range(1, len(normalized)):
                        prev_g = cv2.cvtColor(normalized[j-1], cv2.COLOR_RGB2GRAY)
                        curr_g = cv2.cvtColor(normalized[j],   cv2.COLOR_RGB2GRAY)

                        # Contours Canny sur chaque patch normalisé
                        edges_prev = cv2.Canny(prev_g, 50, 150).astype(np.float32)
                        edges_curr = cv2.Canny(curr_g, 50, 150).astype(np.float32)

                        # Incohérence contours : fake → contours fluctuants (artefacts GAN)
                        edge_diff = np.abs(edges_prev - edges_curr)
                        edge_inconsistencies.append(float(np.mean(edge_diff)))

                        # Variation de texture locale entre frames
                        std_prev = float(np.std(prev_g.astype(np.float32)))
                        std_curr = float(np.std(curr_g.astype(np.float32)))
                        texture_stds.append(abs(std_prev - std_curr))

                    mean_edge_incons = float(np.mean(edge_inconsistencies))
                    mean_tex_var = float(np.mean(texture_stds))

                    # Seuils calibrés sur données observées :
                    # edge_incons : fake~70, real~44 → base=40, range=25
                    # → fake: (70-40)/25=1.0, real: (44-40)/25=0.16
                    edge_score = min(1.0, max(0.0, (mean_edge_incons - 40.0) / 25.0))
                    tex_score  = min(1.0, max(0.0, mean_tex_var / 10.0))

                    temporal_fake_score = 0.6 * edge_score + 0.4 * tex_score

                    logger.info(f"  Analyse temporelle : edge_incons={mean_edge_incons:.2f} "
                                f"tex_var={mean_tex_var:.2f} → "
                                f"edge={edge_score:.3f} tex={tex_score:.3f} "
                                f"temporal={temporal_fake_score:.3f}")

                except Exception as e:
                    logger.warning(f"Analyse temporelle ignorée: {e}")

            # Agrégation par frame — fake=0, real=1
            frame_results_list = []
            for frame_idx, results in sorted(frame_results_dict.items()):
                avg_conf = float(np.mean(results["confidences"]))
                fake_votes = sum(1 for p in results["predictions"] if p == 0)
                pred = 0 if fake_votes > len(results["predictions"]) / 2 else 1
                frame_results_list.append({
                    "frame_idx": int(frame_idx),
                    "prediction": int(pred),
                    "confidence": round(avg_conf, 4)
                })

            # Agrégation globale avec pondération temporelle
            agg_result = self.video_processor.aggregate_frame_results(frame_results_list)

            # Score combiné avec logique adaptative
            # Le modèle est entraîné sur images statiques → bon sur images
            # L'analyse temporelle est un signal d'appui, pas le signal principal
            model_fake_prob = 1.0 - agg_result["avg_confidence"]
            model_real_conf = agg_result["avg_confidence"]

            # Zones de confiance :
            # >70% confiant → faire confiance au modèle directement
            # 40-70% → zone grise, le temporel peut faire basculer
            # <40% → modèle confiant FAKE, lui faire confiance
            if model_real_conf >= 0.70:
                # Modèle confiant REAL mais peut se tromper sur les vidéos fake
                # Si signal temporel significatif (>0.30) : donner le poids au temporel
                if temporal_fake_score > 0.30:
                    combined_fake_score = 0.10 * model_fake_prob + 0.90 * temporal_fake_score
                else:
                    combined_fake_score = model_fake_prob
            elif model_fake_prob >= 0.70:
                # Modèle confiant FAKE → FAKE direct
                combined_fake_score = model_fake_prob
            else:
                # Zone grise 30-70% → combiner équitablement
                combined_fake_score = 0.55 * model_fake_prob + 0.45 * temporal_fake_score

            combined_confidence = 1.0 - combined_fake_score
            final_prediction = 0 if combined_fake_score >= 0.5 else 1  # 0=FAKE, 1=REAL

            # Mettre à jour stats avec infos temporelles
            stats = agg_result.get("stats", {})
            stats["temporal_fake_score"] = round(temporal_fake_score, 4)
            stats["combined_fake_score"] = round(combined_fake_score, 4)

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"✓ Vidéo analysée en {elapsed/1000:.1f}s → "
                        f"{'FAKE' if final_prediction == 0 else 'REAL'} "
                        f"(model_fake={model_fake_prob:.2f} temporal={temporal_fake_score:.2f} "
                        f"combined={combined_fake_score:.2f})")

            return ResultFormatter.format_video_result(
                frame_results=frame_results_list if return_frame_details else [],
                overall_prediction=final_prediction,
                avg_confidence=1.0 - combined_fake_score,
                processing_time_ms=elapsed,
                stats={**metadata, **stats}
            )

        except Exception as e:
            logger.error(f"Erreur detect_video: {e}", exc_info=True)
            return ResultFormatter.format_video_result(
                frame_results=[], overall_prediction=-1, avg_confidence=0.0,
                stats={"error": str(e)},
                processing_time_ms=(time.time() - start_time) * 1000
            )

    # ─── Détection batch ──────────────────────────────────────────────────────

    def detect_batch(
        self,
        image_inputs: List[Union[str, bytes, np.ndarray]],
        return_features: bool = False
    ) -> List[Dict]:
        """
        Détecte deepfakes dans une liste d'images.

        Args:
            image_inputs: Liste de chemins, bytes ou arrays
            return_features: Inclure features dans les réponses

        Returns:
            Liste de résultats dans le même ordre que l'entrée
        """
        results = []
        total = len(image_inputs)

        for i, img_input in enumerate(image_inputs):
            logger.info(f"Batch: {i+1}/{total}")
            result = self.detect_image(img_input, return_features=return_features)
            results.append(result)

        logger.info(f"✓ Batch traité: {len(results)} images")
        return results

    # ─── Utilitaires ──────────────────────────────────────────────────────────

    def _load_image_input(
        self,
        image_input: Union[str, bytes, "Image.Image", np.ndarray]
    ) -> Optional[np.ndarray]:
        """Charge image depuis différents formats vers numpy RGB"""
        try:
            if isinstance(image_input, str):
                image, success = self.image_processor.load_image(image_input)
                return image if success else None

            elif isinstance(image_input, bytes):
                image, success = self.image_processor.load_image_from_bytes(image_input)
                return image if success else None

            elif isinstance(image_input, Image.Image):
                return np.array(image_input.convert("RGB"))

            elif isinstance(image_input, np.ndarray):
                if image_input.ndim == 3 and image_input.shape[2] == 3:
                    return image_input.astype(np.uint8)
                elif image_input.ndim == 3 and image_input.shape[2] == 4:
                    # RGBA → RGB
                    return image_input[:, :, :3].astype(np.uint8)

            logger.error(f"Type d'image non supporté: {type(image_input)}")
            return None

        except Exception as e:
            logger.error(f"Erreur chargement image: {e}")
            return None

    def get_model_info(self) -> Dict:
        """Retourne informations sur les modèles chargés"""
        return {
            "efficientnet_loaded": self.model_efficientnet is not None,
            "resnet_loaded": self.model_resnet is not None,
            "models_ready": self._models_loaded,
            "device": self.device,
            "quantization_enabled": self.enable_quantization,
            "batch_size": self.batch_size,
            "pytorch_threads": torch.get_num_threads(),
        }

    def warm_up(self):
        """
        Chauffe les modèles avec une image factice.
        Recommandé avant utilisation intensive pour éviter le lag initial.
        """
        logger.info("Warm-up des modèles...")
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = self.image_processor.preprocess(dummy)
        if tensor is not None:
            self._infer_efficientnet(tensor)
        logger.info("✓ Warm-up terminé")


# ─── Fonctions de commodité ───────────────────────────────────────────────────

_detector_singleton: Optional[DeepfakeDetector] = None


def get_detector() -> DeepfakeDetector:
    """Retourne une instance singleton du détecteur (évite recharger les modèles)"""
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = DeepfakeDetector(device=CONFIG.device)
    return _detector_singleton


def quick_detect_image(image_path: str) -> Dict:
    """Raccourci: détection rapide d'une image"""
    return get_detector().detect_image(image_path)


def quick_detect_video(video_path: str) -> Dict:
    """Raccourci: détection rapide d'une vidéo"""
    return get_detector().detect_video(video_path)


if __name__ == "__main__":
    print("✓ Module detector chargé")
    print(f"Device: {CONFIG.device}")
    print(f"Threads PyTorch: {torch.get_num_threads()}")