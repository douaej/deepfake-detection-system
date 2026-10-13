#!/usr/bin/env python3
"""
Benchmark comparatif des performances des modèles
Mesure temps d'inférence, mémoire, précision sur un dataset de test
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import psutil
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from config import CONFIG
from src.detector import DeepfakeDetector
from src.image_processor import ImageProcessor
from src.utils import ModelLoader, MetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Classe pour benchmarker les modèles individuellement"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.process = psutil.Process()
        
    def measure_memory(self) -> float:
        """Retourne mémoire utilisée en MB"""
        return self.process.memory_info().rss / (1024 ** 2)
    
    def benchmark_efficientnet(self, num_iterations: int = 100) -> Dict:
        """Benchmark EfficientNet-B0"""
        logger.info("=" * 60)
        logger.info("BENCHMARK EFFICIENTNET-B0")
        logger.info("=" * 60)
        
        # Charger modèle
        mem_before = self.measure_memory()
        model = ModelLoader.load_efficientnet_b0(device=self.device)
        model.eval()
        mem_after = self.measure_memory()
        mem_model = mem_after - mem_before
        
        # Quantization
        if CONFIG.optimization.ENABLE_QUANTIZATION:
            from torch.quantization import quantize_dynamic
            model_quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            mem_quantized = self.measure_memory()
            mem_model_quantized = mem_quantized - mem_after
        else:
            model_quantized = model
            mem_model_quantized = mem_model
        
        # Input factice
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model_quantized(dummy_input)
        
        # Benchmark inférence
        times = []
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc="EfficientNet inference"):
                start = time.perf_counter()
                _ = model_quantized(dummy_input)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
        
        return {
            'model_name': 'EfficientNet-B0',
            'memory_mb': mem_model,
            'memory_quantized_mb': mem_model_quantized,
            'mean_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'fps': 1000.0 / np.mean(times),
            'num_params': sum(p.numel() for p in model.parameters()),
        }
    
    def benchmark_resnet18(self, num_iterations: int = 100) -> Dict:
        """Benchmark ResNet18"""
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RESNET18")
        logger.info("=" * 60)
        
        mem_before = self.measure_memory()
        model = ModelLoader.load_resnet18(device=self.device)
        model.eval()
        mem_after = self.measure_memory()
        mem_model = mem_after - mem_before
        
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc="ResNet18 inference"):
                start = time.perf_counter()
                _ = model(dummy_input)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
        
        return {
            'model_name': 'ResNet18',
            'memory_mb': mem_model,
            'memory_quantized_mb': mem_model,  # Pas quantizé dans le projet
            'mean_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'fps': 1000.0 / np.mean(times),
            'num_params': sum(p.numel() for p in model.parameters()),
        }
    
    def benchmark_mediapipe(self, num_iterations: int = 100) -> Dict:
        """Benchmark MediaPipe Face Detection"""
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK MEDIAPIPE FACE DETECTION")
        logger.info("=" * 60)
        
        mem_before = self.measure_memory()
        processor = ImageProcessor(device=self.device)
        mem_after = self.measure_memory()
        mem_model = mem_after - mem_before
        
        # Image factice
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warm-up
        for _ in range(10):
            _ = processor.detect_faces(dummy_image)
        
        # Benchmark
        times = []
        for _ in tqdm(range(num_iterations), desc="MediaPipe inference"):
            start = time.perf_counter()
            _ = processor.detect_faces(dummy_image)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        return {
            'model_name': 'MediaPipe FaceDetection',
            'memory_mb': mem_model,
            'memory_quantized_mb': mem_model,
            'mean_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'fps': 1000.0 / np.mean(times),
            'num_params': 'N/A (TFLite)',
        }
    
    def benchmark_full_pipeline(self, test_images: List[str] = None) -> Dict:
        """Benchmark pipeline complet (image réelle)"""
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK PIPELINE COMPLET")
        logger.info("=" * 60)
        
        if not test_images:
            logger.warning("Aucune image de test fournie, benchmark pipeline ignoré")
            return {}
        
        mem_before = self.measure_memory()
        detector = DeepfakeDetector(device=self.device)
        detector.warm_up()
        mem_after = self.measure_memory()
        mem_total = mem_after - mem_before
        
        times = []
        for img_path in tqdm(test_images, desc="Pipeline complet"):
            start = time.perf_counter()
            _ = detector.detect_image(img_path, return_heatmap=False, return_features=False)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        return {
            'model_name': 'Pipeline complet (detect_image)',
            'memory_total_mb': mem_total,
            'mean_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'fps': 1000.0 / np.mean(times),
            'num_images': len(test_images),
        }


def print_comparison_table(results: List[Dict]):
    """Affiche tableau comparatif formaté"""
    print("\n" + "=" * 100)
    print("TABLEAU COMPARATIF DES PERFORMANCES")
    print("=" * 100)
    
    # Header
    print(f"{'Modèle':<30} {'Mémoire (MB)':<15} {'Inf. moy. (ms)':<18} {'FPS':<10} {'Params':<15}")
    print("-" * 100)
    
    # Lignes
    for r in results:
        name = r.get('model_name', 'Unknown')
        mem = r.get('memory_mb', 0)
        mem_q = r.get('memory_quantized_mb', mem)
        mean_time = r.get('mean_inference_ms', 0)
        fps = r.get('fps', 0)
        params = r.get('num_params', 'N/A')
        
        # Format params
        if isinstance(params, int):
            if params > 1e6:
                params_str = f"{params / 1e6:.1f}M"
            else:
                params_str = f"{params / 1e3:.1f}K"
        else:
            params_str = str(params)
        
        # Ligne avec mémoire avant quantization
        mem_str = f"{mem:.1f}"
        if mem_q != mem and mem_q > 0:
            mem_str += f" → {mem_q:.1f}"
        
        print(f"{name:<30} {mem_str:<15} {mean_time:<18.2f} {fps:<10.1f} {params_str:<15}")
    
    print("=" * 100)
    
    # Résumé
    print("\n📊 RÉSUMÉ")
    print("-" * 50)
    
    # Plus rapide
    fastest = min(results, key=lambda x: x.get('mean_inference_ms', float('inf')))
    print(f"⚡ Plus rapide:      {fastest['model_name']} ({fastest['mean_inference_ms']:.2f}ms)")
    
    # Plus léger
    lightest = min(results, key=lambda x: x.get('memory_mb', float('inf')))
    print(f"💾 Plus léger:       {lightest['model_name']} ({lightest['memory_mb']:.1f} MB)")
    
    # Meilleur FPS
    best_fps = max(results, key=lambda x: x.get('fps', 0))
    print(f"🎯 Meilleur FPS:     {best_fps['model_name']} ({best_fps['fps']:.1f} FPS)")
    
    print("-" * 50)


def print_detailed_stats(results: List[Dict]):
    """Affiche statistiques détaillées"""
    print("\n" + "=" * 100)
    print("STATISTIQUES DÉTAILLÉES")
    print("=" * 100)
    
    for r in results:
        print(f"\n{r['model_name']}")
        print("-" * 50)
        print(f"  Temps inférence:")
        print(f"    Moyenne:  {r.get('mean_inference_ms', 0):.3f} ms")
        print(f"    Écart-type: {r.get('std_inference_ms', 0):.3f} ms")
        print(f"    Min:      {r.get('min_inference_ms', 0):.3f} ms")
        print(f"    Max:      {r.get('max_inference_ms', 0):.3f} ms")
        print(f"  Mémoire:    {r.get('memory_mb', 0):.2f} MB")
        if r.get('memory_quantized_mb') and r.get('memory_quantized_mb') != r.get('memory_mb'):
            print(f"  Mém. quantizée: {r['memory_quantized_mb']:.2f} MB")
        print(f"  FPS:        {r.get('fps', 0):.2f}")
        if r.get('num_params') != 'N/A':
            print(f"  Paramètres: {r.get('num_params', 0):,}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark des modèles")
    parser.add_argument('--iterations', type=int, default=100, help="Nombre d'itérations par modèle")
    parser.add_argument('--device', default=CONFIG.device, help="Device (cpu, cuda, xpu)")
    parser.add_argument('--test-images', nargs='*', help="Images de test pour pipeline complet")
    parser.add_argument('--detailed', action='store_true', help="Afficher stats détaillées")
    args = parser.parse_args()
    
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║       BENCHMARK COMPARATIF DES MODÈLES               ║")
    print(f"║  Device: {args.device:<10}  Iterations: {args.iterations:<3}             ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    
    benchmark = ModelBenchmark(device=args.device)
    results = []
    
    try:
        # Benchmark EfficientNet-B0
        results.append(benchmark.benchmark_efficientnet(args.iterations))
    except Exception as e:
        logger.error(f"Erreur EfficientNet: {e}")
    
    try:
        # Benchmark ResNet18
        results.append(benchmark.benchmark_resnet18(args.iterations))
    except Exception as e:
        logger.error(f"Erreur ResNet18: {e}")
    
    try:
        # Benchmark MediaPipe
        results.append(benchmark.benchmark_mediapipe(args.iterations))
    except Exception as e:
        logger.error(f"Erreur MediaPipe: {e}")
    
    # Pipeline complet (si images fournies)
    if args.test_images:
        try:
            results.append(benchmark.benchmark_full_pipeline(args.test_images))
        except Exception as e:
            logger.error(f"Erreur pipeline: {e}")
    
    # Affichage
    print_comparison_table(results)
    
    if args.detailed:
        print_detailed_stats(results)
    
    print("\n✓ Benchmark terminé\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
