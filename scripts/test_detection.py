"""
Script de test et benchmark du système de détection
Teste accuracy, performance, mémoire, etc.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List

# Fix imports: add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np
import psutil
import torch

from config import CONFIG
from src.detector import DeepfakeDetector
from src.utils import MetricsCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark de performance du détecteur"""
    
    def __init__(self, detector: DeepfakeDetector):
        self.detector = detector
        self.process = psutil.Process()
    
    def test_image_inference(self, image_path: str, num_runs: int = 5) -> Dict:
        """Test inférence image"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"IMAGE INFERENCE BENCHMARK ({num_runs} runs)")
            logger.info(f"{'='*60}")
            
            times = []
            confidences = []
            
            for i in range(num_runs):
                # Mesurer temps
                start_time = time.time()
                result = self.detector.detect_image(image_path)
                elapsed = (time.time() - start_time) * 1000  # ms
                
                times.append(elapsed)
                confidences.append(result.get('confidence', 0.0))
                
                logger.info(f"Run {i+1}/{num_runs}: {elapsed:.1f}ms - "
                           f"{result['result']} ({result['confidence']:.2%})")
            
            # Stats
            stats = {
                'mean_time_ms': float(np.mean(times)),
                'std_time_ms': float(np.std(times)),
                'min_time_ms': float(np.min(times)),
                'max_time_ms': float(np.max(times)),
                'fps': float(1000.0 / np.mean(times)),
                'mean_confidence': float(np.mean(confidences))
            }
            
            logger.info(f"\nResults:")
            logger.info(f"  Mean time: {stats['mean_time_ms']:.1f}ms")
            logger.info(f"  Std dev: {stats['std_time_ms']:.1f}ms")
            logger.info(f"  FPS: {stats['fps']:.1f}")
            logger.info(f"  Range: {stats['min_time_ms']:.1f}ms - {stats['max_time_ms']:.1f}ms")
            
            return stats
        
        except Exception as e:
            logger.error(f"Error in image benchmark: {e}")
            return {}
    
    def test_memory_usage(self) -> Dict:
        """Teste utilisation mémoire"""
        logger.info(f"\n{'='*60}")
        logger.info("MEMORY USAGE TEST")
        logger.info(f"{'='*60}")
        
        try:
            # Baseline
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            mem_before = self.process.memory_info().rss / (1024 ** 2)
            
            logger.info(f"Memory before: {mem_before:.1f} MB")
            
            # Créer teste image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Inférence
            result = self.detector.detect_image(test_image)
            
            # Mémoire après
            mem_after = self.process.memory_info().rss / (1024 ** 2)
            mem_peak = max(mem_before, mem_after)
            
            stats = {
                'memory_before_mb': float(mem_before),
                'memory_after_mb': float(mem_after),
                'memory_used_mb': float(mem_after - mem_before),
                'peak_memory_mb': float(mem_peak)
            }
            
            logger.info(f"Memory after: {mem_after:.1f} MB")
            logger.info(f"Memory used: {stats['memory_used_mb']:.1f} MB")
            logger.info(f"Peak memory: {mem_peak:.1f} MB")
            
            return stats
        
        except Exception as e:
            logger.error(f"Error in memory test: {e}")
            return {}
    
    def test_batch_processing(self, num_images: int = 10) -> Dict:
        """Teste traitement batch"""
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH PROCESSING TEST ({num_images} images)")
        logger.info(f"{'='*60}")
        
        try:
            # Créer images test
            test_images = [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                for _ in range(num_images)
            ]
            
            start_time = time.time()
            results = self.detector.detect_batch(test_images)
            elapsed = (time.time() - start_time) * 1000  # ms
            
            avg_time = elapsed / num_images
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            
            stats = {
                'total_time_ms': float(elapsed),
                'avg_time_per_image_ms': float(avg_time),
                'throughput_fps': float(fps),
                'images_processed': len(results)
            }
            
            logger.info(f"Total time: {elapsed:.1f}ms")
            logger.info(f"Average per image: {avg_time:.1f}ms")
            logger.info(f"Throughput: {fps:.1f} images/sec")
            
            return stats
        
        except Exception as e:
            logger.error(f"Error in batch test: {e}")
            return {}
    
    def test_model_loading(self) -> Dict:
        """Teste chargement des modèles"""
        logger.info(f"\n{'='*60}")
        logger.info("MODEL LOADING TEST")
        logger.info(f"{'='*60}")
        
        try:
            start_time = time.time()
            detector = DeepfakeDetector(device=CONFIG.device, load_models=True)
            elapsed = (time.time() - start_time) * 1000
            
            stats = {
                'loading_time_ms': float(elapsed),
                'device': CONFIG.device,
                'models_loaded': detector.get_model_info()
            }
            
            logger.info(f"Loading time: {elapsed:.1f}ms")
            logger.info(f"Models: {stats['models_loaded']}")
            
            return stats
        
        except Exception as e:
            logger.error(f"Error in model loading test: {e}")
            return {}


def create_sample_test_image() -> str:
    """Crée une image test temporaire"""
    try:
        from PIL import Image
        
        # Créer image simple
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Sauvegarder
        test_path = CONFIG.data_dir / "test_image.jpg"
        img.save(test_path)
        
        logger.info(f"✓ Test image created: {test_path}")
        return str(test_path)
    
    except Exception as e:
        logger.error(f"Error creating test image: {e}")
        return None


def run_full_test_suite():
    """Exécute suite complète de tests"""
    logger.info("\n")
    logger.info("╔════════════════════════════════════════════════════╗")
    logger.info("║         DEEPFAKE DETECTION - TEST SUITE            ║")
    logger.info("╚════════════════════════════════════════════════════╝")
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Device: {CONFIG.device}")
    logger.info(f"  Batch size: {CONFIG.batch_size}")
    logger.info(f"  Input size: {CONFIG.inference.INPUT_SIZE}")
    
    try:
        # Initialiser détecteur
        logger.info("\nInitializing detector...")
        detector = DeepfakeDetector(device=CONFIG.device)
        logger.info("✓ Detector ready")
        
        # Créer benchmark
        benchmark = PerformanceBenchmark(detector)
        
        # Tests
        results = {
            'model_loading': benchmark.test_model_loading(),
            'memory_usage': benchmark.test_memory_usage(),
            'batch_processing': benchmark.test_batch_processing(num_images=5),
        }
        
        # Test image si possible
        test_image_path = create_sample_test_image()
        if test_image_path:
            results['image_inference'] = benchmark.test_image_inference(test_image_path, num_runs=3)
        
        # Résumé
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        for test_name, test_results in results.items():
            if test_results:
                logger.info(f"\n{test_name.upper()}:")
                for key, value in test_results.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.2f}")
                    else:
                        logger.info(f"  {key}: {value}")
        
        logger.info(f"\n{'='*60}")
        logger.info("✓ ALL TESTS COMPLETE")
        logger.info(f"{'='*60}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in test suite: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Tests')
    parser.add_argument('--mode', choices=['full', 'quick', 'benchmark'], 
                       default='full', help='Test mode')
    parser.add_argument('--device', default=CONFIG.device, help='Device')
    parser.add_argument('--image', help='Image path for testing')
    
    args = parser.parse_args()
    
    # Override device si spécifié
    if args.device:
        CONFIG.device = args.device
    
    if args.mode == 'full':
        run_full_test_suite()
    elif args.mode == 'quick':
        detector = DeepfakeDetector(device=args.device)
        benchmark = PerformanceBenchmark(detector)
        
        test_image = args.image or create_sample_test_image()
        if test_image:
            benchmark.test_image_inference(test_image, num_runs=1)
    
    elif args.mode == 'benchmark':
        detector = DeepfakeDetector(device=args.device)
        benchmark = PerformanceBenchmark(detector)
        
        benchmark.test_model_loading()
        benchmark.test_memory_usage()
        benchmark.test_batch_processing()


if __name__ == '__main__':
    main()
