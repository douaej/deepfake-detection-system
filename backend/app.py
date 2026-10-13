"""
API Backend Flask pour détection de deepfakes
Endpoints pour détection images/vidéos, health check, etc.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import CONFIG
from src.detector import DeepfakeDetector
from src.utils import ResultFormatter

# Setup Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = CONFIG.api.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = CONFIG.api.UPLOAD_FOLDER

# Create upload folder
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=CONFIG.logging.LOG_FORMAT,
    handlers=[
        logging.FileHandler(CONFIG.logging.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Détecteur global — lazy init au premier appel (évite de bloquer le démarrage Flask)
_DETECTOR = None
_DETECTOR_LOCK = __import__('threading').Lock()


def get_detector() -> "DeepfakeDetector":
    """Retourne l'instance singleton du détecteur, en l'initialisant si nécessaire."""
    global _DETECTOR
    if _DETECTOR is None:
        with _DETECTOR_LOCK:
            if _DETECTOR is None:
                logger.info("Initializing DeepfakeDetector...")
                _DETECTOR = DeepfakeDetector(
                    device=CONFIG.device,
                    batch_size=CONFIG.batch_size,
                    load_models=True
                )
                logger.info("✓ Detector ready")
    return _DETECTOR


# ==================== UTILITY FUNCTIONS ====================

def allowed_file(filename: str) -> bool:
    """Vérifie si fichier a extension autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in CONFIG.api.ALLOWED_EXTENSIONS


def get_request_id() -> str:
    """Récupère request ID depuis headers ou génère"""
    return request.headers.get('X-Request-ID', f"req_{datetime.now().timestamp()}")


# ==================== HEALTH & INFO ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': str(datetime.now()),
        'detector_models': get_detector().get_model_info(),
        'device': CONFIG.device,
        'version': '1.0.0'
    }), 200


@app.route('/api/info', methods=['GET'])
def get_info():
    """Retourne info système et configuration"""
    return jsonify({
        'system': {
            'device': CONFIG.device,
            'batch_size': CONFIG.batch_size,
            'max_content_length_mb': CONFIG.api.MAX_CONTENT_LENGTH / (1024 ** 2),
        },
        'models': get_detector().get_model_info(),
        'inference': {
            'confidence_threshold': CONFIG.inference.CONFIDENCE_THRESHOLD,
            'fake_threshold': CONFIG.inference.FAKE_THRESHOLD,
            'input_size': CONFIG.inference.INPUT_SIZE,
        },
        'features': {
            'texture_analysis': CONFIG.analysis.ENABLE_TEXTURE_ANALYSIS,
            'frequency_analysis': CONFIG.analysis.ENABLE_FREQUENCY_ANALYSIS,
        }
    }), 200


# ==================== IMAGE DETECTION ====================

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """
    Détecte si image est deepfake
    
    POST /api/detect/image
    
    Form data:
        - file: Image file (jpg, png, etc.)
        - return_heatmap: bool (optionnel)
        - return_features: bool (optionnel)
    
    Returns: JSON avec résultats détection
    """
    request_id = get_request_id()
    
    try:
        logger.info(f"[{request_id}] Image detection request")
        
        # Vérifier fichier
        if 'file' not in request.files:
            logger.warning(f"[{request_id}] No file provided")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning(f"[{request_id}] Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"[{request_id}] File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Options
        return_heatmap = request.form.get('return_heatmap', 'true').lower() == 'true'
        return_features = request.form.get('return_features', 'false').lower() == 'true'
        
        # Lire fichier
        file_data = file.read()
        
        logger.info(f"[{request_id}] Processing image: {file.filename} ({len(file_data)} bytes)")
        
        # Détection
        result = get_detector().detect_image(
            file_data,
            return_features=return_features,
            return_heatmap=return_heatmap
        )
        
        result['request_id'] = request_id
        
        logger.info(f"[{request_id}] ✓ Detection complete: {result['result']}")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        return jsonify({
            'error': str(e),
            'request_id': request_id
        }), 500


# ==================== VIDEO DETECTION ====================

@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    """
    Détecte deepfake dans vidéo
    
    POST /api/detect/video
    
    Form data:
        - file: Video file (mp4, avi, etc.)
    
    Returns: JSON avec résultats détection
    """
    request_id = get_request_id()
    
    try:
        logger.info(f"[{request_id}] Video detection request")
        
        # Vérifier fichier
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Sauvegarder fichier temporaire
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"[{request_id}] Processing video: {filename}")
        
        # Détection
        result = get_detector().detect_video(filepath)
        result['request_id'] = request_id
        
        # Nettoyer fichier temporaire
        try:
            os.remove(filepath)
        except:
            pass
        
        logger.info(f"[{request_id}] ✓ Video analysis complete")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        return jsonify({
            'error': str(e),
            'request_id': request_id
        }), 500


# ==================== BATCH PROCESSING ====================

@app.route('/api/detect/batch', methods=['POST'])
def detect_batch():
    """
    Traite batch d'images
    
    POST /api/detect/batch
    
    JSON body:
    {
        "image_paths": [
            "/path/to/image1.jpg",
            "/path/to/image2.jpg"
        ]
    }
    
    Returns: JSON avec liste de résultats
    """
    request_id = get_request_id()
    
    try:
        logger.info(f"[{request_id}] Batch detection request")
        
        data = request.get_json()
        
        if not data or 'image_paths' not in data:
            return jsonify({'error': 'Missing image_paths'}), 400
        
        image_paths = data['image_paths']
        
        if not isinstance(image_paths, list):
            return jsonify({'error': 'image_paths must be a list'}), 400
        
        logger.info(f"[{request_id}] Processing {len(image_paths)} images")
        
        results = get_detector().detect_batch(image_paths)
        
        return jsonify({
            'request_id': request_id,
            'num_images': len(results),
            'results': results
        }), 200
    
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        return jsonify({
            'error': str(e),
            'request_id': request_id
        }), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Gère fichier trop gros"""
    return jsonify({
        'error': 'File too large',
        'max_size_mb': CONFIG.api.MAX_CONTENT_LENGTH / (1024 ** 2)
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Gère endpoint non trouvé"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Gère erreur serveur"""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepFake Detection API')
    parser.add_argument('--host', default=CONFIG.api.HOST, help='Host')
    parser.add_argument('--port', type=int, default=CONFIG.api.PORT, help='Port')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--device', default=CONFIG.device, help='Device (cpu, cuda, xpu)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Debug: {args.debug}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False
    )