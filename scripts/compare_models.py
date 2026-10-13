#!/usr/bin/env python3
"""
Comparaison de modèles - EfficientNet-B0 vs ResNet18 vs SVM
Script d'évaluation comparative pour rapport académique

Ne touche PAS au modèle fine-tuné principal (models/pretrained/deepfake_efficientnet.pth)
Tous les modèles comparatifs sont sauvegardés dans models/comparison/
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import timm

import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from tqdm import tqdm
import pickle

# Setup paths
BASE_DIR = Path(__file__).parent.parent
COMPARISON_DIR = BASE_DIR / "models" / "comparison"
OUTPUT_DIR = BASE_DIR / "outputs" / "comparison"
DATA_PATH = BASE_DIR / "data" / "my_real_vs_ai_dataset"

COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'comparison.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEVICE = "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
MAX_SAMPLES_PER_CLASS = 10000  # 10k fake + 10k real = 20k total
RANDOM_SEED = 42

# Fix seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ==================== DATA PREPARATION ====================

def prepare_dataset():
    """
    Prépare le dataset avec le même subset et split que l'entraînement original
    Returns: train_loader, val_loader, dataset
    """
    logger.info("=" * 80)
    logger.info("PRÉPARATION DU DATASET")
    logger.info("=" * 80)
    
    # Transforms (mêmes que train_model.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load full dataset
    dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
    logger.info(f"Classes: {dataset.classes}")  # ['fake', 'real']
    logger.info(f"Total images disponibles: {len(dataset)}")
    
    # Subset 20k (10k per class) - MÊME LOGIQUE que train_model.py
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    
    selected_indices = []
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        sampled = random.sample(indices, min(MAX_SAMPLES_PER_CLASS, len(indices)))
        selected_indices.extend(sampled)
        logger.info(f"  Classe {label} ({dataset.classes[label]}): {len(sampled)} images")
    
    subset = Subset(dataset, selected_indices)
    logger.info(f"Subset total: {len(subset)} images")
    
    # Train/Val split 80/20 - MÊME SPLIT que train_model.py
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_dataset, val_dataset = random_split(
        subset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    logger.info(f"Train: {train_size} | Val: {val_size}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, dataset


# ==================== MODEL 1: EFFICIENTNET-B0 (BASELINE) ====================

def evaluate_efficientnet_baseline(val_loader, dataset):
    """
    Évalue le modèle EfficientNet-B0 fine-tuné EXISTANT (baseline)
    COPIE le modèle dans comparison/ sans le modifier
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODÈLE 1: EfficientNet-B0 (Baseline - Modèle fine-tuné existant)")
    logger.info("=" * 80)
    
    # Charger le modèle EXISTANT
    model_path = BASE_DIR / "models" / "pretrained" / "deepfake_efficientnet.pth"
    if not model_path.exists():
        logger.error(f"Modèle baseline introuvable: {model_path}")
        logger.error("Lancez d'abord: python scripts/train_model.py")
        return None
    
    logger.info(f"Chargement du modèle baseline: {model_path}")
    
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    
    # Copier dans comparison/ pour référence
    comparison_path = COMPARISON_DIR / "efficientnet_baseline.pth"
    import shutil
    shutil.copy(model_path, comparison_path)
    logger.info(f"✓ Copie sauvegardée: {comparison_path}")
    
    # Évaluation
    logger.info("Évaluation sur validation set...")
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="EfficientNet eval"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Proba classe 1 (real)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calcul métriques
    metrics = calculate_metrics(all_labels, all_preds, all_probs, dataset.classes)
    
    logger.info("✓ EfficientNet-B0 évalué")
    log_metrics("EfficientNet-B0", metrics)
    
    return {
        'name': 'EfficientNet-B0',
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'metrics': metrics
    }


# ==================== MODEL 2: RESNET18 (FINE-TUNED) ====================

def train_and_evaluate_resnet18(train_loader, val_loader, dataset):
    """
    Fine-tune ResNet18 avec même stratégie que EfficientNet
    Sauvegarde dans models/comparison/resnet18_finetuned.pth
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODÈLE 2: ResNet18 (Fine-tuning from scratch)")
    logger.info("=" * 80)
    
    # Charger ResNet18 pré-entraîné ImageNet
    from torchvision import models
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Remplacer classifier: 512 features → 2 classes
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)
    
    logger.info("Architecture ResNet18 chargée (ImageNet → 2 classes)")
    
    # Freeze backbone sauf les 2 derniers layers
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Unfreeze layer4 (derniers blocs)
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Paramètres entraînables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    
    # Training loop
    logger.info(f"\nEntraînement: {EPOCHS} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | "
                   f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = COMPARISON_DIR / "resnet18_finetuned.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"  → Meilleur modèle sauvegardé: {save_path}")
    
    logger.info(f"✓ Entraînement terminé. Meilleure val accuracy: {best_val_acc:.2f}%")
    
    # Reload best model pour évaluation finale
    model.load_state_dict(torch.load(COMPARISON_DIR / "resnet18_finetuned.pth", map_location=DEVICE))
    model.eval()
    
    # Évaluation finale
    logger.info("Évaluation finale sur validation set...")
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="ResNet18 eval"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs, dataset.classes)
    log_metrics("ResNet18", metrics)
    
    return {
        'name': 'ResNet18',
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'metrics': metrics
    }


# ==================== MODEL 3: SVM (MACHINE LEARNING CLASSIQUE) ====================

def train_and_evaluate_svm(train_loader, val_loader, dataset):
    """
    SVM sur features extraites de ResNet18 gelé (ImageNet pré-entraîné)
    Approche ML traditionnelle: ResNet18 = extracteur de features, SVM = classifieur
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODÈLE 3: SVM sur features ResNet18 (ML classique)")
    logger.info("=" * 80)
    
    # ResNet18 comme extracteur de features (gelé, ImageNet)
    from torchvision import models
    feature_extractor = models.resnet18(weights='IMAGENET1K_V1')
    # Retirer le classifier final → on garde jusqu'à avgpool
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.to(DEVICE)
    feature_extractor.eval()
    
    logger.info("Extraction des features d'entraînement...")
    train_features, train_labels = extract_features(feature_extractor, train_loader)
    logger.info(f"Features train: {train_features.shape}")
    
    logger.info("Extraction des features de validation...")
    val_features, val_labels = extract_features(feature_extractor, val_loader)
    logger.info(f"Features val: {val_features.shape}")
    
    # Entraîner SVM
    logger.info("Entraînement du SVM (kernel RBF)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RANDOM_SEED)
    
    start_time = time.time()
    svm.fit(train_features, train_labels)
    elapsed = time.time() - start_time
    logger.info(f"✓ SVM entraîné en {elapsed:.1f}s")
    
    # Sauvegarder SVM
    svm_path = COMPARISON_DIR / "svm_model.pkl"
    with open(svm_path, 'wb') as f:
        pickle.dump(svm, f)
    logger.info(f"✓ SVM sauvegardé: {svm_path}")
    
    # Prédictions
    logger.info("Prédiction sur validation set...")
    all_preds = svm.predict(val_features)
    all_probs = svm.predict_proba(val_features)[:, 1]  # Proba classe 1 (real)
    
    metrics = calculate_metrics(val_labels, all_preds, all_probs, dataset.classes)
    log_metrics("SVM", metrics)
    
    return {
        'name': 'SVM',
        'predictions': all_preds,
        'labels': val_labels,
        'probabilities': all_probs,
        'metrics': metrics
    }


def extract_features(model, dataloader):
    """Extrait features (512,) depuis ResNet18 avgpool"""
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Feature extraction"):
            images = images.to(DEVICE)
            features = model(images)
            features = features.squeeze()  # (batch, 512, 1, 1) → (batch, 512)
            
            features_list.append(features.cpu().numpy())
            labels_list.extend(labels.numpy())
    
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    
    return features, labels


# ==================== METRICS CALCULATION ====================

def calculate_metrics(y_true, y_pred, y_prob, class_names):
    """Calcule toutes les métriques"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
    }
    
    # AUC-ROC (si classes équilibrées)
    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob))
    except:
        metrics['auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report détaillé par classe
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics['per_class'] = {
        'fake': report['fake'],
        'real': report['real']
    }
    
    return metrics


def log_metrics(model_name, metrics):
    """Affiche métriques formatées"""
    logger.info(f"\n{'─' * 60}")
    logger.info(f"RÉSULTATS: {model_name}")
    logger.info(f"{'─' * 60}")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc']:.4f}")
    logger.info(f"{'─' * 60}\n")


# ==================== VISUALIZATION ====================

def create_comparison_plots(results):
    """Génère tous les graphiques de comparaison"""
    logger.info("\n" + "=" * 80)
    logger.info("GÉNÉRATION DES GRAPHIQUES")
    logger.info("=" * 80)
    
    # 1. Barplot comparatif des métriques
    plot_metrics_comparison(results)
    
    # 2. Matrices de confusion (3 subplots)
    plot_confusion_matrices(results)
    
    # 3. Courbes ROC
    plot_roc_curves(results)
    
    # 4. Tableau récapitulatif
    save_metrics_table(results)
    
    logger.info("✓ Tous les graphiques générés")


def plot_metrics_comparison(results):
    """Barplot: 3 modèles × 4 métriques"""
    logger.info("  → Barplot comparatif des métriques...")
    
    models = [r['name'] for r in results]
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    
    data = {metric: [r['metrics'][metric] for r in results] for metric in metrics_names}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    
    for i, (metric, values) in enumerate(data.items()):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=colors[i], alpha=0.8)
        
        # Annotations
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Modèles', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des métriques de classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    ✓ Sauvegardé: {OUTPUT_DIR / 'metrics_comparison.png'}")


def plot_confusion_matrices(results):
    """3 matrices de confusion côte à côte"""
    logger.info("  → Matrices de confusion...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, result in enumerate(results):
        cm = np.array(result['metrics']['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   cbar=True, square=True, linewidths=1, linecolor='gray')
        
        axes[i].set_title(f"{result['name']}\nAccuracy: {result['metrics']['accuracy']:.3f}",
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Prédiction', fontsize=10)
        axes[i].set_ylabel('Vérité terrain', fontsize=10)
        axes[i].set_xticklabels(['Fake', 'Real'])
        axes[i].set_yticklabels(['Fake', 'Real'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    ✓ Sauvegardé: {OUTPUT_DIR / 'confusion_matrices.png'}")


def plot_roc_curves(results):
    """Courbes ROC des 3 modèles sur même graphe"""
    logger.info("  → Courbes ROC...")
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    for i, result in enumerate(results):
        y_true = result['labels']
        y_prob = result['probabilities']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = result['metrics']['auc']
        
        plt.plot(fpr, tpr, label=f"{result['name']} (AUC = {auc:.3f})",
                linewidth=2, color=colors[i])
    
    # Ligne diagonale (random)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Courbes ROC - Comparaison des modèles', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    ✓ Sauvegardé: {OUTPUT_DIR / 'roc_curves.png'}")


def save_metrics_table(results):
    """Sauvegarde tableaux CSV et JSON"""
    logger.info("  → Tableaux de métriques...")
    
    # JSON complet
    metrics_dict = {r['name']: r['metrics'] for r in results}
    with open(OUTPUT_DIR / 'metrics_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"    ✓ JSON: {OUTPUT_DIR / 'metrics_detailed.json'}")
    
    # CSV simple (pour copier-coller dans rapport)
    import csv
    with open(OUTPUT_DIR / 'metrics_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
        
        for r in results:
            m = r['metrics']
            writer.writerow([
                r['name'],
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['auc']:.4f}"
            ])
    logger.info(f"    ✓ CSV: {OUTPUT_DIR / 'metrics_summary.csv'}")
    
    # Classification report texte formaté
    with open(OUTPUT_DIR / 'classification_reports.txt', 'w', encoding='utf-8') as f:
        for r in results:
            f.write("=" * 80 + "\n")
            f.write(f"MODÈLE: {r['name']}\n")
            f.write("=" * 80 + "\n\n")
            
            # Résumé global
            m = r['metrics']
            f.write(f"Accuracy:  {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {m['precision']:.4f}\n")
            f.write(f"Recall:    {m['recall']:.4f}\n")
            f.write(f"F1-Score:  {m['f1']:.4f}\n")
            f.write(f"AUC-ROC:   {m['auc']:.4f}\n\n")
            
            # Par classe
            f.write("Métriques par classe:\n")
            f.write("-" * 60 + "\n")
            for cls in ['fake', 'real']:
                pc = m['per_class'][cls]
                f.write(f"  {cls.upper()}:\n")
                f.write(f"    Precision: {pc['precision']:.4f}\n")
                f.write(f"    Recall:    {pc['recall']:.4f}\n")
                f.write(f"    F1-Score:  {pc['f1-score']:.4f}\n")
                f.write(f"    Support:   {int(pc['support'])}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    logger.info(f"    ✓ Rapport texte: {OUTPUT_DIR / 'classification_reports.txt'}")


# ==================== MAIN ====================

def main():
    global EPOCHS, BATCH_SIZE
    
    parser = argparse.ArgumentParser(
        description="Comparaison de modèles: EfficientNet-B0 vs ResNet18 vs SVM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple:
  python scripts/compare_models.py --epochs 5 --batch-size 8

Résultats sauvegardés dans: outputs/comparison/
  - metrics_comparison.png      (barplot)
  - confusion_matrices.png       (3 heatmaps)
  - roc_curves.png              (courbes ROC)
  - metrics_summary.csv         (tableau pour rapport)
  - classification_reports.txt  (détails par classe)
        """
    )
    
    parser.add_argument('--epochs', type=int, default=5,
                       help=f'Nombre d\'epochs (défaut: 5)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help=f'Batch size (défaut: 8)')
    parser.add_argument('--skip-resnet', action='store_true',
                       help='Skip ResNet18 training (utilise modèle existant si disponible)')
    parser.add_argument('--skip-svm', action='store_true',
                       help='Skip SVM training')
    
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    
    logger.info("\n" + "╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "COMPARAISON DE MODÈLES" + " " * 36 + "║")
    logger.info("║" + " " * 10 + "EfficientNet-B0  vs  ResNet18  vs  SVM" + " " * 20 + "║")
    logger.info("╚" + "═" * 78 + "╝\n")
    
    logger.info(f"Configuration:")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Random seed: {RANDOM_SEED}")
    logger.info(f"  Dataset: {DATA_PATH}")
    logger.info(f"  Outputs: {OUTPUT_DIR}\n")
    
    # Préparer dataset
    train_loader, val_loader, dataset = prepare_dataset()
    
    # Container pour résultats
    results = []
    
    # Modèle 1: EfficientNet-B0 (baseline)
    result_eff = evaluate_efficientnet_baseline(val_loader, dataset)
    if result_eff:
        results.append(result_eff)
    
    # Modèle 2: ResNet18
    if not args.skip_resnet:
        result_res = train_and_evaluate_resnet18(train_loader, val_loader, dataset)
        results.append(result_res)
    else:
        logger.info("\n[INFO] ResNet18 training skipped")
    
    # Modèle 3: SVM
    if not args.skip_svm:
        result_svm = train_and_evaluate_svm(train_loader, val_loader, dataset)
        results.append(result_svm)
    else:
        logger.info("\n[INFO] SVM training skipped")
    
    # Générer graphiques
    if len(results) >= 2:
        create_comparison_plots(results)
    else:
        logger.warning("Pas assez de modèles évalués pour générer les graphiques")
    
    # Résumé final
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ FINAL")
    logger.info("=" * 80)
    
    for r in results:
        m = r['metrics']
        logger.info(f"\n{r['name']}:")
        logger.info(f"  Accuracy:  {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
        logger.info(f"  Precision: {m['precision']:.4f}")
        logger.info(f"  Recall:    {m['recall']:.4f}")
        logger.info(f"  F1-Score:  {m['f1']:.4f}")
        logger.info(f"  AUC-ROC:   {m['auc']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ COMPARAISON TERMINÉE")
    logger.info("=" * 80)
    logger.info(f"\nRésultats disponibles dans: {OUTPUT_DIR}/")
    logger.info("  - metrics_comparison.png")
    logger.info("  - confusion_matrices.png")
    logger.info("  - roc_curves.png")
    logger.info("  - metrics_summary.csv")
    logger.info("  - classification_reports.txt")
    logger.info("  - metrics_detailed.json\n")


if __name__ == "__main__":
    main()