#!/usr/bin/env python3
"""
Visualisation des architectures des modèles
Génère diagrammes, graphes de flux, tableaux de couches pour le rapport
"""

import os
import sys
from pathlib import Path
import json
import logging

import torch
import torch.nn as nn
from torchvision import models
import timm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns

# Setup paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "architecture"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== ARCHITECTURE ANALYSIS ====================

def get_model_summary(model, model_name):
    """Extrait informations détaillées sur l'architecture"""
    summary = {
        'name': model_name,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'layers': [],
        'layer_types': {}
    }
    
    # Analyser chaque couche
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
                'params': sum(p.numel() for p in module.parameters())
            }
            
            # Extraire dimensions si disponible
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                layer_info['in_features'] = module.in_features
                layer_info['out_features'] = module.out_features
            elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                layer_info['in_channels'] = module.in_channels
                layer_info['out_channels'] = module.out_channels
                if hasattr(module, 'kernel_size'):
                    layer_info['kernel_size'] = module.kernel_size
            
            summary['layers'].append(layer_info)
            
            # Compter types de couches
            layer_type = module.__class__.__name__
            summary['layer_types'][layer_type] = summary['layer_types'].get(layer_type, 0) + 1
    
    return summary


def save_architecture_text(summary, output_path):
    """Sauvegarde architecture en texte formaté"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"ARCHITECTURE: {summary['name']}\n")
        f.write("=" * 100 + "\n\n")
        
        # Stats globales
        f.write(f"Paramètres totaux:      {summary['total_params']:,}\n")
        f.write(f"Paramètres entraînables: {summary['trainable_params']:,}\n")
        f.write(f"Paramètres gelés:       {summary['total_params'] - summary['trainable_params']:,}\n")
        f.write(f"Nombre de couches:      {len(summary['layers'])}\n\n")
        
        # Types de couches
        f.write("Types de couches:\n")
        f.write("-" * 100 + "\n")
        for layer_type, count in sorted(summary['layer_types'].items(), key=lambda x: -x[1]):
            f.write(f"  {layer_type:40s} : {count:4d}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("DÉTAIL DES COUCHES\n")
        f.write("=" * 100 + "\n\n")
        
        # Détail des couches (premiers 50 pour éviter trop de verbosité)
        for i, layer in enumerate(summary['layers'][:50]):
            f.write(f"[{i+1}] {layer['name']}\n")
            f.write(f"    Type: {layer['type']}\n")
            f.write(f"    Params: {layer['params']:,}\n")
            
            if 'in_features' in layer:
                f.write(f"    Dimensions: {layer['in_features']} → {layer['out_features']}\n")
            elif 'in_channels' in layer:
                f.write(f"    Channels: {layer['in_channels']} → {layer['out_channels']}")
                if 'kernel_size' in layer:
                    f.write(f" | Kernel: {layer['kernel_size']}")
                f.write("\n")
            f.write("\n")
        
        if len(summary['layers']) > 50:
            f.write(f"... et {len(summary['layers']) - 50} autres couches\n")
    
    logger.info(f"  ✓ Texte: {output_path}")


# ==================== VISUAL DIAGRAMS ====================

def plot_architecture_comparison(summaries):
    """Diagramme comparatif des 3 architectures"""
    logger.info("  → Comparaison visuelle des architectures...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparaison des architectures CNN', fontsize=18, fontweight='bold')
    
    models = [s['name'] for s in summaries]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    # 1. Nombre total de paramètres
    ax = axes[0, 0]
    total_params = [s['total_params'] / 1e6 for s in summaries]
    bars = ax.bar(models, total_params, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, total_params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
               f'{val:.2f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Millions de paramètres', fontsize=12, fontweight='bold')
    ax.set_title('Taille des modèles', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. Paramètres entraînables vs gelés
    ax = axes[0, 1]
    trainable = [s['trainable_params'] / 1e6 for s in summaries]
    frozen = [(s['total_params'] - s['trainable_params']) / 1e6 for s in summaries]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, trainable, width, label='Entraînables', color='#4CAF50', alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, frozen, width, label='Gelés', color='#9E9E9E', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Millions de paramètres', fontsize=12, fontweight='bold')
    ax.set_title('Paramètres entraînables vs gelés', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Nombre de couches par type (top 5)
    ax = axes[1, 0]
    
    # Agréger les types de couches
    all_types = set()
    for s in summaries:
        all_types.update(s['layer_types'].keys())
    
    # Top 5 types les plus fréquents
    type_totals = {}
    for layer_type in all_types:
        type_totals[layer_type] = sum(s['layer_types'].get(layer_type, 0) for s in summaries)
    
    top_types = sorted(type_totals.items(), key=lambda x: -x[1])[:5]
    top_type_names = [t[0] for t in top_types]
    
    # Créer barplot groupé
    x = np.arange(len(models))
    width = 0.15
    
    for i, layer_type in enumerate(top_type_names):
        counts = [s['layer_types'].get(layer_type, 0) for s in summaries]
        offset = width * (i - 2)
        ax.bar(x + offset, counts, width, label=layer_type, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Nombre de couches', fontsize=12, fontweight='bold')
    ax.set_title('Types de couches (Top 5)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Profondeur du réseau (nombre total de couches)
    ax = axes[1, 1]
    depths = [len(s['layers']) for s in summaries]
    bars = ax.bar(models, depths, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, depths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Nombre de couches', fontsize=12, fontweight='bold')
    ax.set_title('Profondeur des réseaux', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'architecture_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    ✓ {save_path}")


def plot_network_diagram(summary, filename):
    """Diagramme simplifié du flux du réseau (style bloc)"""
    logger.info(f"  → Diagramme réseau: {summary['name']}...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Titre
    ax.text(5, 9.5, f"Architecture: {summary['name']}", 
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Analyser structure (simplifiée pour visualisation)
    # On regroupe les couches par blocs logiques
    
    if 'EfficientNet' in summary['name']:
        blocks = [
            {'name': 'Input', 'type': 'Conv2d', 'size': '224×224×3'},
            {'name': 'Stem', 'type': 'Conv+BN+Act', 'size': '112×112×32'},
            {'name': 'Block 1-2', 'type': 'MBConv', 'size': '56×56×24'},
            {'name': 'Block 3-4', 'type': 'MBConv', 'size': '28×28×40'},
            {'name': 'Block 5-6', 'type': 'MBConv', 'size': '14×14×112'},
            {'name': 'Block 7', 'type': 'MBConv', 'size': '7×7×320'},
            {'name': 'Head', 'type': 'Conv+Pool', 'size': '1×1×1280'},
            {'name': 'Classifier', 'type': 'FC', 'size': '2'},
        ]
    elif 'ResNet' in summary['name']:
        blocks = [
            {'name': 'Input', 'type': 'Conv2d', 'size': '224×224×3'},
            {'name': 'Conv1', 'type': 'Conv+BN+ReLU', 'size': '112×112×64'},
            {'name': 'Layer1', 'type': 'BasicBlock×2', 'size': '56×56×64'},
            {'name': 'Layer2', 'type': 'BasicBlock×2', 'size': '28×28×128'},
            {'name': 'Layer3', 'type': 'BasicBlock×2', 'size': '14×14×256'},
            {'name': 'Layer4', 'type': 'BasicBlock×2', 'size': '7×7×512'},
            {'name': 'AvgPool', 'type': 'AdaptiveAvgPool', 'size': '1×1×512'},
            {'name': 'FC', 'type': 'Linear', 'size': '2'},
        ]
    else:  # SVM (représentation symbolique)
        blocks = [
            {'name': 'Input', 'type': 'Image', 'size': '224×224×3'},
            {'name': 'ResNet18', 'type': 'Feature Extractor', 'size': '512'},
            {'name': 'SVM', 'type': 'RBF Kernel', 'size': '2'},
        ]
    
    # Dessiner blocs
    y_start = 8.5
    y_step = 1.0
    box_width = 6
    box_height = 0.6
    
    colors_map = {
        'Conv': '#4CAF50',
        'MBConv': '#2196F3',
        'BasicBlock': '#FF9800',
        'FC': '#E91E63',
        'Pool': '#9C27B0',
        'BN': '#00BCD4',
        'Feature': '#FFC107',
        'SVM': '#795548',
        'RBF': '#607D8B',
    }
    
    for i, block in enumerate(blocks):
        y = y_start - i * y_step
        
        # Déterminer couleur
        color = '#9E9E9E'  # Default
        for key, c in colors_map.items():
            if key in block['type']:
                color = c
                break
        
        # Dessiner bloc
        box = FancyBboxPatch(
            (2, y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.05", 
            edgecolor='black', facecolor=color, alpha=0.7, linewidth=2
        )
        ax.add_patch(box)
        
        # Texte dans le bloc
        ax.text(2 + box_width/2, y, block['name'],
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Type et taille à droite
        ax.text(8.5, y, f"{block['type']}\n{block['size']}",
               ha='left', va='center', fontsize=9, style='italic')
        
        # Flèche vers le bloc suivant
        if i < len(blocks) - 1:
            ax.annotate('', xy=(5, y - y_step/2 - 0.1), xytext=(5, y - box_height/2 - 0.05),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Légende
    legend_y = 0.8
    ax.text(0.5, legend_y, 'Légende:', fontsize=10, fontweight='bold')
    legend_items = [
        ('Conv/Stem', '#4CAF50'),
        ('MBConv/BasicBlock', '#2196F3'),
        ('FC/Classifier', '#E91E63'),
        ('Pooling', '#9C27B0'),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(Rectangle((0.5, legend_y - 0.3 - i*0.25), 0.3, 0.15, 
                               facecolor=color, edgecolor='black', alpha=0.7))
        ax.text(0.9, legend_y - 0.225 - i*0.25, label, fontsize=8, va='center')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    ✓ {save_path}")


def plot_layer_flow_detailed(summary, filename):
    """Diagramme détaillé du flux avec dimensions"""
    logger.info(f"  → Flux détaillé: {summary['name']}...")
    
    # Filtrer couches principales (Conv, Linear, etc.)
    main_layers = [
        l for l in summary['layers'] 
        if any(t in l['type'] for t in ['Conv', 'Linear', 'BatchNorm', 'AdaptiveAvgPool'])
    ][:30]  # Limiter à 30 pour lisibilité
    
    if not main_layers:
        logger.warning(f"    Pas de couches principales trouvées pour {summary['name']}")
        return
    
    fig, ax = plt.subplots(figsize=(16, len(main_layers) * 0.4 + 2))
    ax.set_xlim(0, 12)
    ax.set_ylim(-len(main_layers) - 1, 1)
    ax.axis('off')
    
    # Titre
    ax.text(6, 0.5, f"Flux des couches: {summary['name']}", 
           ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Dessiner chaque couche
    for i, layer in enumerate(main_layers):
        y = -i * 1.0
        
        # Nom de la couche
        layer_name = layer['name'].split('.')[-1] if '.' in layer['name'] else layer['name']
        ax.text(0.5, y, f"[{i+1}] {layer_name}", ha='left', va='center', 
               fontsize=9, fontweight='bold')
        
        # Type
        ax.text(3.5, y, layer['type'], ha='left', va='center', 
               fontsize=8, style='italic', color='#2196F3')
        
        # Dimensions
        if 'in_channels' in layer:
            dim_text = f"{layer['in_channels']}→{layer['out_channels']}"
            if 'kernel_size' in layer:
                k = layer['kernel_size']
                dim_text += f" K:{k}"
        elif 'in_features' in layer:
            dim_text = f"{layer['in_features']}→{layer['out_features']}"
        else:
            dim_text = "—"
        
        ax.text(6.5, y, dim_text, ha='left', va='center', 
               fontsize=8, color='#FF9800')
        
        # Params
        if layer['params'] > 0:
            params_text = f"{layer['params']:,}"
        else:
            params_text = "0"
        ax.text(9.5, y, params_text, ha='right', va='center', 
               fontsize=8, color='#4CAF50')
        
        # Flèche vers suivant
        if i < len(main_layers) - 1:
            ax.annotate('', xy=(0.3, y - 0.5), xytext=(0.3, y - 0.05),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # Headers
    ax.text(0.5, 0.2, 'Nom', ha='left', fontsize=9, fontweight='bold', color='gray')
    ax.text(3.5, 0.2, 'Type', ha='left', fontsize=9, fontweight='bold', color='gray')
    ax.text(6.5, 0.2, 'Dimensions', ha='left', fontsize=9, fontweight='bold', color='gray')
    ax.text(9.5, 0.2, 'Params', ha='right', fontsize=9, fontweight='bold', color='gray')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    ✓ {save_path}")


# ==================== PARAMETERS TABLE ====================

def create_params_table(summaries):
    """Tableau comparatif des paramètres"""
    logger.info("  → Tableau comparatif...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Données du tableau
    table_data = [
        ['Modèle', 'Params totaux', 'Entraînables', 'Gelés', '% Entraînable', 'Couches'],
    ]
    
    for s in summaries:
        total = s['total_params']
        trainable = s['trainable_params']
        frozen = total - trainable
        pct = (trainable / total * 100) if total > 0 else 0
        
        table_data.append([
            s['name'],
            f"{total / 1e6:.2f}M",
            f"{trainable / 1e6:.2f}M",
            f"{frozen / 1e6:.2f}M",
            f"{pct:.1f}%",
            str(len(s['layers']))
        ])
    
    # Créer tableau
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows
    colors = ['#E8F5E9', '#C8E6C9', '#FFF9C4']
    for i in range(1, len(table_data)):
        for j in range(6):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i-1])
            cell.set_edgecolor('gray')
    
    ax.set_title('Comparaison des paramètres des modèles', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'params_comparison_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    ✓ {save_path}")


# ==================== MAIN ====================

def main():
    logger.info("\n" + "╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "VISUALISATION DES ARCHITECTURES" + " " * 27 + "║")
    logger.info("╚" + "═" * 78 + "╝\n")
    
    # Charger les 3 modèles
    logger.info("Chargement des modèles...\n")
    
    # 1. EfficientNet-B0
    logger.info("1. EfficientNet-B0")
    efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
    eff_summary = get_model_summary(efficientnet, 'EfficientNet-B0')
    
    # 2. ResNet18
    logger.info("2. ResNet18")
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    res_summary = get_model_summary(resnet, 'ResNet18')
    
    # 3. SVM (représentation symbolique via ResNet18 feature extractor)
    logger.info("3. SVM (via ResNet18 features)")
    resnet_features = models.resnet18(weights=None)
    resnet_features = nn.Sequential(*list(resnet_features.children())[:-1])
    svm_summary = get_model_summary(resnet_features, 'SVM-ResNet18')
    
    summaries = [eff_summary, res_summary, svm_summary]
    
    logger.info("\n" + "=" * 80)
    logger.info("GÉNÉRATION DES VISUALISATIONS")
    logger.info("=" * 80 + "\n")
    
    # Sauvegarder architectures en texte
    logger.info("📄 Architectures texte...")
    for summary in summaries:
        filename = f"architecture_{summary['name'].replace(' ', '_').replace('-', '_').lower()}.txt"
        save_architecture_text(summary, OUTPUT_DIR / filename)
    
    # Sauvegarder JSON
    logger.info("\n📊 Export JSON...")
    json_path = OUTPUT_DIR / 'architectures_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([{
            'name': s['name'],
            'total_params': s['total_params'],
            'trainable_params': s['trainable_params'],
            'num_layers': len(s['layers']),
            'layer_types': s['layer_types']
        } for s in summaries], f, indent=2)
    logger.info(f"  ✓ {json_path}")
    
    # Graphiques
    logger.info("\n📈 Génération des graphiques...")
    plot_architecture_comparison(summaries)
    create_params_table(summaries)
    
    # Diagrammes individuels
    logger.info("\n🔷 Diagrammes par modèle...")
    plot_network_diagram(eff_summary, 'efficientnet_diagram.png')
    plot_network_diagram(res_summary, 'resnet18_diagram.png')
    plot_network_diagram(svm_summary, 'svm_diagram.png')
    
    # Flux détaillés
    logger.info("\n🔸 Flux détaillés...")
    plot_layer_flow_detailed(eff_summary, 'efficientnet_flow.png')
    plot_layer_flow_detailed(res_summary, 'resnet18_flow.png')
    
    # Résumé final
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ DES ARCHITECTURES")
    logger.info("=" * 80)
    
    for s in summaries:
        logger.info(f"\n{s['name']}:")
        logger.info(f"  Total params:      {s['total_params']:,} ({s['total_params']/1e6:.2f}M)")
        logger.info(f"  Trainable params:  {s['trainable_params']:,} ({s['trainable_params']/1e6:.2f}M)")
        logger.info(f"  Frozen params:     {s['total_params']-s['trainable_params']:,}")
        logger.info(f"  Nombre de couches: {len(s['layers'])}")
        logger.info(f"  Types principaux:  {', '.join(list(s['layer_types'].keys())[:5])}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ VISUALISATIONS GÉNÉRÉES")
    logger.info("=" * 80)
    logger.info(f"\nFichiers disponibles dans: {OUTPUT_DIR}/")
    logger.info("\nGraphiques:")
    logger.info("  - architecture_comparison.png    (comparaison globale)")
    logger.info("  - params_comparison_table.png    (tableau paramètres)")
    logger.info("  - efficientnet_diagram.png       (schéma bloc)")
    logger.info("  - resnet18_diagram.png           (schéma bloc)")
    logger.info("  - svm_diagram.png                (schéma bloc)")
    logger.info("  - efficientnet_flow.png          (flux détaillé)")
    logger.info("  - resnet18_flow.png              (flux détaillé)")
    logger.info("\nTexte:")
    logger.info("  - architecture_efficientnet_b0.txt")
    logger.info("  - architecture_resnet18.txt")
    logger.info("  - architecture_svm_resnet18.txt")
    logger.info("  - architectures_summary.json\n")


if __name__ == "__main__":
    main()