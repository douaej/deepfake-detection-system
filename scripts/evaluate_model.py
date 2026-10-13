import torch
import timm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = "cpu"
DATA_PATH = "data/my_real_vs_ai_dataset"

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset
dataset = datasets.ImageFolder(DATA_PATH, transform=transform)

# On reprend le même subset 20k
from collections import defaultdict
import random

MAX_SAMPLES_PER_CLASS = 10000
class_indices = defaultdict(list)

for idx, (_, label) in enumerate(dataset.samples):
    class_indices[label].append(idx)

selected_indices = []
for label in class_indices:
    selected_indices.extend(
        random.sample(class_indices[label], MAX_SAMPLES_PER_CLASS)
    )

subset = torch.utils.data.Subset(dataset, selected_indices)

# Split
train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size
_, val_dataset = torch.utils.data.random_split(subset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=8)

# Load model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/pretrained/deepfake_efficientnet.pth"))
model.to(DEVICE)
model.eval()

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))
