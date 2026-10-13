import torch
import timm
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ===== CONFIG =====
DATA_PATH = "data/my_real_vs_ai_dataset"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
DEVICE = "cpu"

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== DATASET =====
dataset = datasets.ImageFolder(DATA_PATH, transform=transform)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

# ---- Subset 20k (10k per class) ----
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

print("Subset size:", len(subset))

# ---- Train / Val Split ----
train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size

train_dataset, val_dataset = random_split(subset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ===== MODEL =====
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/deepfake_efficientnet.pth"))
model.to(DEVICE)

# ---- Freeze backbone ----
# Freeze tout
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# Unfreeze last 2 blocks
for name, param in model.named_parameters():
    if "blocks.6" in name or "blocks.7" in name:
        param.requires_grad = True


criterion = nn.CrossEntropyLoss()

# Optimizer seulement sur le classifier
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)


# ===== TRAINING LOOP =====
for epoch in range(EPOCHS):

    # ===== TRAINING =====
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {accuracy:.2f}%")
    print("-" * 40)


# ===== SAVE MODEL =====
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/deepfake_efficientnet.pth")

print("Model saved successfully!")


