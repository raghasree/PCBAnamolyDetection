import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import timm
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4

PCB_PATH = "../stage2_classification/train"
NON_PCB_PATHS = [
    "../natural_images"
]

# ─────────────────────────────────────────────
# DEBUG PATH CHECK (VERY IMPORTANT)
# ─────────────────────────────────────────────
print("Checking dataset paths...")
print("PCB:", os.path.exists(PCB_PATH))
for p in NON_PCB_PATHS:
    print(f"NON-PCB ({p}):", os.path.exists(p))
print("-" * 40)

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ─────────────────────────────────────────────
# CUSTOM DATASET
# ─────────────────────────────────────────────
class PCBNonPCBDataset(Dataset):
    def __init__(self, pcb_path, non_pcb_paths, transform=None):
        self.samples = []
        self.transform = transform

        # PCB → label 1
        pcb_dataset = datasets.ImageFolder(pcb_path)
        for path, _ in pcb_dataset.samples:
            self.samples.append((path, 1))

        # NON-PCB → label 0
        for np_path in non_pcb_paths:
            if not os.path.exists(np_path):
                print(f"⚠ Skipping missing path: {np_path}")
                continue

            np_dataset = datasets.ImageFolder(np_path)
            for path, _ in np_dataset.samples:
                self.samples.append((path, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
dataset = PCBNonPCBDataset(
    pcb_path=PCB_PATH,
    non_pcb_paths=NON_PCB_PATHS,
    transform=transform
)

print(f"Total samples: {len(dataset)}")

# Train/Validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = timm.create_model(
    "mobilenetv3_small_100",
    pretrained=True,
    num_classes=2
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ─────────────────────────────────────────────
# TRAIN LOOP
# ─────────────────────────────────────────────
for epoch in range(EPOCHS):
    # TRAIN
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = outputs.argmax(1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)

            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print("-" * 40)

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
torch.save(model.state_dict(), "pcb_classifier.pth")

print("\n✅ Model saved as pcb_classifier.pth")