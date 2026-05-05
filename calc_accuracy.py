import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── 1. Run inference on test set ──────────────────────────────────────────────
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── 2. Class names ────────────────────────────────────────────────────────────
class_names = [
    'Missing Hole',
    'Mouse Bite',
    'Open Circuit',
    'Short',
    'Spur',
    'Spurious Copper'
]

# ── 3. Per-class accuracy (diagonal of normalized confusion matrix) ───────────
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # row normalize
per_class_accuracy = cm_normalized.diagonal() * 100  # percentage

print("Per-Class Accuracy:")
for name, acc in zip(class_names, per_class_accuracy):
    print(f"  {name:20s}: {acc:.2f}%")

# ── 4. Full classification report (Precision, Recall, F1) ────────────────────
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ── 5. Plot: Confusion Matrix ─────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix – MobileViT-S Defect Classifier')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ── 6. Plot: Per-class accuracy bar chart ────────────────────────────────────
plt.figure(figsize=(8, 4))
colors = ['#2E5D8B' if a >= 90 else '#E8593C' for a in per_class_accuracy]
bars = plt.bar(class_names, per_class_accuracy, color=colors, edgecolor='white')
plt.axhline(y=per_class_accuracy.mean(), color='gray',
            linestyle='--', label=f'Mean: {per_class_accuracy.mean():.1f}%')
plt.ylabel('Accuracy (%)')
plt.title('Per-Class Accuracy – MobileViT-S')
plt.ylim(70, 100)
plt.xticks(rotation=25, ha='right')
for bar, acc in zip(bars, per_class_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
plt.legend()
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=150)
plt.show()

# ── 7. Plot: Class distribution bar chart (for Chapter 3) ────────────────────
class_counts = [len(np.where(all_labels == i)[0]) for i in range(len(class_names))]
plt.figure(figsize=(7, 4))
plt.bar(class_names, class_counts, color='#2E5D8B', edgecolor='white')
plt.ylabel('Number of samples')
plt.title('Defect Class Distribution in Test Set')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()