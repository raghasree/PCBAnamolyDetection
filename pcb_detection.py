"""
pcb_detection.py
----------------
Phase 0: PCB vs NON-PCB classifier
"""

import torch
import timm
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "pcb_classifier.pth"   # make sure this exists
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["non_pcb", "pcb"]

# ─────────────────────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
def load_model():
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=False,
        num_classes=2
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

# Load once
model = load_model()

# ─────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────
def detect_pcb(img_pil, threshold=0.80):
    """
    Input: PIL Image
    Output: (is_pcb, label, confidence)
    """

    x = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    label = CLASSES[pred_idx]

    # Apply confidence threshold
    is_pcb = (label == "pcb") and (confidence > threshold)

    return is_pcb, label, confidence


# ─────────────────────────────────────────────────────────────
# TEST (run file directly)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img = Image.open("test.jpg").convert("RGB")

    is_pcb, label, conf = detect_pcb(img)

    print(f"Prediction : {label}")
    print(f"Confidence : {conf:.2%}")

    if is_pcb:
        print("✅ PCB detected")
    else:
        print("❌ Not a PCB")