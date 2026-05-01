"""
PCB Defect Detection - Streamlit App (Three-Phase Pipeline)
===========================================================
Phase 0 : PCB Validator      - MobileNetV3 binary classifier (PCB vs Non-PCB)
Phase 1 : Autoencoder        - Reconstruction error -> Normal vs Anomaly
Phase 2 : MobileViT-S        - Defect type + Grad-CAM + Bounding Box + Gemini Explanation
Explanation: Google Gemini API (free tier) via plain requests -- no openai package needed.
"""

import io
import os
import json
import warnings
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

warnings.filterwarnings("ignore")
# ---------------------------------------------------------------------------
# HUGGING FACE CONFIG (UPDATED)
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # MUST set in environment

HF_MODELS = [
    "google/flan-t5-base",
    "google/flan-t5-large",
]

HF_BASE = "https://api-inference.huggingface.co/models/"
MAX_RETRIES = 5
BACKOFF = 2


# ---------------------------------------------------------------------------
# FALLBACK EXPLANATION (NEVER FAILS)
# ---------------------------------------------------------------------------
def fallback_explanation(label: str, confidence: float) -> str:
    return f"""
Defect Type: {label.replace('_',' ').title()}
Confidence: {confidence:.1%}

1. What is this defect?
This defect indicates an irregularity in the PCB layout or manufacturing process that may affect circuit performance and reliability.

2. Possible causes:
- Misalignment during drilling or etching
- Over-etching or under-etching of copper layers
- Material contamination or defects
- Improper fabrication calibration

3. Recommended corrective actions:
- Perform visual and electrical inspection
- Recalibrate manufacturing equipment
- Improve quality control checks
- Use higher precision fabrication settings
"""


# ---------------------------------------------------------------------------
# HUGGING FACE EXPLANATION ENGINE (ROBUST VERSION)
# ---------------------------------------------------------------------------
def generate_explanation(label: str, confidence: float) -> str:
    import time

    if not HF_TOKEN:
        return fallback_explanation(label, confidence) + "\n\n(HF token not set)"

    prompt = (
        f"You are an expert PCB manufacturing engineer.\n\n"
        f"Defect detected: {label.replace('_', ' ')}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"Provide:\n"
        f"1. Explanation\n"
        f"2. Causes\n"
        f"3. Corrective actions\n"
    )

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    for model in HF_MODELS:
        url = HF_BASE + model

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.3,
                "return_full_text": False,
            },
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)

                # -------------------------
                # Handle API states
                # -------------------------
                if resp.status_code == 503:
                    time.sleep(BACKOFF ** attempt)
                    continue

                if resp.status_code == 429:
                    time.sleep(BACKOFF ** attempt)
                    continue

                if resp.status_code == 404:
                    break  # try next model

                if resp.status_code == 401:
                    return fallback_explanation(label, confidence) + "\n\n(Invalid HF token)"

                resp.raise_for_status()
                data = resp.json()

                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "").strip()
                    if text:
                        return f"[Model: {model.split('/')[-1]}]\n\n{text}"

            except requests.exceptions.Timeout:
                continue
            except Exception:
                break

    # If everything fails → fallback
    return fallback_explanation(label, confidence) + "\n\n(HF servers busy, fallback used)"
#------------
# PAGE CONFIG  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon=":mag:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# IMPORTS from project modules
# ---------------------------------------------------------------------------
try:
    from test import (PCBViT, CFG, load_checkpoint, predict_tta,
                      get_gradcam_target_layer, GradCAM,
                      overlay, get_bounding_box, draw_bbox, _VAL_TRANSFORM)
    from ae_model import AutoEncoderFlat, load_ae, reconstruction_error
    import cv2
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
AE_CHECKPOINT = Path("outputs/ae_model.pth")
AE_IMG_SIZE   = 128
AE_THRESHOLD  = 0.01

_AE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((AE_IMG_SIZE, AE_IMG_SIZE)),
    transforms.ToTensor(),
])

PCB_CLASSES = ["non_pcb", "pcb"]
PCB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------------------------------------------------------
# MODEL LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pcb_classifier():
    import timm
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("pcb_classifier.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_phase1_model():
    if not AE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"ae_model.pth not found at: {AE_CHECKPOINT.resolve()} -- "
            f"make sure outputs/ae_model.pth is present."
        )
    return load_ae(AE_CHECKPOINT, device=CFG.device)

@st.cache_resource
def load_phase2_model():
    model = PCBViT().to(CFG.device)
    load_checkpoint(CFG.CHECKPOINT, model)
    model.eval()
    return model

@st.cache_resource
def load_seg_model():
    try:
        from segmentation_model import UNet
        seg = UNet().to(CFG.device)
        seg_path = CFG.output_dir / "unet.pth"
        if seg_path.exists():
            seg.load_state_dict(torch.load(seg_path, map_location=CFG.device))
            seg.eval()
            return seg
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# PHASE 0 - PCB VALIDATOR
# ---------------------------------------------------------------------------
def is_pcb_image(img_pil, threshold=0.80):
    x = PCB_TRANSFORM(img_pil).unsqueeze(0)
    with torch.no_grad():
        out = PCB_MODEL(x)
        probs = torch.softmax(out, dim=1)[0]
    pred = probs.argmax().item()
    confidence = probs[pred].item()
    label = PCB_CLASSES[pred]
    is_pcb = (label == "pcb") and (confidence > threshold)
    return is_pcb, label, confidence

# ---------------------------------------------------------------------------
# PHASE 1 - AUTOENCODER
# ---------------------------------------------------------------------------
def run_phase1(ae_model, img_pil, threshold):
    x = _AE_TRANSFORM(img_pil).unsqueeze(0).to(CFG.device)
    with torch.no_grad():
        recon = ae_model(x)
    error      = F.mse_loss(recon, x).item()
    is_anomaly = error > threshold

    recon_np = recon[0, 0].cpu().numpy()
    r_min, r_max = recon_np.min(), recon_np.max()
    if r_max > r_min:
        recon_np = (recon_np - r_min) / (r_max - r_min)
    else:
        recon_np = np.zeros_like(recon_np)
    recon_np  = (recon_np * 255).astype(np.uint8)
    recon_pil = Image.fromarray(recon_np, mode="L").convert("RGB")
    return error, is_anomaly, recon_pil

# ---------------------------------------------------------------------------
# PHASE 2 - GRAD-CAM
# ---------------------------------------------------------------------------
def run_gradcam(model, img_pil, class_idx):
    tl     = get_gradcam_target_layer(model)
    cam    = np.zeros((CFG.img_size, CFG.img_size), dtype=np.float32)
    img_np = np.array(img_pil.resize((CFG.img_size, CFG.img_size)))
    if tl is not None:
        gc = GradCAM(model, tl)
        x  = (_VAL_TRANSFORM(img_pil)
              .unsqueeze(0).to(CFG.device).requires_grad_(True))
        try:
            cam, _ = gc.generate(x, class_idx)
        except Exception as e:
            st.warning(f"Grad-CAM: {e}")
        gc.remove()
    bbox = get_bounding_box(cam)
    return cam, bbox, img_np


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)

# ---------------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------------
with st.spinner("Loading models..."):
    try:
        PCB_MODEL = load_pcb_classifier()
        ae_model  = load_phase1_model()
        p2_model  = load_phase2_model()
        seg_model = load_seg_model()
        st.sidebar.success(f"Models loaded ({CFG.device.upper()})")
        try:
            ckpt = torch.load(CFG.CHECKPOINT, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                st.sidebar.metric("Phase 2 Val Acc", f"{ckpt.get('best_acc', 0):.1%}")
                st.sidebar.metric("Trained Epochs",  str(ckpt.get("epoch", "?")))
        except Exception:
            pass
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown("""
**PCB Defect Detection -- 3-Phase Pipeline**

**Phase 0 -- PCB Validator**
MobileNetV3 binary classifier checks whether
the uploaded image is actually a PCB.
Non-PCB images are rejected immediately.

**Phase 1 -- Autoencoder**
Detects normal vs anomaly using
reconstruction error threshold.

**Phase 2 -- MobileViT-S**
Classifies defect type and localises
with Grad-CAM + bounding box.
Mistral-7B via HuggingFace free API
generates the defect explanation.

**6 defect classes:**
Missing Hole | Mouse Bite | Open Circuit
Short | Spur | Spurious Copper
""")
    st.divider()
    st.header("Settings")
    threshold = st.slider(
        "Phase 1 Anomaly Threshold",
        min_value=0.001, max_value=0.050,
        value=AE_THRESHOLD, step=0.001, format="%.3f",
        help="Lower = more sensitive. Higher = fewer false alarms.",
    )

    st.divider()
    st.header("How to use")
    st.markdown("""
1. Upload a PCB image (JPG/PNG)
2. **Phase 0** validates it is a PCB
3. **Phase 1** checks normal vs anomaly
4. If anomaly -- **Phase 2** classifies type
5. Read the AI explanation
6. Click **Generate Heatmap** for Grad-CAM
""")

# ---------------------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------------------
st.title("PCB Defect Detection System")
st.markdown(
    "**Three-phase pipeline:** PCB Validator -> Autoencoder anomaly detection -> "
    "MobileViT-S defect classification + Grad-CAM localisation."
)

uploaded = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png", "bmp"])
if not uploaded:
    st.info("Upload a PCB image to begin.")
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")

# ===========================================================================
# PHASE 0 -- PCB VALIDATION
# ===========================================================================
st.markdown("---")
st.subheader("Phase 0 -- PCB Validator")

with st.spinner("Validating image..."):
    pcb_valid, pcb_label, pcb_conf = is_pcb_image(img_pil)

col_img, col_result = st.columns(2)
with col_img:
    st.image(img_pil, caption="Uploaded Image", use_container_width=True)
with col_result:
    st.metric("Prediction",  pcb_label.upper())
    st.metric("Confidence",  f"{pcb_conf:.1%}")

    if pcb_valid:
        st.success("PCB detected -- proceeding to Phase 1 & 2")
    else:
        st.error("Not a PCB image")
        st.warning("Please upload a proper PCB image to continue.")
        st.stop()

# ===========================================================================
# PHASE 1 -- ANOMALY DETECTION
# ===========================================================================
st.markdown("---")
st.subheader("Phase 1 -- Anomaly Detection (Autoencoder)")

with st.spinner("Phase 1 running..."):
    error, anomaly_flag, recon_pil = run_phase1(ae_model, img_pil, threshold)

c1, c3 = st.columns(2)
with c1:
    st.image(img_pil, caption="Original Image", use_container_width=True)
with c3:
    st.markdown("#### Phase 1 Result")
    st.metric("Reconstruction Error", f"{error:.5f}")
    st.metric("Threshold",            f"{threshold:.3f}")
    st.metric("Error / Threshold",    f"{error/threshold:.2f}x")

    pct       = min(error / (threshold * 2), 1.0)
    bar_color = "#e74c3c" if anomaly_flag else "#2ecc71"
    st.markdown(
        f"""<div style="background:#ddd;border-radius:6px;height:14px;margin:8px 0">
              <div style="width:{pct*100:.0f}%;background:{bar_color};
                          height:14px;border-radius:6px"></div>
            </div>""",
        unsafe_allow_html=True,
    )
    if anomaly_flag:
        st.error("ANOMALY DETECTED -- continuing to Phase 2")
    else:
        st.success("NORMAL PCB -- no defect found")
        st.info("Pipeline stops here.")
        st.markdown("---")
        st.subheader("Pipeline Summary")
        st.success(
            f"This PCB appears **normal**. Reconstruction error ({error:.5f}) "
            f"is below threshold ({threshold:.3f})."
        )
        st.stop()

# ===========================================================================
# PHASE 2 -- DEFECT CLASSIFICATION
# ===========================================================================
st.markdown("---")
st.subheader("Phase 2 -- Defect Classification (MobileViT-S)")

with st.spinner("Phase 2 running..."):
    result = predict_tta(p2_model, img_pil)

if len(result) == 4:
    label, probs, is_ambiguous, ambig_info = result
else:
    label, probs = result
    is_ambiguous, ambig_info = False, None

probs = np.array(probs)

if is_ambiguous and ambig_info:
    class1, class2, prob1, prob2 = ambig_info
    predicted_idx = CFG.class2idx[class1]
    display_label = f"{class1} / {class2}"
    confidence    = max(prob1, prob2)
else:
    predicted_idx = CFG.class2idx[label]
    confidence    = float(probs[predicted_idx])
    display_label = label

cp, cc = st.columns([1, 2])
with cp:
    st.markdown("#### Prediction")
    if is_ambiguous and ambig_info:
        st.warning("Ambiguous prediction")
        st.metric("Class 1", class1.replace("_"," ").title(), f"{prob1:.1%}")
        st.metric("Class 2", class2.replace("_"," ").title(), f"{prob2:.1%}")
        st.caption(f"Diff: {abs(prob1-prob2)*100:.1f}%")
    else:
        st.success(f"**{label.replace('_',' ').title()}**")
        st.metric("Confidence", f"{confidence:.1%}")

with cc:
    st.markdown("#### Class Probabilities")
    fig_bar, ax = plt.subplots(figsize=(7, 3))
    colors = []
    for i, cls in enumerate(CFG.classes):
        if is_ambiguous and ambig_info and cls in [ambig_info[0], ambig_info[1]]:
            colors.append("#e67e22")
        elif not is_ambiguous and cls == label:
            colors.append("#2ecc71")
        else:
            colors.append("#3498db")
    bars = ax.barh([c.replace("_"," ").title() for c in CFG.classes],
                   probs, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    for bar, prob in zip(bars, probs):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                f"{prob:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

# ===========================================================================
# GEMINI EXPLANATION
# ===========================================================================
st.markdown("---")
st.subheader("AI Explanation & Recommended Actions")

EXPLAIN_THRESHOLD = 0.20

if confidence >= EXPLAIN_THRESHOLD or is_ambiguous:
    if is_ambiguous:
        st.warning(
            "Ambiguous prediction -- model is uncertain between two defect classes. "
            "Explanation covers both candidates."
        )
    with st.spinner("Asking Mistral-7B via HuggingFace..."):
        explanation = generate_explanation(display_label, confidence)
    st.info(explanation)
    if confidence < 0.5:
        st.caption(
            f"Low confidence ({confidence:.1%}) -- verify result manually "
            f"or upload a clearer image."
        )
else:
    st.warning(
        f"Confidence too low ({confidence:.1%}) -- skipping explanation. "
        f"Try adjusting the Phase 1 threshold or upload a higher-quality image."
    )

# ===========================================================================
# GRAD-CAM + BOUNDING BOX
# ===========================================================================
st.markdown("---")
st.subheader("Defect Localisation -- Grad-CAM + Bounding Box")

if st.button("Generate Heatmap + Bounding Box"):
    with st.spinner("Running Grad-CAM..."):
        try:
            cam, bbox, img_np = run_gradcam(p2_model, img_pil, predicted_idx)
            box_color = (255, 165, 0) if is_ambiguous else (0, 255, 80)
            img_box   = draw_bbox(img_np, bbox, display_label, confidence, color=box_color)

            fig_vis, axes = plt.subplots(1, 3, figsize=(16, 5))
            title = (
                f"AMBIGUOUS: {display_label.upper()}"
                if is_ambiguous
                else f"Defect: {label.replace('_',' ').upper()}  |  Confidence: {confidence:.1%}"
            )
            fig_vis.suptitle(title, fontsize=13, fontweight="bold",
                             color="#e67e22" if is_ambiguous else "#c0392b")
            axes[0].imshow(img_np);                         axes[0].set_title("Original");    axes[0].axis("off")
            axes[1].imshow(overlay(img_np, cam, alpha=0.55)); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
            axes[2].imshow(img_box);                        axes[2].set_title("Bounding Box"); axes[2].axis("off")
            plt.tight_layout()
            st.image(fig_to_pil(fig_vis), use_container_width=True)
            plt.close(fig_vis)

            if bbox:
                x1, y1, x2, y2 = bbox
                w, h = x2-x1, y2-y1
                st.info(
                    f"Bbox: ({x1},{y1}) -> ({x2},{y2})  |  "
                    f"Size: {w}x{h} px  |  Area: {w*h} px^2"
                )
            else:
                st.warning("No bounding box detected.")

        except Exception as e:
            st.error(f"Visualisation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# ===========================================================================
# PIPELINE SUMMARY
# ===========================================================================
st.markdown("---")
st.subheader("Pipeline Summary")
cs1, cs2, cs3 = st.columns(3)
with cs1:
    st.markdown("**Phase 0 -- PCB Validator**")
    st.table({
        "Metric": ["Prediction", "Confidence", "Result"],
        "Value":  [pcb_label.upper(), f"{pcb_conf:.1%}", "VALID PCB"],
    })
with cs2:
    st.markdown("**Phase 1 -- Autoencoder**")
    st.table({
        "Metric": ["Reconstruction Error", "Threshold", "Result"],
        "Value":  [f"{error:.5f}", f"{threshold:.3f}", "ANOMALY"],
    })
with cs3:
    st.markdown("**Phase 2 -- MobileViT-S**")
    st.table({
        "Metric": ["Predicted Class", "Confidence", "Ambiguous"],
        "Value":  [display_label, f"{confidence:.1%}",
                   "Yes" if is_ambiguous else "No"],
    })