"""
PCB Defect Detection — Streamlit App (Two-Phase Pipeline)
==========================================================
Phase 1 : Autoencoder reconstruction error → Normal vs Anomaly
Phase 2 : MobileViT-S classification → Defect type + Grad-CAM + Bbox
"""

import io
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path   # ← ADD this line

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    from test import (PCBViT, CFG, load_checkpoint, predict_tta,
                      get_gradcam_target_layer, GradCAM,
                      overlay, get_bounding_box, draw_bbox, _VAL_TRANSFORM)
    from ae_model import AutoEncoderFlat, load_ae, reconstruction_error
    import cv2
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
AE_CHECKPOINT = CFG.output_dir.parent / "ae_model.pth"
AE_IMG_SIZE   = 128
AE_THRESHOLD  = 0.01

_AE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((AE_IMG_SIZE, AE_IMG_SIZE)),
    transforms.ToTensor(),
])

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_phase1_model():
    if not AE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"ae_model.pth not found at: {AE_CHECKPOINT.resolve()} — "
            f"make sure outputs/ae_model.pth is committed to GitHub."
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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run_phase1(ae_model, img_pil, threshold):
    x = _AE_TRANSFORM(img_pil).unsqueeze(0).to(CFG.device)
    with torch.no_grad():
        recon = ae_model(x)
    error      = F.mse_loss(recon, x).item()
    is_anomaly = error > threshold

    recon_np = recon[0, 0].cpu().numpy()

    # Auto-stretch: map actual min→max to 0→255 so it's always visible
    r_min, r_max = recon_np.min(), recon_np.max()
    if r_max > r_min:
        recon_np = (recon_np - r_min) / (r_max - r_min)
    else:
        recon_np = np.zeros_like(recon_np)  # truly empty output

    recon_np  = (recon_np * 255).astype(np.uint8)
    recon_pil = Image.fromarray(recon_np, mode="L").convert("RGB")
    return error, is_anomaly, recon_pil

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


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**PCB Defect Detection — 2-Phase**

**Phase 1 — Autoencoder**
Detects normal vs anomaly using
reconstruction error threshold.

**Phase 2 — MobileViT-S**
Classifies defect type and localises
with Grad-CAM + bounding box.

**6 defect classes:**
Missing Hole · Mouse Bite · Open Circuit
Short · Spur · Spurious Copper
""")
    st.divider()
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Phase 1 Anomaly Threshold",
        min_value=0.001, max_value=0.050,
        value=AE_THRESHOLD, step=0.001, format="%.3f",
        help="Lower = more sensitive. Higher = fewer false alarms.",
    )
    st.divider()
    st.header("🔧 How to use")
    st.markdown("""
1. Upload a PCB image (JPG/PNG)
2. Phase 1 checks normal vs anomaly
3. If anomaly → Phase 2 classifies type
4. Click **Generate Heatmap** for Grad-CAM
""")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    try:
        ae_model  = load_phase1_model()
        p2_model  = load_phase2_model()
        seg_model = load_seg_model()
        st.sidebar.success(f"✅ Models loaded ({CFG.device.upper()})")
        try:
            ckpt = torch.load(CFG.CHECKPOINT, map_location="cpu",
                              weights_only=False)
            if isinstance(ckpt, dict):
                st.sidebar.metric("Phase 2 Val Acc",
                                  f"{ckpt.get('best_acc', 0):.1%}")
                st.sidebar.metric("Trained Epochs",
                                  str(ckpt.get("epoch", "?")))
        except Exception:
            pass
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔍 PCB Defect Detection System")
st.markdown(
    "**Two-phase pipeline:** Autoencoder anomaly detection → MobileViT-S defect classification + Grad-CAM localisation.")

uploaded = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png", "bmp"])
if not uploaded:
    st.info("👆 Upload a PCB image to begin.")
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔵 Phase 1 — Anomaly Detection (Autoencoder)")

with st.spinner("Phase 1 running…"):
    error, anomaly_flag, recon_pil = run_phase1(ae_model, img_pil, threshold)

c1, c3 = st.columns(2)
with c1:
    st.image(img_pil, caption="Original Image", use_container_width=True)
with c3:
    st.markdown("#### Phase 1 Result")
    st.metric("Reconstruction Error", f"{error:.5f}")
    st.metric("Threshold",            f"{threshold:.3f}")
    st.metric("Error / Threshold",    f"{error/threshold:.2f}×")

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
        st.error("🚨 **ANOMALY DETECTED** — continuing to Phase 2")
    else:
        st.success("✅ **NORMAL PCB** — no defect found")
        st.info("Pipeline stops here.")
        st.markdown("---")
        st.subheader("📋 Summary")
        st.success(f"This PCB appears **normal**. Reconstruction error ({error:.5f}) "
                   f"is below threshold ({threshold:.3f}).")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🟠 Phase 2 — Defect Classification (MobileViT-S)")

with st.spinner("Phase 2 running…"):
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
        st.warning("⚠️ **Ambiguous**")
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

# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM + BBOX
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Defect Localisation — Grad-CAM + Bounding Box")

if st.button("🔥 Generate Heatmap + Bounding Box"):
    with st.spinner("Running Grad-CAM…"):
        try:
            cam, bbox, img_np = run_gradcam(p2_model, img_pil, predicted_idx)
            box_color = (255, 165, 0) if is_ambiguous else (0, 255, 80)
            img_box   = draw_bbox(img_np, bbox, display_label,
                                  confidence, color=box_color)

            fig_vis, axes = plt.subplots(1, 3, figsize=(16, 5))
            title = (f"⚠ AMBIGUOUS: {display_label.upper()}"
                     if is_ambiguous
                     else f"Defect: {label.replace('_',' ').upper()}  |  "
                          f"Confidence: {confidence:.1%}")
            fig_vis.suptitle(title, fontsize=13, fontweight="bold",
                             color="#e67e22" if is_ambiguous else "#c0392b")
            axes[0].imshow(img_np);                  axes[0].set_title("Original");    axes[0].axis("off")
            axes[1].imshow(overlay(img_np, cam, alpha=0.55)); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
            axes[2].imshow(img_box);                 axes[2].set_title("Bounding Box"); axes[2].axis("off")
            plt.tight_layout()
            st.image(fig_to_pil(fig_vis), use_container_width=True)
            plt.close(fig_vis)

            if bbox:
                x1, y1, x2, y2 = bbox
                w, h = x2-x1, y2-y1
                st.info(f"📦 **Bbox:** ({x1},{y1}) → ({x2},{y2})  |  "
                        f"Size: {w}×{h} px  |  Area: {w*h} px²")
            else:
                st.warning("No bounding box detected.")

        except Exception as e:
            st.error(f"Visualisation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Pipeline Summary")
cs1, cs2 = st.columns(2)
with cs1:
    st.markdown("**Phase 1 — Autoencoder**")
    st.table({
        "Metric": ["Reconstruction Error", "Threshold", "Result"],
        "Value":  [f"{error:.5f}", f"{threshold:.3f}", "ANOMALY ⚠"],
    })
with cs2:
    st.markdown("**Phase 2 — MobileViT-S**")
    st.table({
        "Metric": ["Predicted Class", "Confidence", "Ambiguous"],
        "Value":  [display_label, f"{confidence:.1%}",
                   "Yes ⚠" if is_ambiguous else "No ✔"],
    })