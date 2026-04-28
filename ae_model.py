"""
ae_model.py — Autoencoder for PCB Anomaly Detection (Phase 1)
==============================================================
Architecture reconstructed from ae_model.pth weights:

Encoder:
  Conv2d(1, 16, 3, padding=1)  → ReLU
  Conv2d(16, 32, 3, padding=1) → ReLU  + MaxPool2d(2)

Decoder:
  ConvTranspose2d(32, 16, 3, padding=1) → ReLU  + Upsample(x2)
  ConvTranspose2d(16, 1, 3, padding=1)  → Sigmoid

Input : grayscale image  (1 × H × W)
Output: reconstructed image (1 × H × W)

Anomaly score = mean squared error between input and reconstruction.
If score > threshold → ANOMALY → pass to Phase 2
If score ≤ threshold → NORMAL  → stop
"""

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder — matches keys encoder.0 and encoder.2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # encoder.0
            nn.ReLU(inplace=True),                         # encoder.1 (no weights)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # encoder.2
            nn.ReLU(inplace=True),                         # encoder.3 (no weights)
            nn.MaxPool2d(2),                               # encoder.4 (no weights)
        )

        # Decoder — matches keys decoder.0 and decoder.2
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),  # decoder.0
            nn.ReLU(inplace=True),                                   # decoder.1
            nn.Upsample(scale_factor=2, mode="nearest"),            # decoder.2 (no weights)
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),   # decoder.3 → wait, see note
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AutoEncoderFlat(nn.Module):
    """
    Exact flat mapping matching state_dict keys:
      encoder.0  encoder.2
      decoder.0  decoder.2
    Only Conv layers have weights; activations and pool/upsample have none.
    """
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),    # [0] → encoder.0
            nn.ReLU(inplace=True),                          # [1]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # [2] → encoder.2
            nn.ReLU(inplace=True),                          # [3]
            nn.MaxPool2d(kernel_size=2, stride=2),          # [4]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),  # [0] → decoder.0
            nn.ReLU(inplace=True),                                   # [1]
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),   # [2] → decoder.2
            nn.Sigmoid(),                                            # [3]
        )

    def forward(self, x):
        z    = self.encoder(x)
        # Upsample back to original spatial size
        z_up = nn.functional.interpolate(z, scale_factor=2, mode="nearest")
        return self.decoder(z_up)


def load_ae(checkpoint_path, device="cpu"):
    """
    Load the autoencoder from ae_model.pth.
    Returns model in eval mode.
    """
    model = AutoEncoderFlat()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle wrapped checkpoints
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  ✔ AE loaded from {checkpoint_path}")
    return model


def reconstruction_error(model, img_tensor):
    """
    Compute mean squared reconstruction error for a single image tensor.
    img_tensor : (1, 1, H, W) grayscale tensor on same device as model
    Returns    : float scalar error
    """
    with torch.no_grad():
        recon = model(img_tensor)
        error = nn.functional.mse_loss(recon, img_tensor).item()
    return error


# ── Default threshold ──────────────────────────────────────────────────────────
# This is a starting estimate. Run calibrate_threshold() on your normal
# validation images to find the best value for your dataset.
DEFAULT_THRESHOLD = 0.01


def is_anomaly(error, threshold=DEFAULT_THRESHOLD):
    """Returns True if the reconstruction error exceeds the threshold."""
    return error > threshold


def calibrate_threshold(model, normal_image_paths, device="cpu",
                        percentile=95, img_size=128):
    """
    Compute the threshold automatically from a set of known-normal images.
    Set threshold = percentile of reconstruction errors on normal images.

    Usage:
        from pathlib import Path
        normal_imgs = list(Path("stage1_anomaly/normal").glob("*.jpg"))
        threshold = calibrate_threshold(model, normal_imgs)
        print(f"Recommended threshold: {threshold:.6f}")
    """
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    errors = []
    for p in normal_image_paths:
        try:
            img = Image.open(p).convert("L")
            x   = tf(img).unsqueeze(0).to(device)
            errors.append(reconstruction_error(model, x))
        except Exception:
            continue

    if not errors:
        print("  ⚠ No images processed — using default threshold")
        return DEFAULT_THRESHOLD

    threshold = float(np.percentile(errors, percentile))
    print(f"  ✔ Calibrated threshold ({percentile}th pct of "
          f"{len(errors)} images): {threshold:.6f}")
    return threshold


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ckpt = Path(r"C:/Users/Sastra/OneDrive/Desktop/PCB_mini/ae_model.pth")
    if not ckpt.exists():
        print(f"❌ Not found: {ckpt}"); sys.exit(1)

    model = load_ae(ckpt)
    print(f"\nModel architecture:\n{model}\n")

    # Quick smoke test
    import torch
    dummy = torch.randn(1, 1, 128, 128)
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    err = reconstruction_error(model, dummy)
    print(f"Recon error  : {err:.6f}")
    print("\n✔ ae_model.py works correctly")