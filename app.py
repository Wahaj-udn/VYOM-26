import os
import io
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template
from PIL import Image, ExifTags
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB for many images

# ── Model (loaded once) ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
encoder = nn.Sequential(*list(resnet.children())[:-2])
encoder = encoder.to(device)
encoder.eval()
print("Model loaded.")

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE        = 256
THRESHOLD       = 0.55
MIN_REGION_SIZE = 30

# ── Core helpers ──────────────────────────────────────────────────────────────

def load_and_preprocess(pil_image, size=IMG_SIZE):
    img        = pil_image.convert("RGB")
    orig_size  = img.size
    img_r      = img.resize((size, size), Image.Resampling.BILINEAR)
    img_np     = np.array(img_r).astype(np.float32) / 255.0
    tensor     = transforms.ToTensor()(img_r)
    normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std =[0.229, 0.224, 0.225])
    tensor     = normalize(tensor)
    return img_r, img_np, tensor.unsqueeze(0), orig_size


def encode(tensor):
    with torch.no_grad():
        return encoder(tensor.to(device))


def compute_change_map(feat_a, feat_b):
    diff       = torch.abs(feat_a - feat_b)
    cmap       = diff.mean(dim=1)
    cmap       = F.interpolate(cmap.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE),
                               mode="bilinear", align_corners=False)
    cmap       = cmap.squeeze().cpu().numpy()
    cmap       = (cmap - cmap.min()) / (cmap.max() - cmap.min() + 1e-8)
    return cmap


def clean_mask(change_map, threshold, min_region):
    mask               = change_map > threshold
    labeled, n         = ndimage.label(mask)
    sizes              = np.bincount(labeled.ravel())
    out                = mask.copy()
    for cid in range(1, n + 1):
        if sizes[cid] < min_region:
            out[labeled == cid] = False
    return out, n


def heatmap_b64(change_map):
    fig, ax = plt.subplots(figsize=(5, 5))
    im      = ax.imshow(change_map, cmap='jet', vmin=0, vmax=1)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Change intensity")
    buf     = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def overlay_b64(img_np, mask):
    orig    = (img_np * 255).astype(np.uint8)
    red     = orig.copy()
    red[mask] = [255, 60, 60]
    blended = cv2.addWeighted(orig, 0.55, red, 0.45, 0)
    return arr_to_b64(blended)


def arr_to_b64(arr_uint8):
    pil = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def cumulative_hotspot_b64(masks):
    """Pixels that changed in the most transitions get brightest colour."""
    stack  = np.stack([m.astype(np.float32) for m in masks], axis=0)
    counts = stack.sum(axis=0)                          # 0 … N-1
    norm   = counts / (len(masks) + 1e-8)              # 0 … 1
    fig, ax = plt.subplots(figsize=(5, 5))
    im  = ax.imshow(norm, cmap='hot', vmin=0, vmax=1)
    ax.set_title("Cumulative Hotspot Map", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Fraction of transitions changed")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Single before/after pair — original behaviour."""
    if "before" not in request.files or "after" not in request.files:
        return jsonify({"error": "Both images are required."}), 400
    try:
        thresh     = float(request.form.get("threshold",   THRESHOLD))
        min_reg    = int(float(request.form.get("min_region", MIN_REGION_SIZE)))

        b_pil = Image.open(request.files["before"].stream)
        a_pil = Image.open(request.files["after"].stream)

        b_r, b_np, b_t, b_sz = load_and_preprocess(b_pil)
        a_r, a_np, a_t, a_sz = load_and_preprocess(a_pil)

        cmap         = compute_change_map(encode(b_t), encode(a_t))
        cm, n_reg    = clean_mask(cmap, thresh, min_reg)

        return jsonify({
            "before_img":     pil_to_b64(b_r),
            "after_img":      pil_to_b64(a_r),
            "heatmap":        heatmap_b64(cmap),
            "overlay":        overlay_b64(a_np, cm),
            "stats": {
                "before_size":      list(b_sz),
                "after_size":       list(a_sz),
                "changed_pixels":   int(cm.sum()),
                "changed_area_pct": round(float(cm.mean()) * 100, 2),
                "num_regions":      n_reg,
                "threshold":        thresh,
                "device":           str(device),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-timeline", methods=["POST"])
def analyze_timeline():
    """
    Multi-image timeline analysis.

    Form fields:
      files[]         – ordered image files (client sends them in desired order)
      labels[]        – optional display labels (filenames by default)
      mode            – 'incremental' | 'cumulative'  (default cumulative)
      threshold       – float
      min_region      – int
    """
    files = request.files.getlist("files[]")
    if len(files) < 2:
        return jsonify({"error": "At least 2 images are required for timeline analysis."}), 400

    try:
        thresh   = float(request.form.get("threshold",   THRESHOLD))
        min_reg  = int(float(request.form.get("min_region", MIN_REGION_SIZE)))
        mode     = request.form.get("mode", "cumulative")   # 'cumulative' | 'incremental'
        labels   = request.form.getlist("labels[]")

        # ── Load & encode all images ──────────────────────────────────────────
        images   = []   # list of (resized_pil, img_np, orig_size)
        features = []   # list of feature tensors

        for i, f in enumerate(files):
            pil               = Image.open(f.stream)
            r, np_, t, sz     = load_and_preprocess(pil)
            images.append((r, np_, sz))
            features.append(encode(t))
            if i >= len(labels) or not labels[i]:
                labels_safe = labels[:]
                while len(labels_safe) <= i:
                    labels_safe.append(f.filename or f"Image {i+1}")
                labels = labels_safe

        n = len(images)

        # ── Build pairs based on mode ─────────────────────────────────────────
        if mode == "cumulative":
            pairs = [(0, i) for i in range(1, n)]         # all vs first
        else:  # incremental
            pairs = [(i, i + 1) for i in range(n - 1)]   # consecutive

        # ── Process each pair ─────────────────────────────────────────────────
        transitions   = []
        all_masks     = []
        changed_pcts  = []

        for a_idx, b_idx in pairs:
            cmap       = compute_change_map(features[a_idx], features[b_idx])
            cm, n_reg  = clean_mask(cmap, thresh, min_reg)
            pct        = round(float(cm.mean()) * 100, 2)

            a_r, a_np, a_sz = images[a_idx]
            b_r, b_np, b_sz = images[b_idx]

            transitions.append({
                "label_a":          labels[a_idx],
                "label_b":          labels[b_idx],
                "before_img":       pil_to_b64(a_r),
                "after_img":        pil_to_b64(b_r),
                "heatmap":          heatmap_b64(cmap),
                "overlay":          overlay_b64(b_np, cm),
                "changed_area_pct": pct,
                "changed_pixels":   int(cm.sum()),
                "num_regions":      n_reg,
            })
            all_masks.append(cm)
            changed_pcts.append(pct)

        # ── Change velocity (Δ% between successive transitions) ───────────────
        velocity = [0.0]   # first point has no predecessor
        for i in range(1, len(changed_pcts)):
            velocity.append(round(changed_pcts[i] - changed_pcts[i - 1], 2))

        # ── Cumulative hotspot map ────────────────────────────────────────────
        hotspot_b64 = cumulative_hotspot_b64(all_masks)

        # ── Image thumbnails for the strip ───────────────────────────────────
        thumbs = [pil_to_b64(img[0]) for img in images]

        return jsonify({
            "mode":          mode,
            "labels":        labels,
            "thumbs":        thumbs,
            "transitions":   transitions,
            "graph": {
                "labels":    [f"{labels[b]} " for _, b in pairs],
                "pcts":      changed_pcts,
                "velocity":  velocity,
            },
            "hotspot":       hotspot_b64,
            "stats": {
                "n_images":      n,
                "n_transitions": len(pairs),
                "peak_pct":      max(changed_pcts),
                "peak_label":    labels[pairs[changed_pcts.index(max(changed_pcts))][1]],
                "mean_pct":      round(float(np.mean(changed_pcts)), 2),
                "device":        str(device),
            },
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
