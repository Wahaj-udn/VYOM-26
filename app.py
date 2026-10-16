import os
import io
import re
import base64
import json
import smtplib
import numpy as np
from email.message import EmailMessage
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from PIL import Image, ExifTags
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# ── Google GenAI SDK setup (Using Gemma Multimodal Model) ─────────────────────
from google import genai

_GEMINI_KEY   = os.getenv("GEMINI_KEY", os.getenv("GEMINI_API_KEY", ""))
# Set to the multimodal Gemma model accessible via the API
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")

client = None
if _GEMINI_KEY:
    client = genai.Client(api_key=_GEMINI_KEY)
    print(f"Gemini client configured with multimodal model: {_GEMINI_MODEL}")
else:
    try:
        client = genai.Client()
        print(f"Gemini client configured via environment with model: {_GEMINI_MODEL}")
    except Exception:
        print("WARNING: GEMINI_API_KEY not set — /gemini-analyze will be disabled.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB for many images

# ── Model (loaded once) ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
encoder = nn.Sequential(*(list(resnet.children())[:-2]))
encoder = encoder.to(device)
encoder.eval()
print("Model loaded.")

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE        = 256
THRESHOLD       = 0.55
MIN_REGION_SIZE = 30

# -- Email alert config --------------------------------------------------------
ALERT_SENDER      = os.getenv("ALERT_SENDER",   "vikasmalluri@gmail.com")
ALERT_RECEIVER    = os.getenv("ALERT_RECEIVER", "mdwahajuddin193@gmail.com")
ALERT_PASSWORD    = os.getenv("ALERT_PASSWORD", "rbal dlnu dutn zaaf")
ALERT_AREA_PCT    = 40.0   # send alert when cumulative changed area > this %
ALERT_FEAT_THRESH = 0.45   # fixed threshold used for alert pixel count


def send_alert_email(n_images, n_transitions, peak_pct, mean_pct, cumul_pct):
    """Send SMTP alert when cumulative changed area exceeds ALERT_AREA_PCT."""
    try:
        msg = EmailMessage()
        msg["Subject"] = "Satellite Change Detector - High Change Alert"
        msg["From"]    = ALERT_SENDER
        msg["To"]      = ALERT_RECEIVER
        msg.set_content(f"""
Satellite Image Change Detection - ALERT

A significant level of cumulative change has been detected in the monitored area.

Cumulative changed area (threshold {ALERT_FEAT_THRESH}): {cumul_pct:.1f}%
Peak single-transition change:                           {peak_pct:.1f}%
Mean change across all transitions:                      {mean_pct:.1f}%
Images in timeline: {n_images}  |  Transitions analysed: {n_transitions}

This level of change (>{ALERT_AREA_PCT:.0f}%) requires attention.
Please review the ChangeScope dashboard for heatmaps and AI analysis.
""")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(ALERT_SENDER, ALERT_PASSWORD)
            smtp.send_message(msg)
        print("Alert email sent successfully.")
        return True
    except Exception as e:
        print(f"Alert email failed: {e}")
        return False


def filename_date_key(label: str):
    """Sort key: extract (year, month) from global_monthly_YYYY_MM_mosaic_... filenames.
    Falls back to alphabetical sort when no YYYY_MM pattern found."""
    m = re.search(r'_(\d{4})_(\d{2})_', label)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (9999, 99)   # unrecognised filenames go to end


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
    mask                 = change_map > threshold
    labeled, n           = ndimage.label(mask)
    sizes                = np.bincount(labeled.ravel())
    out                  = mask.copy()
    for cid in range(1, n + 1):
        if sizes[cid] < min_region:
            out[labeled == cid] = False
    return out, n


def heatmap_b64(change_map):
    fig, ax = plt.subplots(figsize=(5, 5))
    im      = ax.imshow(change_map, cmap='jet', vmin=0, vmax=1)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Change intensity")
    buf       = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def overlay_b64(img_np, mask):
    orig      = (img_np * 255).astype(np.uint8)
    red       = orig.copy()
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
    counts = stack.sum(axis=0)                   # 0 … N-1
    norm   = counts / (len(masks) + 1e-8)               # 0 … 1
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
            "before_img":    pil_to_b64(b_r),
            "after_img":     pil_to_b64(a_r),
            "heatmap":       heatmap_b64(cmap),
            "overlay":       overlay_b64(a_np, cm),
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
      files[]        – ordered image files (client sends them in desired order)
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
            pil                 = Image.open(f.stream)
            r, np_, t, sz     = load_and_preprocess(pil)
            images.append((r, np_, sz))
            features.append(encode(t))
            if i >= len(labels) or not labels[i]:
                labels_safe = labels[:]
                while len(labels_safe) <= i:
                    labels_safe.append(f.filename or f"Image {i+1}")
                labels = labels_safe

        n = len(images)

        # Sort chronologically by YYYY_MM embedded in filename
        paired_sorted  = sorted(
            zip(files, labels, images, features),
            key=lambda x: filename_date_key(x[1])
        )
        files, labels, images, features = map(list, zip(*paired_sorted))
        print("Sorted timeline order:", labels)
        n = len(images)

        # Build pairs based on mode
        if mode == "cumulative":
            pairs = [(0, i) for i in range(1, n)]
        else:  # incremental
            pairs = [(i, i + 1) for i in range(n - 1)]

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

        # Alert: union of all cleaned masks — if >40% of area ever changed, send email
        cumul_union   = np.any(np.stack(all_masks, axis=0), axis=0)
        cumul_pct_val = float(cumul_union.mean()) * 100
        alert_sent    = False
        if cumul_pct_val > ALERT_AREA_PCT:
            alert_sent = send_alert_email(
                n_images=n,
                n_transitions=len(pairs),
                peak_pct=max(changed_pcts),
                mean_pct=round(float(np.mean(changed_pcts)), 2),
                cumul_pct=cumul_pct_val,
            )

        thumbs = [pil_to_b64(img[0]) for img in images]

        return jsonify({
            "mode":        mode,
            "labels":      labels,
            "thumbs":      thumbs,
            "transitions": transitions,
            "graph": {
                "labels":   [f"{labels[b]} " for _, b in pairs],
                "pcts":     changed_pcts,
                "velocity": velocity,
            },
            "hotspot": hotspot_b64,
            "alert": {
                "triggered":      alert_sent,
                "cumulative_pct": round(cumul_pct_val, 2),
                "limit":          ALERT_AREA_PCT,
            },
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


@app.route("/gemini-analyze", methods=["POST"])
def gemini_analyze():
    """
    Accepts:
      first_img   – base64 PNG  (first image in timeline)
      last_img    – base64 PNG  (last image in timeline)
      hotspot     – base64 PNG  (cumulative hotspot map)
      stats       – JSON string of timeline stats

    Returns:
      { "analysis": "<markdown text>" }
    """
    if client is None:
        return jsonify({"error": "Gemini client is not configured on the server (check API key)."}), 503

    try:
        data       = request.get_json(force=True)
        first_b64  = data.get("first_img",  "")
        last_b64   = data.get("last_img",   "")
        hotspot_b64= data.get("hotspot",    "")
        stats      = data.get("stats",      {})

        def b64_to_pil(b64str):
            raw = base64.b64decode(b64str)
            return Image.open(io.BytesIO(raw)).convert("RGB")

        img_first   = b64_to_pil(first_b64)
        img_last    = b64_to_pil(last_b64)
        img_hotspot = b64_to_pil(hotspot_b64)

        # Build a stats summary for the prompt
        stats_text = (
            f"- Total images in timeline: {stats.get('n_images', '?')}\n"
            f"- Total transitions analysed: {stats.get('n_transitions', '?')}\n"
            f"- Peak changed area: {stats.get('peak_pct', '?')}% (at '{stats.get('peak_label', '?')}')\n"
            f"- Mean changed area across all transitions: {stats.get('mean_pct', '?')}%"
        )

        prompt = f"""You are an expert geospatial and remote sensing analyst. \
You have been given three images to analyse:

1. FIRST IMAGE — the earliest captured image of an area.
2. LAST IMAGE  — the most recent captured image of the same area.
3. CUMULATIVE HOTSPOT MAP — a false-colour heat map generated by a deep-learning \
change-detection model (ResNet18 feature encoder). Bright/white pixels changed \
across the most time steps; dark/black pixels were stable throughout the timeline.

Timeline statistics:
{stats_text}

Your task is to provide a SPECIFIC, DETAILED, and INSIGHTFUL written analysis of \
what has physically been happening in this area over the recorded period.

Guidelines:
- DO NOT produce generic statements such as "significant changes are visible" — 
be precise about WHERE and WHAT.
- Identify specific features: structures, vegetation patches, water bodies, roads, 
bare soil, shadows, land-use zones.
- Describe the NATURE of each change (construction, demolition, deforestation, 
flooding, drought, agricultural clearing, urban expansion, erosion, etc.).
- Relate the hotspot map to the visible changes — explain WHY those pixels appear 
hot (most frequently changed).
- Comment on the PATTERN of change: is it localised or widespread? Radial, linear, 
or clustered?
- Infer likely CAUSES and TIMELINE of the changes from the visual evidence.
- Flag anything that could indicate environmental concern, human activity, or 
natural processes.
- Conclude with 2–3 concise ACTIONABLE INSIGHTS or monitoring recommendations.

Structure your response with clear headings: \
**Overview**, **Key Change Zones**, **Hotspot Analysis**, **Likely Causes**, \
**Environmental / Social Implications**, **Recommendations**."""

        # Call the multimodal Gemma model via the new SDK syntax
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[img_first, img_last, img_hotspot, prompt]
        )

        return jsonify({"analysis": response.text})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)