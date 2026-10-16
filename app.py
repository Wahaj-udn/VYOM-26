import os
import io
import re
import base64
import json
import smtplib
import numpy as np
from datetime import datetime, timezone
from email.message import EmailMessage
from xml.sax.saxutils import escape as _xml_escape

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

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, ListFlowable, ListItem, HRFlowable, PageBreak,
)

# ── Google GenAI SDK setup (Using Gemma Multimodal Model) ─────────────────────
from google import genai
from google.genai import types as genai_types

_GEMINI_KEY = os.getenv("GEMINI_KEY", os.getenv("GEMINI_API_KEY", ""))
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
        print("WARNING: GEMINI_API_KEY not set — AI analysis & report generation will be disabled.")

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
IMG_SIZE = 256
THRESHOLD = 0.55
MIN_REGION_SIZE = 30

# -- Email alert config --------------------------------------------------------
# NOTE: no hardcoded fallback credentials here on purpose — a previous version of
# this file shipped a real Gmail app password as the default value, which leaked
# it into the public repo. Set these in a local .env (see .env.example) instead.
ALERT_SENDER = os.getenv("ALERT_SENDER", "")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", "")
ALERT_PASSWORD = os.getenv("ALERT_PASSWORD", "")
ALERT_AREA_PCT = 40.0       # send alert when cumulative changed area > this %
ALERT_FEAT_THRESH = 0.45    # fixed threshold used for alert pixel count

EMAIL_CONFIGURED = bool(ALERT_SENDER and ALERT_RECEIVER and ALERT_PASSWORD)
if not EMAIL_CONFIGURED:
    print("NOTE: ALERT_SENDER / ALERT_RECEIVER / ALERT_PASSWORD are not fully set — "
          "automatic + manual email sending is disabled. Report generation (PDF) still works, "
          "so the report button runs in 'demo mode' until these are set in .env.")


def send_alert_email(n_images, n_transitions, peak_pct, mean_pct, cumul_pct):
    """Lightweight automatic alert (no attachment) — fired from /analyze-timeline
    when cumulative changed area crosses ALERT_AREA_PCT. Kept separate from the
    richer manual report email below so a routine timeline run stays fast."""
    if not EMAIL_CONFIGURED:
        print("Alert threshold crossed but email isn't configured — skipping auto-send.")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = "Satellite Change Detector - High Change Alert"
        msg["From"] = ALERT_SENDER
        msg["To"] = ALERT_RECEIVER
        msg.set_content(f"""
Satellite Image Change Detection - ALERT

A significant level of cumulative change has been detected in the monitored area.

Cumulative changed area (threshold {ALERT_FEAT_THRESH}): {cumul_pct:.1f}%
Peak single-transition change: {peak_pct:.1f}%
Mean change across all transitions: {mean_pct:.1f}%
Images in timeline: {n_images} | Transitions analysed: {n_transitions}

This level of change (>{ALERT_AREA_PCT:.0f}%) requires attention.
Please review the ChangeScope dashboard for heatmaps and AI analysis, or use
the "Generate & Send Report" button there for a full PDF write-up.
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
    return (9999, 99)  # unrecognised filenames go to end


# ── Core helpers ──────────────────────────────────────────────────────────────
def load_and_preprocess(pil_image, size=IMG_SIZE):
    img = pil_image.convert("RGB")
    orig_size = img.size
    img_r = img.resize((size, size), Image.Resampling.BILINEAR)
    img_np = np.array(img_r).astype(np.float32) / 255.0
    tensor = transforms.ToTensor()(img_r)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    tensor = normalize(tensor)
    return img_r, img_np, tensor.unsqueeze(0), orig_size


def encode(tensor):
    with torch.no_grad():
        return encoder(tensor.to(device))


def compute_change_map(feat_a, feat_b):
    diff = torch.abs(feat_a - feat_b)
    cmap = diff.mean(dim=1)
    cmap = F.interpolate(cmap.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE),
                          mode="bilinear", align_corners=False)
    cmap = cmap.squeeze().cpu().numpy()
    cmap = (cmap - cmap.min()) / (cmap.max() - cmap.min() + 1e-8)
    return cmap


def clean_mask(change_map, threshold, min_region):
    mask = change_map > threshold
    labeled, n = ndimage.label(mask)
    sizes = np.bincount(labeled.ravel())
    out = mask.copy()
    for cid in range(1, n + 1):
        if sizes[cid] < min_region:
            out[labeled == cid] = False
    return out, n


def heatmap_b64(change_map):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(change_map, cmap='jet', vmin=0, vmax=1)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Change intensity")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def overlay_b64(img_np, mask):
    orig = (img_np * 255).astype(np.uint8)
    red = orig.copy()
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
    stack = np.stack([m.astype(np.float32) for m in masks], axis=0)
    counts = stack.sum(axis=0)             # 0 … N-1
    norm = counts / (len(masks) + 1e-8)    # 0 … 1
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(norm, cmap='hot', vmin=0, vmax=1)
    ax.set_title("Cumulative Hotspot Map", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Fraction of transitions changed")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _b64_to_pil(b64str):
    raw = base64.b64decode(b64str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _esc(text):
    """Escape text before dropping it into a ReportLab Paragraph (which parses a
    small HTML-like markup) — Gemma's output is untrusted-ish free text and may
    contain '&', '<', '>'."""
    return _xml_escape(str(text or ""))


# ══════════════════════════════════════════════════════════════════════════
#  STRUCTURED AI ANALYSIS (one Gemma call, reused for the on-page "Deep
#  Analysis" card, the PDF report, and the report email body)
# ══════════════════════════════════════════════════════════════════════════

ANALYSIS_SCHEMA_HINT = """Respond with ONLY a single valid JSON object — no markdown
fences, no commentary before or after — matching EXACTLY this shape:

{
  "title": "short, specific report title (<= 12 words)",
  "executive_summary": "2-4 sentences, specific, no generic filler",
  "key_change_zones": [
    {"zone_name": "...", "change_type": "...", "description": "..."}
  ],
  "hotspot_analysis": "why the hot pixels are hot, tied to the visible imagery",
  "pattern_analysis": "localised vs widespread; radial, linear or clustered; scale",
  "likely_causes": ["...", "..."],
  "timeline_estimate": "inferred chronology / pacing of the change",
  "environmental_social_implications": "...",
  "recommendations": ["...", "..."],
  "confidence_note": "one sentence on inference limits (resolution, cloud cover, etc.)"
}

Use 2-5 items for every list field. Every string must be specific to what is visible —
never a generic placeholder."""


def build_analysis_prompt(stats):
    stats_text = (
        f"- Total images in timeline: {stats.get('n_images', '?')}\n"
        f"- Total transitions analysed: {stats.get('n_transitions', '?')}\n"
        f"- Peak changed area: {stats.get('peak_pct', '?')}% (at '{stats.get('peak_label', '?')}')\n"
        f"- Mean changed area across all transitions: {stats.get('mean_pct', '?')}%"
    )
    return f"""You are an expert geospatial and remote sensing analyst writing a report \
for other researchers who will act on it. You have been given three images:

1. FIRST IMAGE — the earliest captured image of an area.
2. LAST IMAGE — the most recent captured image of the same area.
3. CUMULATIVE HOTSPOT MAP — a false-colour heat map generated by a deep-learning \
change-detection model (ResNet18 feature encoder). Bright/white pixels changed \
across the most time steps; dark/black pixels were stable throughout the timeline.

Timeline statistics:
{stats_text}

Be SPECIFIC and DETAILED — "significant changes are visible" is not acceptable. \
Name features (structures, vegetation, water bodies, roads, bare soil, land-use \
zones), describe the nature of each change (construction, demolition, deforestation, \
flooding, drought, agricultural clearing, urban expansion, erosion, etc.), relate the \
hotspot map to what's visible, comment on whether the pattern is localised or \
widespread, infer likely causes and a rough timeline, and flag anything of \
environmental or human-activity concern.

{ANALYSIS_SCHEMA_HINT}"""


def parse_structured_analysis(raw_text):
    """Best-effort parse of Gemma's JSON response. Returns (data, is_structured)."""
    cleaned = (raw_text or "").strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        data = json.loads(cleaned)
        data.setdefault("title", "Change Detection Report")
        data.setdefault("executive_summary", "")
        data.setdefault("key_change_zones", [])
        data.setdefault("hotspot_analysis", "")
        data.setdefault("pattern_analysis", "")
        data.setdefault("likely_causes", [])
        data.setdefault("timeline_estimate", "")
        data.setdefault("environmental_social_implications", "")
        data.setdefault("recommendations", [])
        data.setdefault("confidence_note", "")
        return data, True
    except (json.JSONDecodeError, AttributeError, TypeError):
        # Fall back to treating the whole reply as the executive summary so the
        # UI and PDF still render something useful instead of erroring out.
        return {
            "title": "Change Detection Report", "executive_summary": raw_text or "",
            "key_change_zones": [], "hotspot_analysis": "", "pattern_analysis": "",
            "likely_causes": [], "timeline_estimate": "",
            "environmental_social_implications": "", "recommendations": [],
            "confidence_note": "",
        }, False


def run_gemma_analysis(img_first_b64, img_last_b64, img_hotspot_b64, stats):
    """Single Gemma call → structured dict. Raises RuntimeError if unconfigured."""
    if client is None:
        raise RuntimeError("Gemini/Gemma client is not configured on the server (check API key).")

    img_first = _b64_to_pil(img_first_b64)
    img_last = _b64_to_pil(img_last_b64)
    img_hotspot = _b64_to_pil(img_hotspot_b64)
    prompt = build_analysis_prompt(stats)

    try:
        config = genai_types.GenerateContentConfig(response_mime_type="application/json")
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[img_first, img_last, img_hotspot, prompt],
            config=config,
        )
    except TypeError:
        # Older SDK without structured-output config support — plain call, the
        # prompt's own JSON-only instruction is the fallback guarantee.
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[img_first, img_last, img_hotspot, prompt],
        )

    data, is_structured = parse_structured_analysis(response.text)
    return data, is_structured, response.text


# ══════════════════════════════════════════════════════════════════════════
#  PDF REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════

_styles = getSampleStyleSheet()
_styles.add(ParagraphStyle(name="RptTitle", fontSize=19, leading=23, fontName="Helvetica-Bold",
                            textColor=colors.HexColor("#1a2340"), spaceAfter=4))
_styles.add(ParagraphStyle(name="RptSub", fontSize=9.5, leading=13, fontName="Helvetica",
                            textColor=colors.HexColor("#546080"), spaceAfter=14))
_styles.add(ParagraphStyle(name="RptH2", fontSize=13, leading=16, fontName="Helvetica-Bold",
                            textColor=colors.HexColor("#5b8def"), spaceBefore=16, spaceAfter=6))
_styles.add(ParagraphStyle(name="RptH3", fontSize=10.5, leading=13, fontName="Helvetica-Bold",
                            textColor=colors.HexColor("#1a2340"), spaceBefore=8, spaceAfter=3))
_styles.add(ParagraphStyle(name="RptBody", fontSize=9.5, leading=14.5, fontName="Helvetica",
                            textColor=colors.HexColor("#222831")))
_styles.add(ParagraphStyle(name="RptCaption", fontSize=8, leading=10, fontName="Helvetica-Oblique",
                            textColor=colors.HexColor("#546080"), alignment=1, spaceBefore=3, spaceAfter=10))


def _rl_image_from_b64(b64str, width, height):
    raw = base64.b64decode(b64str)
    return RLImage(io.BytesIO(raw), width=width, height=height)


def _short_chart_label(label):
    """Shorten a (often long) source filename for use as a chart tick label.
    Prefers a YYYY-MM pulled from the filename; falls back to a short truncation."""
    label = (label or "").strip()
    m = re.search(r'(\d{4})_(\d{2})', label)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    name = re.sub(r'\.[^.]+$', '', label)  # strip extension
    return (name[:14] + "…") if len(name) > 14 else name


def _report_chart_png(graph):
    labels = [_short_chart_label(l) for l in (graph.get("labels", []) or [])]
    pcts = graph.get("pcts", []) or []
    velocity = graph.get("velocity", []) or []
    fig, ax1 = plt.subplots(figsize=(6.6, 2.8))
    ax1.plot(range(len(pcts)), pcts, marker="o", color="#5b8def", linewidth=2, label="Changed area (%)")
    ax1.set_ylabel("Changed area (%)", color="#5b8def", fontsize=9)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax1.tick_params(axis='y', labelcolor="#5b8def")
    ax1.grid(alpha=0.25)
    if velocity:
        ax2 = ax1.twinx()
        ax2.bar(range(len(velocity)), velocity, color="#ef5b8d", alpha=0.25, label="Velocity (Δ%)")
        ax2.set_ylabel("Velocity (Δ%)", color="#ef5b8d", fontsize=9)
        ax2.tick_params(axis='y', labelcolor="#ef5b8d")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_report_pdf(payload, analysis):
    """Builds the full researcher-facing PDF report (charts + imagery + the
    structured Gemma write-up). Returns raw PDF bytes."""
    stats = payload.get("stats", {}) or {}
    graph = payload.get("graph", {}) or {}
    hotspot_b64 = payload.get("hotspot", "")
    thumb_first = payload.get("thumb_first", "")
    thumb_last = payload.get("thumb_last", "")
    peak = payload.get("peak_transition")  # optional: heatmap/overlay/label_a/label_b/changed_area_pct

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm, bottomMargin=16 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm,
                             title=analysis.get("title", "ChangeScope Report"))
    story = []
    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    story.append(Paragraph(_esc(analysis.get("title")) or "ChangeScope — Change Detection Report",
                            _styles["RptTitle"]))
    story.append(Paragraph(
        f"Automated satellite/aerial change-detection report &middot; generated {generated} "
        f"&middot; device: {_esc(stats.get('device', '?'))}", _styles["RptSub"]))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#dfe6f3"), thickness=1, spaceAfter=10))

    story.append(Paragraph("Executive Summary", _styles["RptH2"]))
    story.append(Paragraph(_esc(analysis.get("executive_summary")) or "—", _styles["RptBody"]))

    story.append(Paragraph("Timeline Statistics", _styles["RptH2"]))
    rows = [
        ["Images analysed", str(stats.get("n_images", "?"))],
        ["Transitions analysed", str(stats.get("n_transitions", "?"))],
        ["Peak changed area", f"{stats.get('peak_pct', '?')}%  (at \"{stats.get('peak_label', '?')}\")"],
        ["Mean changed area", f"{stats.get('mean_pct', '?')}%"],
    ]
    t = Table(rows, colWidths=[55 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#546080")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5eaf3")),
    ]))
    story.append(t)

    story.append(Paragraph("Change Over Time", _styles["RptH2"]))
    try:
        chart_bytes = _report_chart_png(graph)
        story.append(RLImage(io.BytesIO(chart_bytes), width=165 * mm, height=70 * mm))
    except Exception:
        story.append(Paragraph("(chart unavailable)", _styles["RptCaption"]))

    if thumb_first and thumb_last:
        story.append(Paragraph("Earliest vs. Most Recent Imagery", _styles["RptH2"]))
        story.append(Table([[
            _rl_image_from_b64(thumb_first, 77 * mm, 58 * mm),
            _rl_image_from_b64(thumb_last, 77 * mm, 58 * mm),
        ]], colWidths=[82 * mm, 82 * mm]))
        story.append(Paragraph("Left: earliest image in timeline &middot; Right: most recent image",
                                _styles["RptCaption"]))

    if hotspot_b64:
        story.append(Paragraph("Cumulative Hotspot Map", _styles["RptH2"]))
        story.append(_rl_image_from_b64(hotspot_b64, 100 * mm, 100 * mm))
        story.append(Paragraph("Bright pixels changed across the most timeline transitions.",
                                _styles["RptCaption"]))

    if peak and peak.get("heatmap") and peak.get("overlay"):
        story.append(Paragraph(
            f"Peak Transition &mdash; {_esc(peak.get('label_a', '?'))} &rarr; {_esc(peak.get('label_b', '?'))} "
            f"({peak.get('changed_area_pct', '?')}% changed)", _styles["RptH2"]))
        story.append(Table([[
            _rl_image_from_b64(peak["heatmap"], 77 * mm, 58 * mm),
            _rl_image_from_b64(peak["overlay"], 77 * mm, 58 * mm),
        ]], colWidths=[82 * mm, 82 * mm]))
        story.append(Paragraph("Left: raw change-intensity heatmap &middot; Right: change regions overlaid",
                                _styles["RptCaption"]))

    story.append(PageBreak())

    zones = analysis.get("key_change_zones") or []
    if zones:
        story.append(Paragraph("Key Change Zones", _styles["RptH2"]))
        for z in zones:
            story.append(Paragraph(
                f'{_esc(z.get("zone_name", "Zone"))} <font color="#5b8def">&mdash; {_esc(z.get("change_type", ""))}</font>',
                _styles["RptH3"]))
            story.append(Paragraph(_esc(z.get("description", "")), _styles["RptBody"]))

    def section(title, text):
        if text:
            story.append(Paragraph(title, _styles["RptH2"]))
            story.append(Paragraph(_esc(text), _styles["RptBody"]))

    section("Hotspot Analysis", analysis.get("hotspot_analysis"))
    section("Pattern Analysis", analysis.get("pattern_analysis"))

    causes = analysis.get("likely_causes") or []
    if causes:
        story.append(Paragraph("Likely Causes", _styles["RptH2"]))
        story.append(ListFlowable(
            [ListItem(Paragraph(_esc(c), _styles["RptBody"])) for c in causes],
            bulletType="bullet", start="circle"))

    section("Timeline Estimate", analysis.get("timeline_estimate"))
    section("Environmental / Social Implications", analysis.get("environmental_social_implications"))

    recs = analysis.get("recommendations") or []
    if recs:
        story.append(Paragraph("Recommendations", _styles["RptH2"]))
        story.append(ListFlowable(
            [ListItem(Paragraph(_esc(r), _styles["RptBody"])) for r in recs],
            bulletType="bullet", start="circle"))

    if analysis.get("confidence_note"):
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", color=colors.HexColor("#dfe6f3"), thickness=0.75))
        story.append(Paragraph(f"<i>Confidence note: {_esc(analysis['confidence_note'])}</i>",
                                _styles["RptCaption"]))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated by ChangeScope — AI-assisted change detection. Verify findings against "
        "ground-truth imagery before acting on them.", _styles["RptCaption"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def compose_email_body(analysis, stats, filename):
    """Plain-text email body built from the same Gemma-generated analysis that
    fills the PDF, so the email isn't just a bare attachment."""
    causes = analysis.get("likely_causes") or []
    causes_txt = "\n".join(f"  - {c}" for c in causes[:3]) or "  (see attached report)"
    return f"""Satellite / Aerial Change Detection — Automated Report

{analysis.get('title', 'Change Detection Report')}

{(analysis.get('executive_summary') or '').strip()}

Key stats:
  - Images analysed: {stats.get('n_images', '?')}
  - Transitions analysed: {stats.get('n_transitions', '?')}
  - Peak changed area: {stats.get('peak_pct', '?')}% (at "{stats.get('peak_label', '?')}")
  - Mean changed area: {stats.get('mean_pct', '?')}%

Likely causes (top findings):
{causes_txt}

The full report — including before/after imagery, the change-over-time chart, the
cumulative hotspot map, and a detailed zone-by-zone analysis — is attached as
"{filename}".

— ChangeScope (automated message)
"""


def send_report_email(pdf_bytes, filename, subject, body_text):
    """Attempts to send the report PDF by email. Never raises — always returns a
    dict describing exactly what happened, so the caller can show the user an
    explicit sent/not-sent status instead of guessing."""
    if not EMAIL_CONFIGURED:
        return {
            "configured": False, "attempted": False, "sent": False,
            "recipient": ALERT_RECEIVER or None,
            "error": "Email isn't configured on the server (set ALERT_SENDER, "
                     "ALERT_RECEIVER and ALERT_PASSWORD in .env).",
        }
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = ALERT_SENDER
        msg["To"] = ALERT_RECEIVER
        msg.set_content(body_text)
        msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(ALERT_SENDER, ALERT_PASSWORD)
            smtp.send_message(msg)
        return {"configured": True, "attempted": True, "sent": True,
                "recipient": ALERT_RECEIVER, "error": None}
    except Exception as e:
        return {"configured": True, "attempted": True, "sent": False,
                "recipient": ALERT_RECEIVER, "error": str(e)}


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
        thresh = float(request.form.get("threshold", THRESHOLD))
        min_reg = int(float(request.form.get("min_region", MIN_REGION_SIZE)))

        b_pil = Image.open(request.files["before"].stream)
        a_pil = Image.open(request.files["after"].stream)

        b_r, b_np, b_t, b_sz = load_and_preprocess(b_pil)
        a_r, a_np, a_t, a_sz = load_and_preprocess(a_pil)

        cmap = compute_change_map(encode(b_t), encode(a_t))
        cm, n_reg = clean_mask(cmap, thresh, min_reg)

        return jsonify({
            "before_img": pil_to_b64(b_r),
            "after_img": pil_to_b64(a_r),
            "heatmap": heatmap_b64(cmap),
            "overlay": overlay_b64(a_np, cm),
            "stats": {
                "before_size": list(b_sz),
                "after_size": list(a_sz),
                "changed_pixels": int(cm.sum()),
                "changed_area_pct": round(float(cm.mean()) * 100, 2),
                "num_regions": n_reg,
                "threshold": thresh,
                "device": str(device),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-timeline", methods=["POST"])
def analyze_timeline():
    """
    Multi-image timeline analysis.
    Form fields:
      files[]      – ordered image files (client sends them in desired order)
      labels[]     – optional display labels (filenames by default)
      mode         – 'incremental' | 'cumulative' (default cumulative)
      threshold    – float
      min_region   – int
    """
    files = request.files.getlist("files[]")
    if len(files) < 2:
        return jsonify({"error": "At least 2 images are required for timeline analysis."}), 400

    try:
        thresh = float(request.form.get("threshold", THRESHOLD))
        min_reg = int(float(request.form.get("min_region", MIN_REGION_SIZE)))
        mode = request.form.get("mode", "cumulative")  # 'cumulative' | 'incremental'
        labels = request.form.getlist("labels[]")

        # ── Load & encode all images ──────────────────────────────────────────
        images = []     # list of (resized_pil, img_np, orig_size)
        features = []   # list of feature tensors

        for i, f in enumerate(files):
            pil = Image.open(f.stream)
            r, np_, t, sz = load_and_preprocess(pil)
            images.append((r, np_, sz))
            features.append(encode(t))
            if i >= len(labels) or not labels[i]:
                labels_safe = labels[:]
                while len(labels_safe) <= i:
                    labels_safe.append(f.filename or f"Image {i+1}")
                labels = labels_safe

        # Sort chronologically by YYYY_MM embedded in filename
        paired_sorted = sorted(
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
        transitions = []
        all_masks = []
        changed_pcts = []

        for a_idx, b_idx in pairs:
            cmap = compute_change_map(features[a_idx], features[b_idx])
            cm, n_reg = clean_mask(cmap, thresh, min_reg)
            pct = round(float(cm.mean()) * 100, 2)

            a_r, a_np, a_sz = images[a_idx]
            b_r, b_np, b_sz = images[b_idx]

            transitions.append({
                "label_a": labels[a_idx],
                "label_b": labels[b_idx],
                "before_img": pil_to_b64(a_r),
                "after_img": pil_to_b64(b_r),
                "heatmap": heatmap_b64(cmap),
                "overlay": overlay_b64(b_np, cm),
                "changed_area_pct": pct,
                "changed_pixels": int(cm.sum()),
                "num_regions": n_reg,
            })
            all_masks.append(cm)
            changed_pcts.append(pct)

        # ── Change velocity (Δ% between successive transitions) ───────────────
        velocity = [0.0]
        for i in range(1, len(changed_pcts)):
            velocity.append(round(changed_pcts[i] - changed_pcts[i - 1], 2))

        # ── Cumulative hotspot map ────────────────────────────────────────────
        hotspot_b64_val = cumulative_hotspot_b64(all_masks)

        # Alert: union of all cleaned masks — if >40% of area ever changed, send email
        cumul_union = np.any(np.stack(all_masks, axis=0), axis=0)
        cumul_pct_val = float(cumul_union.mean()) * 100

        alert_sent = False
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
            "mode": mode,
            "labels": labels,
            "thumbs": thumbs,
            "transitions": transitions,
            "graph": {
                "labels": [f"{labels[b]} " for _, b in pairs],
                "pcts": changed_pcts,
                "velocity": velocity,
            },
            "hotspot": hotspot_b64_val,
            "alert": {
                "triggered": alert_sent,
                "cumulative_pct": round(cumul_pct_val, 2),
                "limit": ALERT_AREA_PCT,
            },
            "stats": {
                "n_images": n,
                "n_transitions": len(pairs),
                "peak_pct": max(changed_pcts),
                "peak_label": labels[pairs[changed_pcts.index(max(changed_pcts))][1]],
                "mean_pct": round(float(np.mean(changed_pcts)), 2),
                "device": str(device),
            },
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/gemini-analyze", methods=["POST"])
def gemini_analyze():
    """
    Accepts:
      first_img – base64 PNG (first image in timeline)
      last_img  – base64 PNG (last image in timeline)
      hotspot   – base64 PNG (cumulative hotspot map)
      stats     – JSON object of timeline stats
    Returns:
      { "analysis": {...structured fields...}, "structured": bool, "raw": "<model text>" }
    """
    if client is None:
        return jsonify({"error": "Gemini client is not configured on the server (check API key)."}), 503
    try:
        data = request.get_json(force=True)
        first_b64 = data.get("first_img", "")
        last_b64 = data.get("last_img", "")
        hotspot_b64_in = data.get("hotspot", "")
        stats = data.get("stats", {})

        analysis, is_structured, raw_text = run_gemma_analysis(first_b64, last_b64, hotspot_b64_in, stats)
        return jsonify({"analysis": analysis, "structured": is_structured, "raw": raw_text})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/generate-report", methods=["POST"])
def generate_report():
    """
    On-demand: builds the full PDF report (charts, imagery, hotspot map, and
    the Gemma write-up) and attempts to email it. Always returns the PDF as
    base64 plus an explicit email status, so the report can be demonstrated
    even when SMTP isn't configured or delivery fails.

    Accepts JSON:
      stats, graph, hotspot, thumb_first, thumb_last  – from the timeline result
      peak_transition – optional {heatmap, overlay, label_a, label_b, changed_area_pct}
      analysis        – optional, reuse an already-generated structured analysis
                         (skips a second Gemma call if the user already ran
                         "Generate Deep Analysis")
    """
    try:
        payload = request.get_json(force=True)
        stats = payload.get("stats", {}) or {}
        analysis = payload.get("analysis")
        structured = True

        if not analysis:
            first_b64 = payload.get("thumb_first", "")
            last_b64 = payload.get("thumb_last", "")
            hotspot_b64_in = payload.get("hotspot", "")
            if not (first_b64 and last_b64 and hotspot_b64_in):
                return jsonify({"error": "Missing imagery to generate the report."}), 400
            analysis, structured, _ = run_gemma_analysis(first_b64, last_b64, hotspot_b64_in, stats)

        pdf_bytes = build_report_pdf(payload, analysis)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"ChangeScope_Report_{ts}.pdf"

        email_status = send_report_email(
            pdf_bytes, filename,
            subject=f"ChangeScope Report — {analysis.get('title', 'Change Detection')}",
            body_text=compose_email_body(analysis, stats, filename),
        )

        return jsonify({
            "analysis": analysis,
            "structured": structured,
            "report_pdf": base64.b64encode(pdf_bytes).decode(),
            "filename": filename,
            "email": email_status,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)