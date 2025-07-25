import io
import os
import imghdr
# Flask imports moved to main and route function for lazy loading
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

# app will be created in main block

# ---- Tunable thresholds ----
MAX_EXIF_BYTES = 10000          # Large EXIF may hide payloads
NOISE_MEAN_THRESH = 0.08        # Avg absolute diff vs blur
NOISE_MAX_THRESH  = 0.35        # Strong pixel perturbation
MIN_IMG_SIDE      = 32          # Unusually tiny images flagged
MAX_IMG_SIDE      = 8000        # Sanity upper bound

def _estimate_noise(arr):
    """Rudimentary noise score: mean abs diff vs blurred."""
    # arr is float32 [0,1], shape (H,W,C)
    # Quick blur using PIL-like conv via mean over 3x3 window in numpy
    import scipy.ndimage as ndi  # if not installed, replace w/ fallback below
    blurred = ndi.uniform_filter(arr, size=(3,3,1), mode='reflect')
    diff = np.abs(arr - blurred)
    return float(diff.mean()), float(diff.max())

def _estimate_noise_fallback(img):
    """Fallback blur using PIL if SciPy unavailable."""
    from PIL import ImageFilter
    blurred_img = img.filter(ImageFilter.BoxBlur(radius=1))
    a = np.asarray(img, dtype=np.float32) / 255.0
    b = np.asarray(blurred_img, dtype=np.float32) / 255.0
    diff = np.abs(a - b)
    return float(diff.mean()), float(diff.max())

def analyze_image(file_bytes, filename="uploaded"):
    # Basic container for response
    resp = {
        "is_attack": False,
        "score": 0.0,
        "reason": "OK",
        "details": {}
    }

    # ---- File type sniff ----
    sig_type = imghdr.what(None, h=file_bytes)
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    resp["details"]["file_sig"] = sig_type
    resp["details"]["file_ext"] = ext
    if sig_type is None:
        resp.update(is_attack=True, score=1.0, reason="Not a valid image file.")
        return resp
    if ext and sig_type and ext not in (sig_type, "jpg" if sig_type == "jpeg" else sig_type):
        resp["details"]["ext_mismatch"] = True
        resp["is_attack"] = True
        resp["reason"] = "Extension/file-type mismatch."
        resp["score"] = 0.6  # mild raise

    # ---- Load image via PIL ----
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
    except UnidentifiedImageError:
        resp.update(is_attack=True, score=1.0, reason="Unreadable image data.")
        return resp

    # ---- EXIF length check ----
    exif_bytes = img.info.get("exif", b"")
    exif_len = len(exif_bytes) if isinstance(exif_bytes, (bytes, bytearray)) else 0
    resp["details"]["exif_len"] = exif_len
    if exif_len > MAX_EXIF_BYTES:
        resp["is_attack"] = True
        resp["score"] = max(resp["score"], 0.75)
        resp["reason"] = "Excessive EXIF metadata."

    # ---- Size sanity ----
    w, h = img.size
    resp["details"]["width"] = w
    resp["details"]["height"] = h
    if min(w, h) < MIN_IMG_SIDE or max(w, h) > MAX_IMG_SIDE:
        resp["is_attack"] = True
        resp["score"] = max(resp["score"], 0.8)
        resp["reason"] = "Suspicious image dimensions."

    # ---- Noise heuristic ----
    arr = np.asarray(img, dtype=np.float32) / 255.0
    try:
        noise_mean, noise_max = _estimate_noise(arr)
    except Exception:
        noise_mean, noise_max = _estimate_noise_fallback(img)
    resp["details"]["noise_mean"] = noise_mean
    resp["details"]["noise_max"] = noise_max

    if noise_mean > NOISE_MEAN_THRESH or noise_max > NOISE_MAX_THRESH:
        resp["is_attack"] = True
        resp["score"] = max(resp["score"], 0.9)
        resp["reason"] = "High perturbation energy (possible adversarial noise)."

    # If still not flagged:
    if not resp["is_attack"]:
        resp["score"] = noise_mean  # low = safe
        resp["reason"] = "Image appears safe."

    return resp


def scan_image():
    """
    POST multipart/form-data with 'file' = image.
    Returns JSON {is_attack, score, reason, details}
    """
    from flask import request, jsonify
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    file_bytes = f.read()
    result = analyze_image(file_bytes, filename=f.filename)
    return jsonify(result), 200


if __name__ == "__main__":
    # Run locally
    # pip install flask pillow numpy scipy
    from flask import Flask
    app = Flask(__name__)
    app.route("/scan-image", methods=["POST"])(scan_image)
    app.run(host="0.0.0.0", port=5001, debug=False)
