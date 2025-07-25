import os
import argparse
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# ------------------------------------------
# Optional TensorFlow FGSM support
# ------------------------------------------
def try_import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception:
        return None

# ------------------------------------------
# Clean synthetic image generator
# ------------------------------------------
def gen_clean_image(size=(224, 224)):
    """Generate a synthetic 'clean' RGB image with random colored shapes."""
    base_color = tuple(np.random.randint(0, 256, size=3).tolist())
    img = Image.new("RGB", size, color=base_color)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(2, 5)):
        x1, y1 = random.randint(0, size[0] - 60), random.randint(0, size[1] - 60)
        x2, y2 = x1 + random.randint(20, 100), y1 + random.randint(20, 100)
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        if random.random() < 0.5:
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)
    return img

# ------------------------------------------
# Adversarial transforms (synthetic)
# ------------------------------------------
def adv_noise(img, sigma=25):
    arr = np.array(img).astype("float32")
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(noisy)

def adv_blur(img, radius=3):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def adv_rotate(img, max_deg=25):
    deg = random.uniform(-max_deg, max_deg)
    return img.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

def adv_patch(img, patch_size=(60, 60)):
    w, h = img.size
    pw = random.randint(20, patch_size[0])
    ph = random.randint(20, patch_size[1])
    x1 = random.randint(0, w - pw)
    y1 = random.randint(0, h - ph)
    color = tuple(np.random.randint(0, 256, size=3).tolist())
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    draw.rectangle([x1, y1, x1 + pw, y1 + ph], fill=color)
    return img2

def adv_jpeg(img, quality=15):
    # Save to buffer then reload
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# ------------------------------------------
# FGSM-like perturbation (optional)
# ------------------------------------------
def adv_fgsm(img, tf_module, eps=0.01):
    """
    Apply a single-step FGSM-like perturbation using MobileNetV2 pretrained on ImageNet.
    If something goes wrong, return original image.
    """
    try:
        tf = tf_module
        from tensorflow.keras.applications import mobilenet_v2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

        model = mobilenet_v2.MobileNetV2(weights="imagenet", include_top=True)
        img_resized = img.resize((224, 224))
        x = np.array(img_resized).astype("float32")
        x_exp = np.expand_dims(x, axis=0)
        x_pre = preprocess_input(x_exp.copy())  # model-specific preprocessing

        with tf.GradientTape() as tape:
            inp = tf.convert_to_tensor(x_pre)
            tape.watch(inp)
            preds = model(inp)
            target = tf.argmax(preds[0])
            loss = tf.keras.losses.sparse_categorical_crossentropy(target, preds)

        grad = tape.gradient(loss, inp)
        signed_grad = tf.sign(grad)
        adv = inp + eps * signed_grad
        adv = tf.clip_by_value(adv, -1.0, 1.0)  # mobilenet_v2 preprocess range [-1,1]

        # de-preprocess: convert back to uint8 RGB
        adv_np = adv.numpy()[0]
        # mobilenet_v2 preprocess_input: x/127.5-1 => invert:
        adv_img_arr = ((adv_np + 1.0) * 127.5).clip(0, 255).astype("uint8")
        adv_pil = Image.fromarray(adv_img_arr).resize(img.size)
        return adv_pil
    except Exception:
        return img  # fallback

# ------------------------------------------
# Dataset generation
# ------------------------------------------
def build_dataset(
    out_dir,
    num_clean=500,
    include_fgsm=False,
    merge_adv=True,
    seed=123,
):
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = out_dir / "clean"
    clean_dir.mkdir(exist_ok=True)

    # Adversarial variant directories (temporary or final)
    adv_dirs = {
        "adv_noise": out_dir / "adv_noise",
        "adv_blur": out_dir / "adv_blur",
        "adv_rotate": out_dir / "adv_rotate",
        "adv_patch": out_dir / "adv_patch",
        "adv_jpeg": out_dir / "adv_jpeg",
    }

    if include_fgsm:
        adv_dirs["adv_fgsm"] = out_dir / "adv_fgsm"

    for d in adv_dirs.values():
        d.mkdir(exist_ok=True)

    tf_mod = try_import_tf() if include_fgsm else None

    # Generate clean & adversarial
    for i in range(num_clean):
        img = gen_clean_image()

        clean_path = clean_dir / f"clean_{i:04d}.jpg"
        img.save(clean_path, format="JPEG")

        # Create adversarial variants
        adv_noise(img).save(adv_dirs["adv_noise"] / f"adv_noise_{i:04d}.jpg", format="JPEG")
        adv_blur(img).save(adv_dirs["adv_blur"] / f"adv_blur_{i:04d}.jpg", format="JPEG")
        adv_rotate(img).save(adv_dirs["adv_rotate"] / f"adv_rotate_{i:04d}.jpg", format="JPEG")
        adv_patch(img).save(adv_dirs["adv_patch"] / f"adv_patch_{i:04d}.jpg", format="JPEG")
        adv_jpeg(img).save(adv_dirs["adv_jpeg"] / f"adv_jpeg_{i:04d}.jpg", format="JPEG")

        if include_fgsm and tf_mod is not None:
            adv_fgsm(img, tf_mod).save(adv_dirs["adv_fgsm"] / f"adv_fgsm_{i:04d}.jpg", format="JPEG")

    # Merge all adversarial variants into single folder (for binary clean vs adversarial training)
    if merge_adv:
        merged_adv_dir = out_dir / "adversarial"
        merged_adv_dir.mkdir(exist_ok=True)
        for k, d in adv_dirs.items():
            for p in d.glob("*.jpg"):
                target = merged_adv_dir / p.name
                p.replace(target)  # move
            d.rmdir()  # remove empty folder
        print(f"[INFO] Merged adversarial images into: {merged_adv_dir}")

    print(f"[DONE] Dataset generated at: {out_dir}")
    return out_dir

# ------------------------------------------
# CLI
# ------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate synthetic clean + adversarial image dataset.")
    ap.add_argument("--out", type=str, default="adversarial_image_dataset_ext", help="Output dataset folder.")
    ap.add_argument("--num", type=int, default=500, help="Number of clean images to generate (adversarial created per clean).")
    ap.add_argument("--fgsm", action="store_true", help="Add FGSM-like gradient perturbations (requires TensorFlow + internet for weights).")
    ap.add_argument("--no-merge-adv", action="store_true", help="Keep separate adversarial technique folders (multi-class).")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    return ap.parse_args()

def main():
    args = parse_args()
    merge_adv = not args.no_merge_adv
    build_dataset(
        out_dir=args.out,
        num_clean=args.num,
        include_fgsm=args.fgsm,
        merge_adv=merge_adv,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
