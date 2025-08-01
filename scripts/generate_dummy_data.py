#!/usr/bin/env python
"""
generate_dummy_data.py  (PIL version)

Creates synthetic *tissue-like* image pairs:

    out_dir/
      ├─ unstained/   (grayscale → “Bone-like” colormap)
      └─ stained/     (pseudo H&E with pink cytoplasm, blue nuclei, stroma)

Usage
-----
python scripts/generate_dummy_data.py --out_dir dummy_vs_dataset --n 48 --size 256
"""
import argparse, math, random, pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# ──────────────────────────────────────────────────────────────────────────────
def bone_colormap(gray: np.ndarray) -> np.ndarray:
    """Map a [0,255] grayscale array to a Bone-like RGB colormap (PIL-friendly)."""
    # simple 2-point linear ramp (dark blue-gray → off-white)
    c0 = np.array([56,  56, 128], dtype=float)  # dark bluish gray
    c1 = np.array([245,245,255], dtype=float)   # nearly white
    t  = gray[..., None] / 255.0                # normalize 0–1
    return (c0*(1.0-t) + c1*t).astype(np.uint8)

def gen_pair(size=256, n_nuclei=120):
    """Return (unstained_PIL, stained_PIL) as Pillow RGB images."""
    # --- pseudo-H&E canvas ----------------------------------------------------
    img = Image.new("RGB", (size, size))
    base_rgb = (230, 180, 200)         # light pink (R,G,B)
    np_base  = np.tile(base_rgb, (size, size, 1)).astype(np.uint8)
    noise    = np.random.randint(-10, 11, size=(size, size, 3))
    np_img   = np.clip(np_base + noise, 0, 255).astype(np.uint8)
    stained  = Image.fromarray(np_img)

    draw = ImageDraw.Draw(stained)

    # --- stroma fibres -------------------------------------------------------
    for _ in range(int(size * 0.25)):
        x1, y1 = np.random.randint(0, size, 2)
        angle  = random.random() * 2 * math.pi
        length = np.random.randint(30, 60)
        x2 = int(np.clip(x1 + length * math.cos(angle), 0, size - 1))
        y2 = int(np.clip(y1 + length * math.sin(angle), 0, size - 1))
        draw.line([(x1, y1), (x2, y2)], fill=(200, 150, 200), width=1)

    # --- nuclei --------------------------------------------------------------
    for _ in range(n_nuclei):
        x, y  = np.random.randint(0, size, 2)
        r     = np.random.randint(5, 12)
        blue  = (
            random.randint(150, 200),   # R
            random.randint(70, 120),    # G
            random.randint(100, 140)    # B
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=blue, outline=None)

    # --- slight blur ---------------------------------------------------------
    stained = stained.filter(ImageFilter.GaussianBlur(radius=1))

    # --- unstained version (grayscale + bone colormap) -----------------------
    gray   = ImageOps.equalize(stained.convert("L"))   # enhance contrast
    gray_np = np.asarray(gray)
    bone_np = bone_colormap(gray_np)
    unstained = Image.fromarray(bone_np, mode="RGB")

    return unstained, stained

# ──────────────────────────────────────────────────────────────────────────────
def main(out_dir="dummy_vs_dataset", n=24, size=256):
    out_dir = pathlib.Path(out_dir)
    (out_dir / "unstained").mkdir(parents=True, exist_ok=True)
    (out_dir / "stained").mkdir(parents=True, exist_ok=True)

    for i in range(n):
        u, s = gen_pair(size=size)
        u.save(out_dir / "unstained" / f"{i:03d}.png")
        s.save(out_dir / "stained"   / f"{i:03d}.png")

    print(f"[✓] Saved {n} paired images under: {out_dir}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="dummy_vs_dataset")
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    main(args.out_dir, args.n, args.size)
