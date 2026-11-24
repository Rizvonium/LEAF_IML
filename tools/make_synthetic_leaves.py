"""Generate a SYNTHETIC stand-in for leaves.zip, for testing the pipeline only.

The images are drawn shapes, not real leaves, and nothing trained on them is
used for the assignment. The only point is to check that the notebook runs
from top to bottom before the real data is available.

Layout produced (a guess at the real layout; the notebook detects it):
    leaves/train/<species>/<id>.png   40 per species, 15 species
    leaves/test/<id>.png              35 per species, labels withheld

Usage: python tools/make_synthetic_leaves.py OUT_ZIP [--size 320]
"""
import argparse
import io
import math
import random
import zipfile

from PIL import Image, ImageDraw, ImageFilter

SPECIES = [f"species_{i:02d}" for i in range(15)]


def leaf_polygon(cls, rng, n=360):
    # each class gets its own aspect ratio, lobe count, serration and tip shape
    r = random.Random(cls * 7919)
    aspect = 1.0 + 2.5 * r.random()
    lobes = r.choice([0, 0, 3, 5, 7])
    lobe_amp = 0.0 if lobes == 0 else 0.15 + 0.2 * r.random()
    serr = r.choice([0, 0, 24, 40, 60])
    serr_amp = 0.0 if serr == 0 else 0.02 + 0.03 * r.random()
    tip = 0.3 * r.random()
    pts = []
    for k in range(n):
        t = 2 * math.pi * k / n
        rad = 1.0 + lobe_amp * math.cos(lobes * t) + serr_amp * math.cos(serr * t)
        rad *= 1.0 + tip * math.sin(t) + rng.gauss(0, 0.005)
        pts.append((rad * math.cos(t), rad * math.sin(t) * aspect))
    return pts, r


def draw_leaf(cls, rng, size):
    pts, r = leaf_polygon(cls, rng)
    base_green = (40 + int(60 * r.random()), 90 + int(80 * r.random()), 30 + int(40 * r.random()))
    img = Image.new("RGB", (size, size), (245, 245, 242))
    d = ImageDraw.Draw(img)
    scale = size * rng.uniform(0.18, 0.26) / max(1.0, max(abs(p[1]) for p in pts) / 1.6)
    ang = rng.uniform(-0.35, 0.35)  # scans are roughly upright
    cx, cy = size / 2 + rng.uniform(-10, 10), size / 2 + rng.uniform(-10, 10)

    def tf(x, y):
        x, y = x * scale, y * scale
        return (cx + x * math.cos(ang) - y * math.sin(ang), cy + x * math.sin(ang) + y * math.cos(ang))

    poly = [tf(x, y) for x, y in pts]
    col = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in base_green)
    d.polygon(poly, fill=col)
    ymax = max(abs(p[1]) for p in pts)
    d.line([tf(0, ymax * 0.95), tf(0, ymax * 1.35)], fill=tuple(c // 2 for c in col), width=max(2, size // 120))
    d.line([tf(0, -ymax * 0.9), tf(0, ymax * 0.95)], fill=tuple(min(255, c + 40) for c in col), width=1)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_zip")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=35)
    a = ap.parse_args()
    rng = random.Random(0)
    test_items = []
    with zipfile.ZipFile(a.out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for c, sp in enumerate(SPECIES):
            for i in range(a.n_train):
                buf = io.BytesIO()
                draw_leaf(c, rng, a.size).save(buf, "PNG")
                z.writestr(f"leaves/train/{sp}/{sp}_{i:03d}.png", buf.getvalue())
            for i in range(a.n_test):
                test_items.append(c)
        rng.shuffle(test_items)
        for j, c in enumerate(test_items):
            buf = io.BytesIO()
            draw_leaf(c, rng, a.size).save(buf, "PNG")
            z.writestr(f"leaves/test/test_{j:04d}.png", buf.getvalue())
    print(f"wrote {a.out_zip}: {len(SPECIES) * a.n_train} train, {len(test_items)} test")


if __name__ == "__main__":
    main()
