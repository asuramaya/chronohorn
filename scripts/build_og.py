#!/usr/bin/env python3
"""Generate site/assets/og.png — Naoto Ohshima dream sky aesthetic.

1200×630, gold/pink/violet sunset, Blinx-style time crystals, Fredoka heading.
Run from anywhere — paths resolve relative to the repo root.

Usage:
    pip install pillow
    python scripts/build_og.py
"""
from __future__ import annotations

import math
import random
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / ".cache" / "og-fonts"
OUTPUT = REPO / "site" / "assets" / "og.png"

FONTS = {
    "Fredoka-700.ttf": "https://github.com/google/fonts/raw/main/ofl/fredoka/Fredoka%5Bwdth%2Cwght%5D.ttf",
    "Inter-500.ttf": "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
}

W, H = 1200, 630

NIGHT_1 = (13, 8, 32)
NIGHT_2 = (26, 15, 58)
GOLD = (255, 209, 102)
PINK = (255, 126, 193)
VIOLET = (160, 123, 255)
VIOLET_GLOW = (200, 168, 255)
CYAN = (102, 227, 255)
CREAM = (250, 246, 255)
CREAM_DIM = (214, 207, 238)


def ensure_fonts() -> dict[str, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in FONTS.items():
        p = CACHE / name
        if not p.exists():
            print(f"  downloading {name}")
            urllib.request.urlretrieve(url, p)
        paths[name] = p
    return paths


def vertical_gradient(w, h, top, bot):
    img = Image.new("RGB", (w, h), top)
    px = img.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        c = tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3))
        for x in range(w):
            px[x, y] = c
    return img


def radial_glow(w, h, cx, cy, radius, color, alpha=180):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = img.load()
    for y in range(h):
        for x in range(w):
            d = math.hypot(x - cx, y - cy) / radius
            if d >= 1:
                continue
            a = int(alpha * (1 - d) ** 2)
            px[x, y] = (*color, a)
    return img


def crystal(size, rotation=0, opacity=255):
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    poly = [
        (s * 0.5, s * 0.05),
        (s * 0.875, s * 0.27),
        (s * 0.7, s * 0.95),
        (s * 0.3, s * 0.95),
        (s * 0.125, s * 0.27),
    ]
    grad = vertical_gradient(s, s, GOLD, PINK)
    grad2 = vertical_gradient(s, s, PINK, VIOLET)
    grad.paste(grad2, (0, s // 2))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).polygon(poly, fill=opacity)
    img.paste(grad, (0, 0), mask)
    top = [
        (s * 0.5, s * 0.05),
        (s * 0.875, s * 0.27),
        (s * 0.5, s * 0.36),
        (s * 0.125, s * 0.27),
    ]
    overlay = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(overlay).polygon(top, fill=(255, 255, 255, 90))
    img = Image.alpha_composite(img, overlay)
    spine = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(spine).line(
        [(s * 0.5, s * 0.05), (s * 0.5, s * 0.95)],
        fill=(255, 255, 255, 100),
        width=2,
    )
    img = Image.alpha_composite(img, spine)
    img = img.resize((size, size), Image.LANCZOS)
    if rotation:
        img = img.rotate(rotation, resample=Image.BICUBIC, expand=False)
    return img


def main() -> None:
    fonts = ensure_fonts()

    base = vertical_gradient(W, H, NIGHT_1, NIGHT_2).convert("RGBA")
    for cx, cy, r, color, alpha in [
        (W * 0.85, H * 0.25, 480, PINK, 120),
        (W * 0.05, H * 0.85, 520, VIOLET, 110),
        (W * 0.5, H * 0.05, 380, GOLD, 70),
    ]:
        base = Image.alpha_composite(base, radial_glow(W, H, cx, cy, r, color, alpha))

    rng = random.Random(7)
    star_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(star_layer)
    for _ in range(180):
        x, y = rng.randint(0, W), rng.randint(0, H)
        r = rng.choice([1, 1, 1, 2])
        c = rng.choice([CREAM, GOLD, VIOLET_GLOW, CYAN])
        sd.ellipse((x - r, y - r, x + r, y + r), fill=(*c, rng.randint(140, 220)))
    base = Image.alpha_composite(base, star_layer)

    for cs, x, y, rot, op in [
        (110, 1010, 80, 8, 230),
        (78, 80, 110, -10, 220),
        (54, 120, 470, 6, 200),
        (42, 1090, 510, -4, 180),
    ]:
        c = crystal(cs, rotation=rot, opacity=op)
        glow = c.filter(ImageFilter.GaussianBlur(radius=14))
        base.paste(glow, (x - 8, y - 8), glow)
        base.paste(c, (x, y), c)

    draw = ImageDraw.Draw(base)
    h_font = ImageFont.truetype(str(fonts["Fredoka-700.ttf"]), 92)
    sub_font = ImageFont.truetype(str(fonts["Fredoka-700.ttf"]), 36)
    body_font = ImageFont.truetype(str(fonts["Inter-500.ttf"]), 30)
    chip_font = ImageFont.truetype(str(fonts["Inter-500.ttf"]), 22)

    draw.text((76, 96), "✦  CHRONOHORN  ✦", font=chip_font, fill=GOLD)
    draw.text((76, 154), "Sweep the clock.", font=h_font, fill=CREAM)
    draw.text((76, 256), "Track the frontier.", font=h_font, fill=CREAM)
    draw.text(
        (76, 388),
        "Family-agnostic experiment tracker and architecture-search runtime",
        font=body_font,
        fill=CREAM_DIM,
    )
    draw.text(
        (76, 426),
        "for predictive descendants. SQLite truth · 64 MCP tools · CPU/Metal/CUDA.",
        font=body_font,
        fill=CREAM_DIM,
    )

    url = "chronohorn.com"
    bbox = draw.textbbox((0, 0), url, font=sub_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pill_w, pill_h = tw + 56, th + 28
    px, py = 76, 510
    pill = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
    ImageDraw.Draw(pill).rounded_rectangle(
        (0, 0, pill_w, pill_h), radius=pill_h // 2, fill=(255, 209, 102, 230)
    )
    base.paste(pill, (px, py), pill)
    draw.text((px + 28, py + 7), url, font=sub_font, fill=NIGHT_1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(OUTPUT, "PNG", optimize=True)
    print(f"wrote {OUTPUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
