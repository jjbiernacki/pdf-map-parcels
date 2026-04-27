"""Eksploracja: skeletonize zielonych + endpoint detection.

Cel: zobaczyć czy domykanie wolnych końców prostymi odcinkami zamknie
"wyciekające" wielokąty (391, 310, 48/1, 49/1, 51/1, 283, 284).
"""
from __future__ import annotations

import numpy as np
import fitz
import cv2
from skimage.morphology import skeletonize
from scipy import ndimage as ndi

PDF = "03 PZT granice.pdf"
SCALE = 4  # 4 × 595×842 ≈ 2380×3370; dla mapy 20000×2500 pt → raster do 80000×10000
# PDF jest bardzo szeroki; zrenderuję cały naraz

GREEN = (0, 0.584, 0)
RED = (1, 0, 0)


def render_layer(page, scale, color_predicate, width_predicate=None):
    """Renderuje warstwę izolując stroki spełniające color_predicate."""
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return arr


def main():
    doc = fitz.open(PDF)
    page = doc[0]
    print(f"Page rect (pt): {page.rect}")
    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    print(f"Raster size: {arr.shape}")

    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    green_mask = (G > 90) & (R < 140) & (B < 140)
    red_mask = (R > 150) & (G < 90) & (B < 90)
    print(f"Green px: {green_mask.sum()}, Red px: {red_mask.sum()}")

    # skeletonize green
    gsk = skeletonize(green_mask)
    print(f"Green skel px: {gsk.sum()}")

    # Endpoint detection: piksel szkieletu z dokładnie 1 sąsiadem (3x3 conv)
    k = np.ones((3, 3), dtype=np.uint8)
    ncount = ndi.convolve(gsk.astype(np.uint8), k, mode="constant", cval=0)
    endpoints = gsk & (ncount == 2)  # centralny + 1 sąsiad
    print(f"Endpoints: {endpoints.sum()}")

    ys, xs = np.where(endpoints)
    print(f"Endpoint bbox: x={xs.min()}-{xs.max()}, y={ys.min()}-{ys.max()}")

    # Zapisz wizualizację
    vis = arr.copy()
    vis[gsk] = (0, 0, 255)  # szkielet na niebiesko
    for y, x in zip(ys, xs):
        cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 255), 2)  # magenta circles
    # zapisz pierwsze 5000px szerokości dla podglądu (inaczej plik duży)
    H, W = vis.shape[:2]
    crop = vis[:, :min(W, 5000)]
    cv2.imwrite("/tmp/explore_green_skel.png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print("Saved /tmp/explore_green_skel.png")


if __name__ == "__main__":
    main()
