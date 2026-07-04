#!/usr/bin/env python3
"""
Demo 1 — Feature detection & matching (first SfM stage).

Loads consecutive image pairs from config.yaml, runs the selected detector,
and writes side-by-side match plots to output_dir.

Usage (from repo root):
    cp config.yaml.example config.yaml   # set images_dir
    pip install -r requirements.txt
    python demos/demo_keypoints.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config() -> dict:
    config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}. Copy config.yaml.example to config.yaml."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_images(directory: Path, max_images: int) -> list[Path]:
    paths = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images and max_images > 0:
        paths = paths[:max_images]
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in {directory}, found {len(paths)}")
    return paths


def create_detector(name: str):
    name = name.upper()
    if name == "SIFT":
        return cv2.SIFT_create()
    if name == "ORB":
        return cv2.ORB_create(nfeatures=4000)
    if name == "AKAZE":
        return cv2.AKAZE_create()
    raise ValueError(f"Unsupported detector: {name}")


def main() -> None:
    cfg = load_config()
    images_dir = Path(cfg["images_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list_images(images_dir, int(cfg.get("max_images", 0)))
    detector = create_detector(cfg.get("feature_detector", "SIFT"))
    ratio = float(cfg.get("match_ratio", 0.75))
    ransac_thresh = float(cfg.get("ransac_threshold", 5.0))

    print(f"Found {len(paths)} images in {images_dir}")
    print(f"Detector: {cfg.get('feature_detector', 'SIFT')}")

    bf = cv2.BFMatcher()

    for i in range(len(paths) - 1):
        img1 = cv2.imread(str(paths[i]))
        img2 = cv2.imread(str(paths[i + 1]))
        if img1 is None or img2 is None:
            print(f"Skip unreadable pair: {paths[i].name}, {paths[i + 1].name}")
            continue

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        if des1 is None or des2 is None:
            print(f"No descriptors for pair {i}")
            continue

        raw = bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)

        if len(good) >= 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_thresh)
            if mask is not None:
                good = [m for m, keep in zip(good, mask.ravel()) if keep]

        print(f"Pair {i}: {len(kp1)} / {len(kp2)} keypoints, {len(good)} inlier matches")

        if cfg.get("save_match_plots", True):
            vis = cv2.drawMatches(
                img1, kp1, img2, kp2, good, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            out_path = output_dir / f"matches_{i:03d}_{i + 1:03d}.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"  → saved {out_path}")


if __name__ == "__main__":
    main()
