"""A small, explicit HOG baseline for the Lecture 10 comparison.

This is a documented HOG variant, not a reimplementation claimed to reproduce
Dalal and Triggs exactly. Cells do not interpolate votes across cell boundaries.
Angles interpolate between two unsigned orientation bins. Blocks use L2 norm.
No parameter here is fitted to the dataset.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps
from .core import ROOT, normalize, softmax
from .images import image_views, load_image

PARAMETERS = {
    "image_size": [224, 224],
    "grayscale": "PIL RGB to L (8-bit luminance)",
    "gradient": "central difference, zero derivative at image boundaries",
    "orientations": 9,
    "angle_range_degrees": [0, 180],
    "pixels_per_cell": [16, 16],
    "cells_per_block": [2, 2],
    "block_stride_cells": 1,
    "orientation_interpolation": "linear between circular neighboring bins",
    "spatial_interpolation": False,
    "block_normalization": "L2, epsilon=1e-5",
    "final_normalization": "L2 per image, epsilon floor=1e-12",
    "dimension": 6084,
}


def prepare_hog(image):
    """Use the same global-view geometry as the CNN; then discard color."""
    global_view, _ = image_views(image)
    square = ImageOps.fit(global_view, (224, 224), method=Image.Resampling.BILINEAR)
    return np.asarray(square.convert("L"), dtype=np.float32) / 255.0


def gradients(gray):
    """Return magnitude and unsigned angle for each grayscale pixel."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    return magnitude, angle


def cell_histograms(magnitude, angle, cell_size=16, bins=9):
    """Accumulate magnitude, not pixel frequency, into orientation bins.

    For a direction between two bin centers, split its magnitude in proportion
    to distance. The final bin wraps around to zero because angles are unsigned.
    """
    height, width = magnitude.shape
    if height % cell_size or width % cell_size:
        raise ValueError("Image dimensions must be multiples of cell size")
    yy, xx = np.indices((height, width))
    cell_id = (yy // cell_size) * (width // cell_size) + xx // cell_size
    position = angle * (bins / 180.0)
    lower = np.floor(position).astype(int) % bins
    fraction = position - np.floor(position)
    upper = (lower + 1) % bins
    count = (height // cell_size) * (width // cell_size) * bins
    low_votes = np.bincount((cell_id * bins + lower).ravel(),
                            weights=(magnitude * (1 - fraction)).ravel(), minlength=count)
    high_votes = np.bincount((cell_id * bins + upper).ravel(),
                             weights=(magnitude * fraction).ravel(), minlength=count)
    return (low_votes + high_votes).reshape(height // cell_size, width // cell_size, bins)


def hog_descriptor(image, return_maps=False):
    """224² -> 14x14 cells -> 13x13 overlapping blocks -> 6,084 values."""
    gray = prepare_hog(image)
    magnitude, angle = gradients(gray)
    cells = cell_histograms(magnitude, angle)
    windows = np.lib.stride_tricks.sliding_window_view(cells, (2, 2), axis=(0, 1))
    # Arrange each block as four neighboring cells, with nine bins per cell.
    blocks = windows.transpose(0, 1, 3, 4, 2).reshape(13, 13, 36)
    blocks = blocks / np.sqrt(np.sum(blocks ** 2, axis=-1, keepdims=True) + 1e-10)
    feature = normalize(blocks.reshape(-1)).astype(np.float32)
    if return_maps:
        return feature, gray, magnitude, cells
    return feature


class HOGRecognizer:
    """Read numeric weights only; share extraction with the training script."""
    def __init__(self, model_dir=None):
        folder = Path(model_dir or ROOT / "models/hog")
        self.config = json.loads((folder / "classifier.json").read_text(encoding="utf-8"))
        if self.config["parameters"] != PARAMETERS:
            raise ValueError("HOG model preprocessing does not match this implementation")
        with np.load(folder / "head.npz", allow_pickle=False) as arrays:
            self.coef = arrays["coef"].copy()
            self.intercept = arrays["intercept"].copy()

    def predict_features(self, feature):
        return softmax(np.asarray(feature) @ self.coef.T + self.intercept)

    def predict(self, path):
        feature = hog_descriptor(load_image(path))
        probabilities = self.predict_features(feature)
        index = int(probabilities.argmax())
        return {"label": self.config["classes"][index],
                "score": float(probabilities[index]),
                "probabilities": dict(zip(self.config["classes"], map(float, probabilities))),
                "note": "Exploratory HOG baseline; uncalibrated softmax scores."}
