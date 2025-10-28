"""Utility helpers for the sidewalk segmentation package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

LOGGER = logging.getLogger(__name__)

# Resolve repository root based on this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: Union[str, Path]) -> Path:
    """Return an absolute path (project-root relative if needed)."""
    candidate = Path(path_like).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=False)


def ensure_dir(path_like: Union[str, Path]) -> None:
    """Ensure the directory (or parent directory if a file path) exists."""
    path = Path(path_like)
    target = path if path.suffix == "" else path.parent
    target.mkdir(parents=True, exist_ok=True)


def generate_placeholder_image(out_path: Path, w: int = 512, h: int = 512) -> Path:
    """Create a synthetic tile for quick demos."""
    ensure_dir(out_path)
    tile = Image.new("RGB", (w, h), (68, 74, 83))
    draw = ImageDraw.Draw(tile)

    # Checkerboard base
    step = max(32, min(w, h) // 16)
    for y in range(0, h, step):
        for x in range(0, w, step):
            shade = 80 if (x // step + y // step) % 2 == 0 else 60
            draw.rectangle([x, y, x + step, y + step], fill=(shade, shade, shade))

    # Sidewalk stripes
    stripe_width = max(8, w // 24)
    for offset in range(w // 4, w, w // 4):
        draw.rectangle(
            [offset, 0, offset + stripe_width, h],
            fill=(190, 190, 190),
            outline=None,
        )

    tile.save(out_path)
    return out_path


def make_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Iterable[int],
    alpha: float,
) -> np.ndarray:
    """Blend a colored mask onto an RGB array."""
    overlay_color = np.array(list(color), dtype=np.uint8)
    mask_rgb = np.zeros_like(rgb, dtype=np.uint8)
    mask_rgb[mask > 0] = overlay_color
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(rgb, 1.0 - alpha, mask_rgb, alpha, 0.0)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    resolved = resolve_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fp:
        config: Dict[str, Any] = yaml.safe_load(fp) or {}
    return config


def get_checkpoint_path(config: Dict[str, Any]) -> Path:
    """Return the resolved checkpoint path from configuration."""
    model_cfg = config.get("model", {})
    checkpoint_path = model_cfg.get("sam_checkpoint_path")
    if not checkpoint_path:
        raise ValueError("Missing `model.sam_checkpoint_path` in config.")
    return resolve_path(checkpoint_path)


def configure_logging(verbose: bool = False) -> None:
    """Configure basic logging for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
