"""Segmentation entry point built on top of Segment Anything."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PyTorch is required to run sidewalk_sam. Please install torch first."
    ) from exc

try:
    from samgeo.sam import SamAutomaticMaskGenerator as GeoMaskGenerator
except Exception:  # pragma: no cover - optional dependency
    print(
        "Note: SAMGeo 2 extras not installed. For improved performance install: "
        'pip install "segment-geospatial[samgeo2]"'
    )
    GeoMaskGenerator = None

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "segment-anything is required. Install it via `pip install segment-anything` "
        "or follow the project README."
    ) from exc

from .utils import (
    configure_logging,
    ensure_dir,
    generate_placeholder_image,
    get_checkpoint_path,
    load_config,
    make_overlay,
    resolve_path,
)

LOGGER = logging.getLogger(__name__)

# SamAutomaticMaskGenerator accepts a specific set of tuning options. We filter any
# config keys that are not part of this allowlist to prevent unexpected errors.
GENERATOR_ALLOWED_KEYS: Iterable[str] = (
    "points_per_side",
    "points_per_batch",
    "pred_iou_thresh",
    "stability_score_thresh",
    "stability_score_offset",
    "box_nms_thresh",
    "crop_n_layers",
    "crop_nms_thresh",
    "crop_overlap_ratio",
    "crop_n_points_downscale_factor",
    "min_mask_region_area",
    "output_mode",
)


def _select_device(preferred: Optional[str] = None) -> str:
    """Pick the torch device to run on."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _prepare_generator_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract valid mask generator kwargs from config."""
    inference_cfg = config.get("inference", {}) or {}
    return {
        key: inference_cfg[key]
        for key in GENERATOR_ALLOWED_KEYS
        if key in inference_cfg and inference_cfg[key] is not None
    }


def _aggregate_masks(
    masks: Iterable[Dict[str, Any]], image_shape: tuple[int, int]
) -> np.ndarray:
    """Combine individual mask results into a single binary mask."""
    height, width = image_shape[:2]
    combined = np.zeros((height, width), dtype=np.uint8)
    for item in masks:
        segmentation = item.get("segmentation")
        if segmentation is None:
            continue
        combined = np.maximum(combined, segmentation.astype(np.uint8))
    return combined * 255


def _skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Compute a morphological skeleton of a binary mask."""
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = binary_mask.copy()
    skeleton = np.zeros_like(mask)
    while True:
        eroded = cv2.erode(mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(mask, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        mask = eroded
        if cv2.countNonZero(mask) == 0:
            break
    return skeleton


def postprocess_roads(mask: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Road-focused cleanup heuristics."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)

    close_kernel = int(cfg.get("close_kernel", 0) or 0)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel, close_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    open_kernel = int(cfg.get("open_kernel", 0) or 0)
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    min_area = int(cfg.get("min_area", 0) or 0)
    max_area = int(cfg.get("max_area", 0) or 0) or None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    keep = np.zeros(num_labels, dtype=bool)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if width == 0 or height == 0:
            continue
        short_side = max(1, min(width, height))
        aspect = max(width, height) / short_side
        bbox_area = float(width * height) if width and height else 1.0
        density = area / bbox_area
        if aspect >= 2.0 or density < 0.65:
            keep[label] = True

    filtered = np.where(keep[labels], 255, 0).astype(np.uint8)

    if cfg.get("skeletonize"):
        skeleton = _skeletonize_mask(filtered)
        thicken = int(cfg.get("thicken", 0) or 0)
        if thicken > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thicken, thicken))
            skeleton = cv2.dilate(skeleton, kernel)
        filtered = skeleton

    return filtered


def run_segmentation(
    config_path: str = "configs/config.yaml",
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    overlay_path: Optional[str] = None,
    *,
    task: Optional[str] = None,
    postprocess_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Path:
    """Run SAM automatic mask generation for a single image tile.

    Returns the path to the generated mask.
    """
    configure_logging(verbose=verbose)
    config = load_config(config_path)
    if task:
        config["task"] = task

    if postprocess_overrides:
        post_cfg = config.get("postprocess", {}) or {}
        post_cfg.update({k: v for k, v in postprocess_overrides.items() if v is not None})
        config["postprocess"] = post_cfg

    io_cfg = config.get("io", {}) or {}
    input_candidate = input_path or io_cfg.get("input_image")
    if not input_candidate:
        raise ValueError("Input image path must be provided via CLI or config.")

    resolved_input = resolve_path(input_candidate)
    if not resolved_input.exists():
        if "example_tile" in resolved_input.name.lower():
            ensure_dir(resolved_input.parent)
            generate_placeholder_image(resolved_input)
            LOGGER.info("Created placeholder tile at %s", resolved_input)
        else:
            raise FileNotFoundError(
                f"Input image not found at {resolved_input}. "
                "Update configs/config.yaml or pass --input with a valid tile."
            )
    LOGGER.info("Using input image: %s", resolved_input)

    output_candidate = output_path or io_cfg.get("output_mask")
    if not output_candidate:
        raise ValueError("Output path must be provided via CLI or config.")
    resolved_output = resolve_path(output_candidate)
    ensure_dir(resolved_output)
    LOGGER.info("Writing mask to: %s", resolved_output)

    overlay_candidate = overlay_path or io_cfg.get("output_overlay")
    resolved_overlay = None
    if overlay_candidate:
        resolved_overlay = resolve_path(overlay_candidate)
        ensure_dir(resolved_overlay)
        LOGGER.info("Overlay preview will be saved to: %s", resolved_overlay)

    checkpoint_path = get_checkpoint_path(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint_path}. "
            "Download it and update configs/config.yaml."
        )

    model_type = config.get("model", {}).get("sam_model_type", "vit_h")
    device = _select_device(config.get("model", {}).get("device"))

    image = Image.open(resolved_input).convert("RGB")
    image_np = np.array(image)

    sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam_model.to(device=device)
    if device == "cuda":
        sam_model.eval()

    if not config.get("inference", {}).get("use_automatic_mask_generator", True):
        raise NotImplementedError(
            "Only the automatic mask generator is implemented for this project."
        )

    generator_kwargs = _prepare_generator_kwargs(config)
    generator_cls = GeoMaskGenerator or SamAutomaticMaskGenerator
    mask_generator = generator_cls(sam_model, **generator_kwargs)
    masks = mask_generator.generate(image_np)
    if not masks:
        raise RuntimeError("Mask generator did not return any masks.")

    mask_array = _aggregate_masks(masks, image_np.shape).astype(np.uint8)

    post_cfg = config.get("postprocess", {}) or {}
    mode = (post_cfg.get("mode") or "").lower()
    task_name = (config.get("task") or "roads").lower()
    if task_name == "roads" and mode == "roads":
        mask_array = postprocess_roads(mask_array, post_cfg)

    mask_image = Image.fromarray(mask_array)
    mask_image.save(resolved_output)

    if resolved_overlay is not None:
        viz_cfg = config.get("viz", {}) or {}
        alpha = float(viz_cfg.get("overlay_alpha", 0.45))
        color = viz_cfg.get("overlay_color", [255, 50, 50])
        overlay = make_overlay(image_np, mask_array, color, alpha)
        Image.fromarray(overlay).save(resolved_overlay)

    return resolved_output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a road segmentation mask from a satellite tile."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to the input imagery tile.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        help="Where to write the mask PNG.",
    )
    parser.add_argument(
        "--task",
        choices=["roads", "generic"],
        default="roads",
        help="Task preset to apply (affects postprocessing heuristics).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:  # pragma: no cover - CLI entry
    parser = _build_parser()
    args = parser.parse_args()

    try:
        preview_config = load_config(args.config)
    except FileNotFoundError:
        preview_config = {}

    preview_io = preview_config.get("io", {}) if isinstance(preview_config, dict) else {}
    preview_task = (
        args.task
        or (preview_config.get("task") if isinstance(preview_config, dict) else None)
        or "roads"
    )

    input_candidate = args.input_path or preview_io.get("input_image")
    if input_candidate:
        print(f"Input image: {resolve_path(input_candidate)}")
    else:
        print("Input image: <not configured>")

    output_candidate = args.output_path or preview_io.get("output_mask")
    if output_candidate:
        print(f"Output mask: {resolve_path(output_candidate)}")
    else:
        print("Output mask: <not configured>")

    overlay_candidate = preview_io.get("output_overlay")
    if overlay_candidate:
        print(f"Output overlay: {resolve_path(overlay_candidate)}")
    else:
        print("Output overlay: <not configured>")

    print(f"Task: {preview_task}")

    output = run_segmentation(
        config_path=args.config,
        input_path=args.input_path,
        output_path=args.output_path,
        overlay_path=None,
        task=args.task,
        postprocess_overrides=None,
        verbose=args.verbose,
    )
    print(f"Mask written to {output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
