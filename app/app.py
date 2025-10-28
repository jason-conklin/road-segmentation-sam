from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from sidewalk_sam.segment import run_segmentation
from sidewalk_sam.utils import load_config, resolve_path

DEFAULT_CONFIG_PATH = "configs/config.yaml"
UPLOAD_DIR = Path("demo/uploads")
APP_MASK_PATH = Path("outputs/app_mask.png")
APP_OVERLAY_PATH = Path("outputs/app_overlay.png")


def _ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def _load_config_safe() -> Optional[dict]:
    try:
        return load_config(DEFAULT_CONFIG_PATH)
    except FileNotFoundError:
        return None


app_ui = ui.page_fluid(
    ui.h2("Road Segmentation Demo"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.markdown(
                "Upload a small satellite tile and press **Generate Roads** to run "
                "the Segment Anything pipeline. Configure the SAM checkpoint path in "
                "`configs/config.yaml` before running."
            ),
            ui.input_file(
                "image_file",
                "Imagery Tile",
                accept=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                multiple=False,
            ),
            ui.input_slider("close_kernel", "Close kernel", min=1, max=25, value=5),
            ui.input_slider("open_kernel", "Open kernel", min=1, max=15, value=3),
            ui.input_slider("min_area", "Minimum area", min=0, max=2000, value=150),
            ui.input_slider("max_area", "Maximum area", min=1000, max=500000, value=300000),
            ui.input_action_button("run", "Generate Roads"),
        ),
        ui.panel_main(
            ui.output_text_verbatim("status"),
            ui.layout_columns(
                ui.column(
                    6,
                    ui.h4("Road Mask"),
                    ui.output_image("mask_preview"),
                ),
                ui.column(
                    6,
                    ui.h4("Road Overlay"),
                    ui.output_image("overlay_preview"),
                ),
            ),
        ),
    ),
)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    status_text = reactive.Value("Waiting for input...")
    mask_path = reactive.Value(None)
    overlay_path = reactive.Value(None)

    @output
    @render.text
    def status() -> str:
        cfg = _load_config_safe()
        if cfg is None:
            return "Config not found. Copy configs/config.example.yaml to configs/config.yaml."
        checkpoint_cfg = cfg.get("model", {}).get("sam_checkpoint_path")
        checkpoint = resolve_path(checkpoint_cfg) if checkpoint_cfg else None
        if checkpoint is None or not checkpoint.exists():
            return (
                "SAM checkpoint missing. Download the requested file and update "
                "`model.sam_checkpoint_path`."
            )
        return status_text.get()

    @output
    @render.image
    def mask_preview():
        current = mask_path.get()
        if not current:
            return None
        return {"src": str(current), "width": "100%"}

    @output
    @render.image
    def overlay_preview():
        current = overlay_path.get()
        if not current:
            return None
        return {"src": str(current), "width": "100%"}

    @reactive.effect
    @reactive.event(input.run)
    def _():
        cfg = _load_config_safe()
        if cfg is None:
            status_text.set("Config file not found. Create configs/config.yaml first.")
            mask_path.set(None)
            overlay_path.set(None)
            return

        uploads = input.image_file()
        if uploads:
            upload_info = uploads[0]
            upload_dir = _ensure_upload_dir()
            uploaded_path = upload_dir / upload_info["name"]
            shutil.copy(upload_info["datapath"], uploaded_path)
        else:
            default_input_cfg = cfg.get("io", {}).get("input_image")
            resolved_default = (
                resolve_path(default_input_cfg) if default_input_cfg else None
            )
            if resolved_default and resolved_default.exists():
                uploaded_path = resolved_default
            else:
                status_text.set("No upload provided and default input missing.")
                mask_path.set(None)
                overlay_path.set(None)
                return

        overrides = {
            "close_kernel": int(input.close_kernel()),
            "open_kernel": int(input.open_kernel()),
            "min_area": int(input.min_area()),
            "max_area": int(input.max_area()),
        }

        status_text.set("Running segmentation...")
        try:
            output_path = run_segmentation(
                config_path=DEFAULT_CONFIG_PATH,
                input_path=str(uploaded_path),
                output_path=str(APP_MASK_PATH),
                overlay_path=str(APP_OVERLAY_PATH),
                task="roads",
                postprocess_overrides=overrides,
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            status_text.set(f"Segmentation failed: {exc}")
            mask_path.set(None)
            overlay_path.set(None)
            return

        status_text.set(f"Mask saved to {output_path}")
        mask_path.set(output_path)
        overlay_path.set(APP_OVERLAY_PATH)


app = App(app_ui, server)
