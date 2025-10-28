from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "sam_vit_b_01ec64.pth"
CONFIG_PATH = Path("configs/config.yaml")


def stream_download(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    """Download a file in chunks to the destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as outfile:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            outfile.write(chunk)


def load_config(path: Path) -> Optional[dict]:
    if not path.exists():
        print(f"Config file not found: {path}. Create it before running this script.")
        return None
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def update_checkpoint_path(config: dict, path: Path) -> bool:
    config.setdefault("model", {})
    current = config["model"].get("sam_checkpoint_path")
    target = str(path.as_posix())
    if current == target:
        return False
    config["model"]["sam_checkpoint_path"] = target
    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)
    return True


def main() -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint to {CHECKPOINT_PATH} ...")
    stream_download(CHECKPOINT_URL, CHECKPOINT_PATH)
    size = CHECKPOINT_PATH.stat().st_size
    print(f"Downloaded checkpoint size: {size:,} bytes")

    config = load_config(CONFIG_PATH)
    if config is None:
        return 1

    changed = update_checkpoint_path(config, CHECKPOINT_PATH)
    if changed:
        print(
            f"Updated {CONFIG_PATH} with model.sam_checkpoint_path = "
            f'"{CHECKPOINT_PATH.as_posix()}"'
        )
    else:
        print("Config already points to the downloaded checkpoint.")

    print("Now rerun: python -m sidewalk_sam.segment --input demo/sample_data/example_tile.tif --out outputs/mask.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

