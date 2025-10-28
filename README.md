# Road Segmentation from Satellite Imagery (SAM + heuristics)

This project wires the Segment Anything Model (SAM) into a lightweight pipeline that turns a satellite tile into a road mask. It bundles a CLI, a Shiny for Python demo, and a reproducible notebook that mirror the workflow used during the research spikes.

**Workflow highlights**
- configure SAM checkpoint + tile paths in `configs/config.yaml`
- run `python -m sidewalk_sam.segment` to make a mask and overlay
- explore the result through the Shiny demo or notebook

## Quickstart
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the sample config and edit it:
   ```bash
   cp configs/config.example.yaml configs/config.yaml
   ```
   - point `model.sam_checkpoint_path` at a downloaded SAM checkpoint (see [SAM release](https://github.com/facebookresearch/segment-anything#model-checkpoints))
   - set `io.input_image` to a small satellite tile (place it in `demo/sample_data/`)
4. (Optional) Adjust mask generator knobs under `inference` or tweak the road postprocessing block to match your data.

## Usage
- **CLI**  
  ```powershell
  $env:PYTHONPATH="src"
  python -m sidewalk_sam.segment --input demo/sample_data/example_tile.tif --out outputs/roads_mask.png
  ```
  Omit flags to use the paths defined in `configs/config.yaml`. If `demo/sample_data/example_tile.tif` is missing, the CLI will auto-create a small placeholder tile the first time. Running the command also creates an overlay PNG if `io.output_overlay` is set.

- **Shiny demo**  
  ```bash
  shiny run --reload app/app.py
  ```
  Upload a tile (or rely on the configured default) and the app will display both the generated road mask and overlay.

## Road heuristics
- Morphological closing stitches broken asphalt segments into continuous ribbons.
- Morphological opening removes speckles from buildings or trees that SAM still picks up.
- Connected-component filtering keeps regions within `min_area` and `max_area`.
- Elongation heuristics retain long, thin components by keeping aspect ratio >= 2.0 or moderately sparse footprints.
- Optional skeletonization lets you compress roads to centerlines and re-thicken by a configurable amount.
- A color overlay blends the mask with the source tile for quick quality checks.

## What works / What's rough edges
- Works: automatic mask generator delivers reasonable road outlines on high-resolution tiles.
- Works: config-driven paths make it easy to swap checkpoints and tiles without changing code.
- Works: notebook and Shiny app both call the shared `run_segmentation` entry point.
- Rough edge: manual download of the 2.6 GB SAM checkpoint blocks first run.
- Rough edge: CPU inference is slow; GPU support depends on your PyTorch install.
- Rough edge: heuristics are intentionally simple and may need tuning per geography.

## Known Limitations
- Needs a SAM checkpoint on disk; follow the [official instructions](https://github.com/facebookresearch/segment-anything#model-checkpoints).
- Defaults target a single tile; batching and georeferenced outputs are future work.
- Demo tile is not bundled-see `demo/sample_data/README.md` for download pointers.

## Data
- Keep large GeoTIFFs out of version control. Drop a small public tile into `demo/sample_data/` and update the config.
- Example source: [Massachusetts Roads sample](https://storage.googleapis.com/open-cities-ai-test/Massachusetts-Roads/roads_image.png) or any small tile from [USGS EarthExplorer](https://earthexplorer.usgs.gov/).

## Troubleshooting
- **Torch/CUDA mismatch** - Reinstall PyTorch with the wheel that matches your CUDA version (`pip install torch --index-url https://download.pytorch.org/whl/cu121`).
- **Checkpoint not found** - Double-check `model.sam_checkpoint_path` in `configs/config.yaml`.
- **Force CPU** - Set `model.device` to `"cpu"` in the config.
- **SAMGeo sam2 warning** - If you see "There was an error importing sam2...", install extras with `pip install "segment-geospatial[samgeo2]"`. The CLI still runs with the baseline samgeo build.

## Milestones (archived)
- **Milestone 1 (Mar 24, 2024)** - Bootstrap repo, install QGIS/Jupyter/PyTorch, and ingest docker example assets.
- **Milestone 2 (Apr 7, 2024)** - Follow the samgeo automatic mask generator tutorial on Colab and capture outputs.
- **Milestone 3 (Apr 21, 2024)** - Extend experimentation in `milestone3.ipynb` using the provided video instructions.
- **Milestone 4 (Apr 28, 2024)** - Deploy a Shiny for Python demo on Hugging Face Spaces for tile uploads.

## Future TODO
- Add polygon post-processing that snaps roads to existing centerlines.
- Parameterize mask thresholding before PNG export.
- Batch processing over folders of tiles.
- Optional GeoTIFF export that preserves georeferencing.

## References
- [segment-geospatial documentation](https://samgeo.gishub.org/)
- [Segment Anything Model paper](https://arxiv.org/abs/2304.02643)
- [Walkthrough video used during milestones](https://youtu.be/YHA_-QMB8_U)
