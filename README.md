# Wall Crack Detection — Inspection Pipeline for a Wall-Climbing Robot

A crack detection, measurement, and classification pipeline built for an
automated wall inspection workflow (wall-climbing robot camera feed), with
two interchangeable detection backends and a Streamlit dashboard for
reviewing results.

## Why two detectors

- **Classical CV detector** (`crack_inspector.py`) — rule-based, needs no
  training data, runs fully on CPU. This is the reliable baseline: it
  works on any wall photo out of the box, at the cost of being sensitive
  to heavy surface texture (raw brick, rough plaster) producing false
  positives.
- **LightUNet detector** (`unet_train_template.py`) — a lightweight U-Net
  trained in three stages: pretrain on the public DIC Crack4 dataset →
  generate pseudo-labels → fine-tune. Both detectors output the same mask
  format, so the app, measurement, and reporting code don't care which
  one produced the mask.

## Honest status

The U-Net is pretrained and verified working on the public DIC Crack4
dataset only. **It has not yet been validated on real wall-climbing-robot
footage** — different lighting, camera, and surface conditions than the
lab dataset it trained on. Treat the classical detector as the
production-ready path today, and the U-Net as a research artifact ready
for fine-tuning once real deployment images are available. This isn't a
caveat I'm hiding — it's the actual next step in the roadmap below.

## Pipeline

```
Wall Image
    |
    v
Preprocessing (lighting correction, denoising)
    |
    v
Crack Segmentation — classical (rule-based) OR trained U-Net
    |
    v
Binary Crack Mask
    |
    +--> Length Measurement (skeleton-based)
    |
    +--> Width Measurement (distance transform)
    |
    v
Classification: Hairline (<0.3mm) / Medium (0.3–1mm) / Large (>1mm)
    |
    v
Inspection Report (annotated image + measurements + confidence)
```

## Install & run

Requires Python 3.10–3.12 (PyTorch does not yet support 3.13+ as of this
writing — check before installing on a newer interpreter).

```bash
pip install -r requirements.txt
streamlit run app.py
```

If using the U-Net path with an NVIDIA GPU, install a CUDA-matched
PyTorch build instead of the CPU default — see
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
for the right command for your driver (`nvidia-smi` to check).

## Calibration (required before measurements mean anything)

Crack length/width are reported in millimetres, converted from pixels via
`mm_per_pixel`. This must be calibrated once per fixed camera setup:

1. Photograph a ruler/object of known length at the camera's fixed
   inspection distance.
2. Measure that object's length in pixels in the photo.
3. `mm_per_pixel = known_length_mm / measured_length_pixels`
4. Enter the value in the app sidebar (default `0.15` is a placeholder,
   not calibrated to any real setup).

## Data & checkpoint

The DIC Crack4 dataset (~530 labeled train/valid/test images) used to
pretrain the U-Net is not included in this repo due to size — download it
from [Zenodo (DOI: 10.5281/zenodo.4307686)](https://zenodo.org/records/4307686),
licensed CC-BY-4.0. Please cite the source if you use it:

> Rezaie, A., Achanta, R., Godio, M., & Beyer, K. (2020). Comparison of
> crack segmentation using digital image correlation measurements and
> deep learning. *Construction and Building Materials*, 261, 120474.
> https://doi.org/10.1016/j.conbuildmat.2020.120474

The pretrained checkpoint (`pretrained_unet.pt`, ~7.5MB) is available as
a [GitHub Release](../../releases) asset rather than committed directly,
to keep the repo lightweight to clone.
Here is the release link for the .pt that was trained using the DIC Crack4 Dataset :
https://github.com/Shazin157/Wall_Crack_Detector/releases/tag/v1.0.0

## Files

| File | Purpose |
|---|---|
| `crack_inspector.py` | Classical CV detection, measurement, classification |
| `unet_train_template.py` | LightUNet: pretrain / pseudo-label / fine-tune pipeline |
| `prepare_dataset.py` | Converts raw labeled data into training-ready format |
| `report_generator.py` | Annotated report generation |
| `app.py` | Streamlit dashboard |

## Known limitations

- Classical detector: false positives on heavily textured surfaces;
  misses very fine hairline cracks under poor lighting.
- Confidence score reflects mask continuity/quality, not a calibrated
  statistical probability.
- U-Net accuracy on real deployment footage is unverified until
  fine-tuned on images from the actual inspection camera.

## Roadmap

1. Collect representative wall images from the actual inspection
   camera/robot across expected wall types and lighting.
2. Fine-tune the U-Net on that data (pseudo-labeling pipeline already
   built — no manual annotation required).
3. Validate fine-tuned model against the classical baseline before using
   it for production decisions.
4. Automated image ingestion from the robot (currently manual upload /
   folder-watch).
