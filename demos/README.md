# Demos

The main [`sfm.py`](../sfm.py) pipeline is a **scaffold** — method bodies are stubs. These demos exercise real OpenCV stages you can run today while implementing the full SfM stack.

## Demo 1 — Keypoint detection & matching

**Script:** [`demo_keypoints.py`](demo_keypoints.py)

```bash
cp config.yaml.example config.yaml
# Set images_dir to a folder with ≥2 overlapping photos
pip install -r requirements.txt
python demos/demo_keypoints.py
```

**Output:** `output/matches_000_001.jpg`, … — side-by-side match visualizations.

**Tip:** Use sequential phone photos of the same scene with 60–80% overlap between consecutive frames.

## Demo 2 — Pipeline skeleton (development)

```python
from pathlib import Path
import cv2
from sfm import StructureFromMotion

cfg_images = sorted(Path("your-images").glob("*.jpg"))
images = [cv2.imread(str(p)) for p in cfg_images]

sfm = StructureFromMotion(images, feature_detector="SIFT")
# Will fail until stub methods are implemented:
# sfm.run()
```

Implement methods in order: `detect_and_describe_keypoints` → `match_keypoints` → `initialize_structure` → …

## Suggested sample datasets

| Dataset | Source | Notes |
|---|---|---|
| Fountain (EPFL) | [Vision datasets](https://www.epfl.ch/labs/cvlab/data/data-em/) | Classic multi-view benchmark |
| Custom phone scan | Local capture | 20–50 images, orbit around object |
| Lab room corners | RBD Lab captures | Pairs with other rbdlabhaifa repos |

No sample images are bundled (size / licensing). Add your own under a local `data/` folder (gitignored via `output/`).
