# Training a YOLOv8n multi-label detector

This guide explains how to train the repository’s multi-label YOLOv8 pipeline on your five-class dataset:
- `standing`
- `sitting`
- `falling`
- `nutrition`
- `unknown`

The examples target YOLOv8n for speed, and they use only the minimal project features required for multi-label training.

## 1) Prepare your data
1. Arrange the dataset in standard YOLO layout:
   ```
   your_dataset/
     images/train/*.jpg
     images/val/*.jpg
     labels/train/*.txt
     labels/val/*.txt
   ```
2. Write label files in **multi-label YOLO format** (one line per box):
   ```
   <class_ids_comma_sep> <x_center> <y_center> <width> <height>
   ```
   - Coordinates are normalized to `[0,1]` relative to image width/height.
   - Comma-separated class IDs allow multiple labels on the same box.
   - Example (a sitting person receiving nutrition):  
     `1,3 0.45 0.52 0.20 0.25`  
     where `1=sitting`, `3=nutrition` per the class order below.

## 2) Create a dataset YAML
Save a file like `your_dataset.yaml` in the repo root:
```yaml
path: /abs/path/to/your_dataset   # folder containing images/ and labels/
train: images/train
val: images/val
nc: 5
names:
  0: standing
  1: sitting
  2: falling
  3: nutrition
  4: unknown
```

## 3) Choose a base model weight
Place the YOLOv8n weights inside `YOLO_models/` (e.g., `YOLO_models/yolov8n.pt`). Any Ultralytics weight path works as long as it’s accessible to the trainer.

## 4) Run training
Use the project’s `YOLOTrainer`, which already supports multi-label targets:
```bash
python - <<'PY'
from YOLO.trainer import YOLOTrainer

overrides = {
    "device": "0",                     # GPU id or "cpu"
    "project": "YOLO_multilabel_demo", # output folder prefix
    "data": "./your_dataset.yaml",
    "task": "detect",
    "model": "YOLO_models/yolov8n.pt",
    "epochs": 80,
    "optimizer": "SGD",
    "lr0": 0.002,
    "imgsz": 224,                     # cropped-person inputs
    "patience": 100,                  # early stopping patience (epochs)
    "save_period": 50,                # save checkpoint every 50 epochs
    # Light augmentations for person crops (full-image box):
    "hsv_h": 0.012, "hsv_s": 0.5, "hsv_v": 0.4,  # mild color jitter to keep tones realistic
    "degrees": 2.5, "translate": 0.04, "scale": 0.2, "shear": 1.0,  # gentle geo jitter; box spans crop
    "fliplr": 0.5, "flipud": 0.0,
    "mosaic": 0.5, "mixup": 0.0, "copy_paste": 0.0,  # higher mosaic for more spatial variety; mixup off for single-person crops
    "close_mosaic": 20,  # disable mosaic in the final epochs (set to ~epochs/4)
}

trainer = YOLOTrainer(overrides=overrides, req_loss=False)  # multi-label loss is enabled by default
trainer.train()
PY
```
Outputs land in `YOLO_multilabel_demo/train*/weights/` (e.g., `best.pt`, `last.pt`), with logs/plots in the same run directory.

## 5) Optional: constraint loss
For your class set, start **without** constraints. If later you want to enforce label rules, you can use the constraint loss:
1. Encode rules in `YOLO/constraints.npy` (shape `[num_constraints, 2 * nc]`).
2. Set `req_loss=True` in `YOLOTrainer` to activate the T-norm regularizer:
   ```python
   trainer = YOLOTrainer(overrides=overrides, req_loss=True)
   ```
Keep constraints minimal and verify validation metrics before/after enabling them.

### Example constraints for this dataset
- Mutual exclusivity: `standing`, `sitting`, `falling`, `unknown` must not co-occur.
- `nutrition` must **only** appear with `sitting` or `unknown`, and never alone.

Below is a helper snippet that builds `constraints.npy` for those rules. It follows the repository’s convention: columns `0..nc-1` are “class present”; columns `nc..2*nc-1` are “class absent.”

```python
import numpy as np
import scipy.sparse as sp

names = ["standing", "sitting", "falling", "nutrition", "unknown"]
nc = len(names)

rows, cols = [], []
def add(literals):
    r = len(rows) if rows else 0
    # ensure each constraint gets a new row id
    current = max(rows)+1 if rows else 0
    r = current
    for c in literals:
        rows.append(r)
        cols.append(c)

# Helper indices
present = lambda cls_id: cls_id
absent = lambda cls_id: cls_id + nc

# 1) Mutual exclusivity among standing/sitting/falling/unknown
exclusive_ids = [0,1,2,4]
for i in range(len(exclusive_ids)):
    for j in range(i+1, len(exclusive_ids)):
        a, b = exclusive_ids[i], exclusive_ids[j]
        add([present(a), present(b)])

# 2) nutrition only with sitting or unknown
# Penalize nutrition with standing or falling
add([present(3), present(0)])  # nutrition + standing
add([present(3), present(2)])  # nutrition + falling
# Penalize nutrition alone (requires at least sitting or unknown)
add([present(3), absent(1)])   # nutrition and NOT sitting
add([present(3), absent(4)])   # nutrition and NOT unknown

num_constraints = max(rows) + 1
data = np.ones(len(rows), dtype=np.float32)
mat = sp.coo_matrix((data, (rows, cols)), shape=(num_constraints, 2 * nc))
np.save("YOLO/constraints.npy", mat)
print("Saved constraints.npy", mat.shape, "rows:", num_constraints)
```

## 6) Inference reminder
- Use the trained weight with Ultralytics’ `model.predict` or `YOLO/tester.py`.
- The custom post-processing keeps per-box multi-label scores in `boxes_all`, even though a single display class may be shown for visualization.

## 7) Helper scripts (224px cropped-person workflow)
Two ready-to-run scripts are included for the 5-class, cropped-person setup:

   1. **Train**  
   ```bash
   python scripts/train_multilabel_crops.py \
     --data ./your_dataset.yaml \
     --weights YOLO_models/yolov8n.pt \
     --project YOLO_multilabel_224 \
     --epochs 80 \
     --batch 32 \
     --imgsz 224 \
     --patience 100 \
     --save-period 50 \
     --use-constraints  # optional: enable if YOLO/constraints.npy is prepared
   ```
   This uses light augmentations tuned for person crops and keeps the bounding box as the full crop.

   Aug rationale:
   - **Color**: small HSV shifts (h=0.012, s=0.5, v=0.4) to avoid over-altering skin/cloth tones.
   - **Geometry**: mild rotation/translate/scale (2.5°, 0.04, 0.2) since each crop is already aligned to one person; box covers the whole crop.
   - **Mixing**: Mosaic at 0.5 for more spatial variety; MixUp/Copy-Paste off to avoid unrealistic blends for single-person crops.
   - **Schedule**: `close_mosaic` ≈ last quarter of training (20 when epochs=80) to stabilize later epochs.
   - **Checkpoints/Early stop**: `save_period=50` saves checkpoints every 50 epochs; `patience=100` halts if validation fitness doesn’t improve for 100 epochs.

2. **Predict** (images or folder)  
   ```bash
   python scripts/predict_multilabel_crops.py \
     --weights YOLO_multilabel_224/train1/weights/best.pt \
     --source path/to/crops_or_folder \
     --imgsz 224 \
     --jsonl-output predictions.jsonl  # optional structured output
   ```
   The script attaches the project’s multi-label postprocess so `boxes_all` contains all per-class scores for each kept box.
