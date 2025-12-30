"""
sahi_obbeval.py
- Requires: ultralytics, sahi, pycocotools, torch, torchvision
  pip install ultralytics sahi pycocotools
- Edit the CONFIG section below for your environment.
"""
import json
from pathlib import Path

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from ultralytics.models.yolo.obb.val import OBBValidator
from types import SimpleNamespace

# ---------- CONFIG ----------
MODEL_PATH = 'final-mosaic-augmentation.pt'          # your YOLO11 OBB model
VAL_IMAGE_DIR = "fresno_obb/valid/images"      # folder with 5000x5000 .tif images
DATA_YAML = "fresno_obb/data.yaml"              # data.yaml pointing to val and names for OBB task

MODEL_PATH = 'best.pt'          # your YOLO11 OBB model
VAL_IMAGE_DIR = "fresno_seg/valid/images"      # folder with 5000x5000 .tif images
DATA_YAML = "fresno_seg/data.yaml"              # data.yaml pointing to val and names for OBB task

img1 = 'fresno_seg/valid/images/11ska610815.tif'

SAVE_DIR = Path("sahi_seg_eval")          # where predictions.json and evaluation outputs are saved
DEVICE = "cuda"                              # use "cpu" if no GPU
SLICE_H = 640                                  # tile height (try 512-640 for imgsz=384)
SLICE_W = 640                                  # tile width
OVERLAP_H = 0.2                                # 20% overlap
OVERLAP_W = 0.2
CONF_THRESH = 0.001                            # keep low to export all preds; validator will filter
# ----------------------------

SAVE_DIR.mkdir(parents=True, exist_ok=True)
PRED_JSON = SAVE_DIR / "predictions.json"

# 1) Run SAHI tiled inference and save COCO-style JSON with poly + rbox
print("Running SAHI sliced inference...")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="best.pt",  # any yolov8/yolov9/yolo11/yolo12/rt-detr det model is supported
    confidence_threshold=0.35,
    device="cuda",  # or 'cuda:0' if GPU is available
)


res = get_sliced_prediction(
    image=img1,
    detection_model=detection_model,
    slice_height=SLICE_H,
    slice_width=SLICE_W,
    overlap_height_ratio=OVERLAP_H,
    overlap_width_ratio=OVERLAP_W,
    verbose=2,
    perform_standard_pred=False,
    # slice_dir=str(SAVE_DIR),
    # slice_export_prefix='slice_',
    # novisual=True,
    # progress_bar=True,
    postprocess_match_metric="IOU",   # ensure polygon/rbox outputs
    # output_coco=True,                 # tell SAHI to write COCO json
    # dataset_json_path=str(PRED_JSON),
    # model_confidence_threshold=CONF_THRESH, # export low-threshold preds; you can increase if desired
    # postprocess_match_threshold=
)

print(res)
print('----')
print(res.object_prediction_list[0])
print('----')
# print(res.to_coco_annotations())
# print('----')
with open(PRED_JSON, 'w') as f:
    # f.write(str(res.to_coco_predictions()))
    json.dump(res.to_coco_predictions(), f, indent=1)


if not PRED_JSON.exists():
    raise FileNotFoundError(f"SAHI did not produce predictions.json at {PRED_JSON}")

print(f"Saved SAHI predictions to {PRED_JSON}")

# 2) Use Ultralytics OBBValidator to evaluate predictions.json against dataset (DOTA-style output + mAP)
print("Running Ultralytics OBBValidator.eval_json...")

# Prepare args similar to how OBBValidator expects them
args = SimpleNamespace()
args.model = MODEL_PATH
args.data = DATA_YAML
args.split = "val"
args.save_json = True                # required for eval_json branch
args.save_dir = str(SAVE_DIR)
args.plots = False
args.device = DEVICE
args.batch = 1

# instantiate validator; it will use data yaml to locate annotations
validator = OBBValidator(args=args)

# load predictions.json into validator.jdict (same structure expected)
with open(PRED_JSON, "r", encoding="utf-8") as f:
    validator.jdict = json.load(f)

# Flag is_dota must be True for eval_json to output DOTA txts; validator.init_metrics sets is_dota
# but we ensure it by checking the data yaml (if your data.yaml path contains 'DOTA' naming, it's fine).
# Now call eval_json to create DOTA prediction txts and merged txts, and compute stats.
stats = {}
stats = validator.eval_json(stats)

# The validator writes DOTA txts to SAVE_DIR/predictions_txt and SAVE_DIR/predictions_merged_txt
print("Evaluation complete.")
print(f"Saved DOTA prediction txts to {SAVE_DIR/'predictions_txt'}")
print(f"Saved merged prediction txts to {SAVE_DIR/'predictions_merged_txt'}")

# Note: OBBValidator.eval_json only writes predictions in DOTA format and returns stats dict.
# To compute mAP numbers programmatically with the OBBValidator pipeline, it's common to run the
# full `model.val(data=..., task='obb')` which performs inference+evaluation and prints metrics.
# If you need the numeric mAP values, run the model.val flow below instead of eval_json:

# Example: run full validation (this will re-run inference through the model, skip if you only want eval_json)
if False:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    results = model.val(data=DATA_YAML, task="obb", device=DEVICE)
    # print results summary
    try:
        print("mAP50:", results.box.map50, "mAP50-95:", results.box.map)
    except Exception:
        print("Full model.val completed; inspect saved logs in", SAVE_DIR)

print("Done.")
