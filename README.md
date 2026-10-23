# EV Battery Component Detection: Dissertation Experiments

Code and outputs from my dissertation for the PGDip in Artificial Intelligence at Aston University. The project compares object-detection variants for locating components in electric-vehicle battery pack teardown images, as a perception step for robotic disassembly.

All experiments were run by me on a cloud NVIDIA RTX A6000 (48 GB) provided through the university. The notebooks keep their original cell outputs, so every number below can be traced to a printed table in the repo.

## What is compared

| Variant | What changes |
|---|---|
| YOLOv9m baseline | Ultralytics YOLOv9m with the default CIoU box loss |
| SIoU-style loss | SIoU-inspired patch to `bbox_iou` (`elif SIoU:`) with a centre-distance term and an aspect-ratio term |
| Bicubic upsampling | Every `nn.Upsample` in the network replaced by a bicubic (`align_corners=False`) version |
| RT-DETR-l | Transformer detector as a contrasting architecture |

## Dataset

- 1,045 annotated images of EV battery pack components, 8 classes
- Split: 732 train / 209 validation / 104 test
- Validation set: 209 images, 1,289 labelled instances
- The dataset is not included in this repo. Paths inside the notebooks point to the original cloud workspace and must be edited to run elsewhere.

## Environment

- GPU: NVIDIA RTX A6000, 48,539 MiB
- Ultralytics 8.4.6, PyTorch 1.12.1 (CUDA 11.3)
- Seed 42, deterministic mode on
- Image size 768, batch size 16 (YOLOv9m runs); batch size 8 for RT-DETR

## Results (validation set, last run of each configuration)

| Run | Setup | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| baseline_siou_1045 | SGD, lr0 0.01, 50 epochs, 0.22 h | 0.819 | 0.765 | 0.808 | **0.618** |
| baseline_yolov9m_1045 | SGD, 50 epochs, custom augmentation, 0.22 h | 0.811 | 0.784 | 0.798 | 0.594 |
| custom_bicubic_siou_1045 | AdamW, lr0 0.003, 80 epochs, 0.35 h | 0.784 | 0.804 | 0.786 | 0.584 |
| custom_bicubic_1045 | AdamW, lr0 0.003, 80 epochs, 0.36 h | 0.815 | 0.777 | 0.799 | 0.571 |
| RT-DETR-l | AdamW, lr0 1e-4, batch 8, 50 epochs, 0.43 h | 0.655 | 0.396 | 0.373 | 0.262 |

The comparison table built from `results.csv` in the notebook (final epoch) reads: baseline mAP50 0.801 / mAP50-95 0.593 versus bicubic 0.778 / 0.570.

## Findings

- **YOLOv9m is a strong fit for this data.** All YOLOv9m variants land around 0.79 to 0.81 mAP50 and 0.57 to 0.62 mAP50-95.
- **The SIoU-style run gave the best mAP50-95 (0.618)** in the notebook, about 2.4 points above the plain baseline run.
- **Bicubic upsampling did not help.** Bicubic runs scored lower mAP50-95 than the baselines (0.571 and 0.584 versus 0.594 and 0.618), though they used a different optimiser and more epochs, so the effect is not isolated.
- **RT-DETR-l was clearly weaker** on this small dataset (0.373 mAP50), with recall of 0 for BMS_Unit, Coolant_tubes and low_volt_cable. Small datasets are a known weak spot for transformer detectors.
- **low_volt_cable is the hardest class in every run** (mAP50 between 0.36 and 0.40 for YOLOv9m; 13 validation images, 35 instances). Thin, flexible, visually variable objects are where the detectors struggle most.

## Limitations (please read before citing these numbers)

- The runs are not a controlled ablation. Baseline runs used SGD for 50 epochs; the bicubic runs used AdamW for 80 epochs. Differences in mAP50-95 mix the effect of the change with the effect of the optimiser and schedule.
- Single run per configuration, single seed. Differences of one to two points are within what I would expect from run-to-run noise on a 209-image validation set.
- The notebook defines the bicubic module after the cell that runs the run named `custom_bicubic_siou_1045`, so that label should be treated with caution. The `custom_bicubic_1045` run is the clearer bicubic result.
- The SIoU patch is applied to the Ultralytics source in site-packages by text substitution, which is fragile across library versions. A clean fork or a custom loss class would be the better design.
- The precision and recall logged by the helper in the comparison cell take the first value of the per-class array rather than the overall figure. The tables above use the validation summary printed by Ultralytics instead.
- The notebooks contain hard-coded cloud paths and a fixed `CUDA_VISIBLE_DEVICES` index.

## Repository layout

| Path | Contents |
|---|---|
| `NewModel.ipynb` | YOLOv9m baseline, SIoU-style patch, bicubic upsampling, training runs and comparison tables |
| `datasets/ev_battery_rt-detr/Rt-detr.ipynb` | RT-DETR-l training and validation on the same split |

## Reproducing

1. Install `ultralytics==8.4.6` with a matching PyTorch build.
2. Place the dataset in YOLO format with a `data.yaml` listing the 8 classes, and update the paths at the top of each notebook.
3. Run the notebook top to bottom; training outputs are written under `runs/`.

## Related work

A deployed version of this problem (FastAPI service, Docker, experiment log and model card) is in [ev_battery_AI](https://github.com/SreedasA/ev_battery_AI).

## Author

Sreedas Anjankudy, [GitHub](https://github.com/SreedasA)
