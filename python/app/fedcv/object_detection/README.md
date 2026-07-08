# FedMSIS-BiDir

**Federated Self-Training with Temporal Pseudo-Label Refinement for Object Detection**

Reference implementation for the paper *FedMSIS-BiDir: Federated Self-Training with Temporal
Pseudo-Label Refinement for Object Detection* (IEEE Access, manuscript Access-2026-27285).

FedMSIS-BiDir trains an object detector across decentralized clients that each hold a small
amount of labeled data and a large amount of **unlabeled** driving video. Each client generates
pseudo-labels on its unlabeled frames, **recovers missed detections using bi-directional
temporal tracking** (forward + backward DeepSORT association), and trains locally. The server
aggregates client models with FedAvg and, instead of trusting the averaged model blindly,
applies a **winner-takes-all best-model selection**: every candidate model is evaluated on a
held-out validation set each round and the highest-scoring one is propagated.

This code is built on top of [FedML](https://github.com/FedML-AI/FedML) (FedCV object-detection
app) with a modified YOLOv5 + DeepSORT pseudo-labeling pipeline.

---

## Method overview

The full pipeline runs each communication round as follows:

1. **Local supervised training** on each client's labeled shard (YOLOv5).
2. **Pseudo-label generation** on the client's unlabeled frames — high-confidence detections
   (`conf_thresh_high`) are kept as pseudo-labels.
3. **Bi-directional temporal recovery** — DeepSORT tracks objects both forward and backward in
   time to recover low-confidence / missed boxes; the forward and backward tracks are merged
   (`merge_forward_backward_v2`). This is the core "BiDir" contribution.
4. **Self-training** — recovered pseudo-labels are merged with the labeled data and the client
   continues training.
5. **Aggregation** — the server aggregates client updates (FedAvg).
6. **Winner-takes-all selection** — the server evaluates every candidate model on the held-out
   validation set and keeps the best one as the next global model.

Key components in this directory:

| Path | Role |
|------|------|
| `main_fedml_object_detection.py` | Entry point — builds the FedML runner, model, trainer, aggregator |
| `trainer/yolov5_trainer.py` | Client trainer: local training, pseudo-label generation, bi-directional recovery, best-model tracking |
| `trainer/yolo_aggregator.py` | Server aggregator: validation, per-class mAP logging, best-model checkpointing |
| `model/init_yolo.py` | Model/dataset/trainer initialization |
| `config/` | FedML + YOLO + dataset configuration files |

---

## Requirements

- Python ≥ 3.8, PyTorch ≥ 1.7 with CUDA (multi-GPU recommended — the paper uses 8× GPUs)
- [FedML](https://github.com/FedML-AI/FedML)
- The bi-directional recovery **git submodule**
  [`Yolov5_DeepSORT_PseudoLabels`](https://github.com/ShoaibSajid/Yolov5_DeepSORT_PseudoLabels)
  (declared in the repo-root `.gitmodules` at
  `python/app/fedcv/object_detection/Yolov5_DeepSORT_PseudoLabels`).

Install:

```bash
# 1. FedML core
pip install fedml --upgrade

# 2. App dependencies
pip install -r requirements.txt

# 3. Bi-directional DeepSORT pseudo-label recovery submodule + its deps (run from this directory)
bash setup.sh
```

`setup.sh` initializes the `Yolov5_DeepSORT_PseudoLabels` submodule (and its patched `yolov5`
fork) and installs their requirements. A Docker image is also referenced inside `setup.sh` for a
reproducible environment.

If you cloned the parent repo without submodules, fetch the tracker directly:

```bash
# clone the whole project with submodules in one step
git clone --recurse-submodules https://github.com/ShoaibSajid/FedML_MSIS_Shoaib.git

# or, from an existing clone, initialize just this submodule (run from the repo root)
git submodule update --init --recursive \
  python/app/fedcv/object_detection/Yolov5_DeepSORT_PseudoLabels
```

---

## Bi-directional temporal tracker (submodule)

The temporal pseudo-label recovery — the core "BiDir" contribution — lives in the
[`Yolov5_DeepSORT_PseudoLabels`](https://github.com/ShoaibSajid/Yolov5_DeepSORT_PseudoLabels)
submodule at `Yolov5_DeepSORT_PseudoLabels/`. During federated training it is called
automatically from `trainer/yolov5_trainer.py` (`recover_labels()` →
`trackv2_from_file.detect()` forward + backward, then `merge_forward_backward_v2.merge()`), but
each stage can also be run **standalone**.

Key scripts in the submodule:

| Script | Role |
|--------|------|
| `trackv2_from_file.py` | **Recovery step used in the paper.** Reads YOLO predictions (`.txt` per frame), runs DeepSORT, and recovers missed/low-confidence boxes by matching missed tracks against low-confidence detections. Supports `--reverse` for backward-in-time recovery. |
| `merge_forward_backward_v2.py` | Merges forward and backward recovered labels into the final pseudo-label set. |
| `track.py` / `trackv2.py` | Standard real-time YOLOv5 + DeepSORT trackers (detect + track directly from images/video/stream). |
| `deep_sort/configs/deep_sort.yaml` | DeepSORT association hyperparameters (`new_deepsort_config` in the FedML configs). |

### Run the bi-directional recovery standalone

Run from the `object_detection/` directory (the scripts add
`Yolov5_DeepSORT_PseudoLabels/` to `sys.path`). Each frame's low/high-confidence YOLO
predictions must already exist as `<frame>.txt` next to the images in `--source`.

```bash
cd Yolov5_DeepSORT_PseudoLabels

# 1. Forward recovery
python trackv2_from_file.py \
  --source <dir_of_frames_and_pred_txt> \
  --output <dir>/Recover-FW \
  --deep_sort_model resnet50_MSMT17 \
  --config_deepsort deep_sort/configs/deep_sort.yaml \
  --conf-thres 0.5 --iou-thres 0.6 --save-txt --device 0

# 2. Backward recovery (same command + --reverse)
python trackv2_from_file.py \
  --source <dir_of_frames_and_pred_txt> \
  --output <dir>/Recover-BW \
  --deep_sort_model resnet50_MSMT17 \
  --config_deepsort deep_sort/configs/deep_sort.yaml \
  --conf-thres 0.5 --iou-thres 0.6 --save-txt --reverse --device 0

# 3. Merge forward + backward
python merge_forward_backward_v2.py \
  --forward  <dir>/Recover-FW \
  --backward <dir>/Recover-BW \
  --merged   <dir>/Recover-Merged
```

The low/high confidence thresholds correspond to `conf_thresh_low` / `conf_thresh_high` in the
FedML configs. The recovered `Recover-Merged` labels are what get merged back into training.

### Run the plain YOLOv5 + DeepSORT tracker (detection from video/images)

```bash
cd Yolov5_DeepSORT_PseudoLabels
python track.py --source <img_or_video_or_dir> \
  --yolo_model best.pt --deep_sort_model resnet50_MSMT17 --save-txt --device 0
```

See `Yolov5_DeepSORT_PseudoLabels/README.md` for the full tracker option list.

---

## Datasets

The experiments use three autonomous-driving datasets, reduced to 3 shared classes
(`vehicle`, `rider`, `pedestrian`):

| Dataset | Role in the paper | Config |
|---------|-------------------|--------|
| [BDD100K](https://bdd-data.berkeley.edu/) | Labeled source domain | `data/3class_bdd.yaml` |
| [Waymo Open](https://waymo.com/open/) | Unlabeled target domain (pseudo-labeled) | `data/3class_waymo.yaml` |
| [Cityscapes](https://www.cityscapes-dataset.com/) | Cross-domain evaluation | see `data/` |

Each `data/*.yaml` points to `train` / `val` / `test` image-list `.txt` files. Convert each
dataset to YOLO format and update the paths in the corresponding YAML before training. In the
paper's federated setup the data is split across **10 clients** using the FedCV object-frequency
non-IID protocol (arXiv:2111.11066, Sec. 7.1.2), with ~7,000 images per client
(1,400 labeled / 5,600 unlabeled).

---

## Configuration

All experiment settings live in `config/*.yaml`. The most relevant blocks:

```yaml
train_args:
  federated_optimizer: "FedAvg"
  client_num_in_total: 10   # clients participating
  client_num_per_round: 10  # full participation each round
  comm_round: 50            # communication rounds
  epochs: 10                # local epochs per round
  batch_size: 128
  client_optimizer: sgd
  lr: 0.01
  weight_decay: 0.0005

model_args:
  model: "yolov5"
  class_num: 3
  yolo_cfg: "./model/yolov5/models/yolov5n.yaml"

new_data_args:
  use_new_data: True            # enable self-training on unlabeled data
  use_new_data_pseudos: True    # enable pseudo-label generation
  use_new_data_recover: True    # enable bi-directional temporal recovery
  conf_thresh_low: 0.1          # low-confidence threshold (recovery candidates)
  conf_thresh_high: 0.5         # high-confidence threshold (accepted pseudo-labels)
  psuedo_generate_clients: [...] # which clients generate pseudo-labels
  psuedo_recovery_on_clients: [...] # which clients run recovery
```

Ready-made configs are provided for the paper's settings, e.g.
`config/fedml_config_3class_bdd.yaml` (BDD source + Waymo target),
`config/fedml_config_3class_waymo.yaml`, and `config/simulation/fedml_config.yaml`.

> **Note:** before running, set your own Weights & Biases key (`tracking_args.wandb_key`) or set
> `enable_wandb: false`. Adjust `device_args.gpu_mapping_key` / `gpu_ids` to match your machine.

---

## Running

### Single-machine simulation (MPI)

```bash
bash run_simulation.sh [CLIENT_NUM]
# uses config/simulation/fedml_config.yaml
```

### Cross-silo (separate server + client processes)

```bash
# Server
python3 main_fedml_object_detection.py \
  --cf config/fedml_config_3class_bdd.yaml --run_id yolov5 --rank 0 --role server

# Client i (in a separate terminal / node), e.g. client 1
python3 main_fedml_object_detection.py \
  --cf config/fedml_config_3class_bdd.yaml --run_id yolov5 --rank 1 --role client
```

Helper scripts `run_server.sh` and `run_client.sh` wrap these commands. When using them across
machines, edit `run_id` and the `comm_args.grpc_ipconfig_path` (`config/grpc_ipconfig.csv`) to
avoid crosstalk.

Aggregated global-model checkpoints are written each `server_checkpoint_interval` rounds to
`runs/train/<project>/weights/aggr_model_<round>.pt`.

---

## Evaluation

Evaluate an aggregated checkpoint with the (patched) YOLOv5 validation script:

```bash
python Yolov5_DeepSORT_PseudoLabels/yolov5/val.py \
  --data config/multi_domain/data/3class_bdd.yaml \
  --weights runs/train/<project>/weights/aggr_model_<round>.pt \
  --device 0
```

The server aggregator also logs per-class mAP@0.5 and mAP@0.5:0.95 to W&B every round
(`trainer/yolo_aggregator.py`).

---

## Reproducing the paper

- **Detector:** YOLOv5n, input `640×640`.
- **Federation:** 10 clients, full participation, 50 communication rounds, 10 local epochs/round,
  SGD (lr 0.01, weight decay 5e-4), batch size 128.
- **Pseudo-labeling:** `conf_thresh_high = 0.5`, temporal window `W = 5`, recovery IoU
  `τ_IoU = 0.90`.
- **Selection:** winner-takes-all — validate all candidate models on a held-out set each round,
  propagate the best.
- **Seeds:** results are reported as mean ± std over seeds `{0, 42, 123}`.

Set the dataset paths in `data/3class_*.yaml`, pick the matching `config/fedml_config_*.yaml`,
and launch the cross-silo run above.

---

## Citation

```bibtex
@article{sajid2026fedmsisbidir,
  title   = {FedMSIS-BiDir: Federated Self-Training with Temporal Pseudo-Label
             Refinement for Object Detection},
  author  = {Sajid, Shoaib and others},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript Access-2026-27285}
}
```

This work builds on [FedML](https://github.com/FedML-AI/FedML) and the authors' prior
centralized bi-directional pseudo-label recovery work.

## License

Inherits the Apache-2.0 license of the upstream FedML project (see the repository-root `LICENSE`).
