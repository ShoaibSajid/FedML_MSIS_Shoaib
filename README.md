# FedMSIS-BiDir

**Federated Self-Training with Temporal Pseudo-Label Refinement for Object Detection**

Code release for the paper *FedMSIS-BiDir: Federated Self-Training with Temporal Pseudo-Label
Refinement for Object Detection* (IEEE Access, manuscript Access-2026-27285).

FedMSIS-BiDir trains an object detector across decentralized clients that each hold a small
amount of labeled data and a large amount of **unlabeled** driving video. Each client generates
pseudo-labels on its unlabeled frames, **recovers missed detections using bi-directional
temporal tracking** (forward + backward DeepSORT association), and trains locally. The server
aggregates client models with FedAvg and applies a **winner-takes-all best-model selection**:
every candidate model is evaluated on a held-out validation set each round and the
highest-scoring one becomes the next global model.

This repository is a fork of [FedML](https://github.com/FedML-AI/FedML); the FedMSIS-BiDir
method is implemented as a FedCV object-detection application.

---

## Where the code lives

The paper's implementation is under:

```
python/app/fedcv/object_detection/
```

See **[`python/app/fedcv/object_detection/README.md`](python/app/fedcv/object_detection/README.md)**
for full setup, dataset preparation, configuration, training, evaluation, and instructions to
reproduce the paper's results.

Key files:

| Path | Role |
|------|------|
| `python/app/fedcv/object_detection/main_fedml_object_detection.py` | Entry point |
| `python/app/fedcv/object_detection/trainer/yolov5_trainer.py` | Client trainer: local training, pseudo-label generation, bi-directional recovery, best-model tracking |
| `python/app/fedcv/object_detection/trainer/yolo_aggregator.py` | Server aggregator: validation, per-class mAP logging, best-model checkpointing |
| `python/app/fedcv/object_detection/config/` | FedML + YOLO + dataset configs |

The bi-directional temporal recovery pipeline (the core "BiDir" contribution) is a separate
**git submodule**,
[`Yolov5_DeepSORT_PseudoLabels`](https://github.com/ShoaibSajid/Yolov5_DeepSORT_PseudoLabels),
at `python/app/fedcv/object_detection/Yolov5_DeepSORT_PseudoLabels/`. Its recovery scripts
(`trackv2_from_file.py` forward/backward + `merge_forward_backward_v2.py`) can also be run
standalone — see the [app README](python/app/fedcv/object_detection/README.md#bi-directional-temporal-tracker-submodule).

---

## Quick start

```bash
# 0. Clone with submodules (or run `git submodule update --init --recursive` in an existing clone)
git clone --recurse-submodules https://github.com/ShoaibSajid/FedML_MSIS_Shoaib.git
cd FedML_MSIS_Shoaib

# 1. FedML core
pip install fedml --upgrade

# 2. Enter the app and install its dependencies
cd python/app/fedcv/object_detection
pip install -r requirements.txt

# 3. Initialize the bi-directional DeepSORT recovery submodule + install its deps
bash setup.sh

# 4. Run a single-machine MPI simulation
bash run_simulation.sh [CLIENT_NUM]
```

For cross-silo (multi-process / multi-node) runs and full reproduction settings, follow the
[app README](python/app/fedcv/object_detection/README.md).

---

## Datasets

Three autonomous-driving datasets, reduced to 3 shared classes (`vehicle`, `rider`,
`pedestrian`):

- [BDD100K](https://bdd-data.berkeley.edu/) — labeled source domain
- [Waymo Open](https://waymo.com/open/) — unlabeled target domain (pseudo-labeled)
- [Cityscapes](https://www.cityscapes-dataset.com/) — cross-domain evaluation

Data is split across 10 clients using the FedCV object-frequency non-IID protocol
(arXiv:2111.11066, Sec. 7.1.2).

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

---

## Acknowledgements & License

Built on top of [FedML: A Research Library and Benchmark for Federated Machine Learning](https://github.com/FedML-AI/FedML)
(He et al., NeurIPS 2020 FL Workshop, Best Paper Award) and the authors' prior centralized
bi-directional pseudo-label recovery work. Object tracking uses
[DeepSORT](https://github.com/nwojke/deep_sort) with YOLOv5.

Licensed under Apache-2.0 (see [`LICENSE`](LICENSE)), inherited from the upstream FedML project.
