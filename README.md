# BeamID: Domain-Adaptive Radio Fingerprinting with MIMO Beamforming Feedback


This repository implements a domain-adaptive radio fingerprinting framework that uses standard-compliant MIMO beamforming feedback matrices to learn client-discriminative embeddings for fingerprinting radios (Network Interface Cards).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{haque2026beamid,
  title={BeamID: Domain-Adaptive Radio Fingerprinting with MIMO Beamforming Feedback},
  author={Haque, Khandaker Foysal and Meneghello, Francesca and Restuccia, Francesco},
  booktitle={IEEE International Conference on Network Softwarization (NetSoft)},
  year={2026}
}
```

## What this repo contains

- `model.py` — encoder / projector / classifier network definitions.
- `datagen.py` — dataset loader for `Dataset/vmatrices/*.npy`.
- `loss.py` — supervised contrastive loss implementation.
- `train.py` — source-domain training using supervised contrastive learning (SupCon) and optional classifier loss.
- `viz_cosine_gap_v3.py` — generate 3-panel cosine-gap plots for same-domain vs cross-domain before/after domain adaptation.

## Core idea

BeamID learns radio fingerprints from MIMO beamforming feedback. It trains on a source domain and then adapts with a few labeled target-domain samples directly in embedding space, restoring cross-environment client identification accuracy.

## Download Dataset
Please download the dataset from [HuggingFace Repo](https://huggingface.co/datasets/foysalhaque/BeamID/tree/main)


First Install the Hugging Face CLI:

```
pip install -U "huggingface_hub[cli]"
```

Download the `vmatrices` folder:

```
hf download foysalhaque/BeamID \
    --repo-type dataset \
    --include "vmatrices/*" \
    --local-dir Dataset
```

The files will be stored under:

```text
Dataset/vmatrices/
```


- Expected filenames are of the form `<mac>_<loc>.npy`.
- Each `.npy` contains complex-valued beamforming feedback samples.
- `datagen.py` converts them to real-valued tensors of shape `(16, 250)` for model input.

## Quick start


1. Train a source-domain model:

```bash
python train.py Dataset/vmatrices \
  --source 1 2 3 4 5 6 \
  --target 7 8 9 \
  --k 15 \
  --epochs 30 \
  --batch 128 \
  --lr 1e-3 \
  --save checkpoints/checkpoint.pt
```

2. Generate the cross-domain adaptation plot:

```bash
python viz_cosine_gap_v3.py Dataset/vmatrices \
  --ckpt checkpoints/checkpoint.pt \
  --source 1 2 3 \
  --target 4 \
  --k 15 \
  --pairs 50000 \
  --ft_steps 800 \
  --ft_lr 5e-4 \
  --unfreeze_last \
  --out cosine_gap.png
```

3. If you already have cached cosine-gap data:

```bash
python viz_cosine_gap_v3.py Dataset/vmatrices \
  --cache cosine_gap_cache_v3.npz \
  --plot_only \
  --out cosine_gap.png
```

For any further information, please contact [K Foysal Haque](https://kfoysalhaque.github.io/) at [haque.k@northeastern.edu](mailto:haque.k@northeastern.edu)


