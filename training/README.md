# S²AM3D Training Code

This directory contains **Encoder Train** and **Decoder Train**. Recommended order: initialize / train the encoder, extract per-point features, then train the decoder.

```text
training/
├── README.md
├── encoder_train/            # contrastive encoder training
└── decoder_train/            # decoder training (pre-extracted features)
```

---

## 1. Encoder Train

Contrastive finetuning of a PartField-style PVCNN + triplane encoder (intra-object).

### 1.1 Layout

| Path | Description |
|------|-------------|
| `encoder_train/train/encoder_train.py` | Entry point |
| `encoder_train/train/encoder_train_model.py` | Lightning module |
| `encoder_train/train/encoder_train.yaml` | Config |
| `encoder_train/partfield/` | Backbone, config, dataloader |
| `encoder_train/meta/train_meta.example.json` | Metadata example |

### 1.2 Environment

```bash
cd training/encoder_train
conda env create -f environment.yml
conda activate partfield
```

### 1.3 Pretrained weights

Get the initialization checkpoint from the official PartField repository:

https://github.com/nv-tlabs/PartField

Do **not** put the weight path in the yaml. Pass it on the command line:

```bash
--opts continue_ckpt /path/to/partfield.ckpt
```

### 1.4 Data format

Each sample is a `.npy` dict with coordinates and part labels (e.g. `coord`, `label`).

Metadata JSON is a list, for example:

```json
[
  {"id": "sample_0001", "data_path": "/path/to/sample_0001.npy", "dataset": "Objaverse"}
]
```

Edit `dataset.train_metadata_paths` in `encoder_train/train/encoder_train.yaml`.

### 1.5 Train

```bash
cd training/encoder_train
CUDA_VISIBLE_DEVICES=0 python train/encoder_train.py -c train/encoder_train.yaml \
  --opts continue_ckpt /path/to/partfield.ckpt
```

Outputs are saved under `encoder_train/train/results/<timestamp>/`.

---

## 2. Decoder Train

Train the scale-controllable part segmentation decoder using frozen / offline encoder features.

### 2.1 Layout

| Path | Description |
|------|-------------|
| `decoder_train/code/main.py` | Entry point (DDP) |
| `decoder_train/code/train.py` | Trainer |
| `decoder_train/code/train_dataset.py` | Data loading and prompt sampling |
| `decoder_train/code/configs/train.yaml` | Config |
| `decoder_train/code/models/` | Enhancer / Decoder / SegHead |
| `decoder_train/code/meta/train_meta.example.json` | Metadata example |

### 2.2 Environment

```bash
cd training/decoder_train
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA driver.

### 2.3 Data and features

1. Point cloud `.npy` (dict) with at least `coord` and `label`
2. Metadata JSON in the same format as the encoder (see `decoder_train/code/meta/train_meta.example.json`)
3. Pre-extracted features under `dataset.feat_base_path`:

```text
{feat_base_path}/part_feat_{basename}_0.npy
```

`basename` is the point-cloud filename without `.npy` (e.g. `sample_0001.npy` → `part_feat_sample_0001_0.npy`). Feature dim is 448.

Set in `decoder_train/code/configs/train.yaml`:

- `dataset.train_split`
- `dataset.feat_base_path`

### 2.4 Train

```bash
cd training/decoder_train/code
python main.py --config configs/train.yaml --ngpu 2

# resume
python main.py --config configs/train.yaml --ngpu 1 --resume /path/to/ckpt.pt
```

Checkpoints are written to `decoder_train/code/exp/<trial_name>/ckpt/`.
