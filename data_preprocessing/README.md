# Data Preprocessing

Input `.npy` dict format:

```python
{
  "coord": np.ndarray,  # (N, 3) float32
  "label": np.ndarray,  # (N,) or (N, 1) int
}
```

## Connectivity Split

Use DBSCAN to check whether each part is connected. Disconnected parts are split into multiple labels.

```bash
python split_disconnected_parts.py \
  --input /path/to/npy_dir \
  --output /path/to/output_dir
```

Outputs: `{id}_original.npy`, `{id}_split.npy`, `{id}_split_info.json`.

## PointNet Validator

Binary classifier: qualified (`Y=1`) / unqualified (`N=0`).  
Input features: `(10000, 4)` = `xyz + label`.

### Annotation File

Prepare `point_cloud_selections.json` before training. Keys are sample IDs:

- `selection`: `Y` = qualified, `N` = unqualified
- `data_path`: path to the corresponding `.npy` file

```json
{
  "sample_001": {
    "selection": "Y",
    "data_path": "/path/to/sample_001.npy"
  },
  "sample_002": {
    "selection": "N",
    "data_path": "/path/to/sample_002.npy"
  }
}
```

### Train

```bash
python pointnet_classifier.py \
  --selections point_cloud_selections.json \
  --model-path pointnet_classifier.pth
```

### Inference

```bash
python pointnet_inference.py \
  --model pointnet_classifier.pth \
  --data-dir /path/to/npy_dir \
  --output-dir inference_results
```
