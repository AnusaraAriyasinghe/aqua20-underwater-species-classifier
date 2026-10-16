# AQUA20 Underwater Species Classifier

My individual contribution to a group machine-learning project (SLIIT, AIML module, 2026), continued here
as a personal project. The task is classifying 20 underwater species in the
[AQUA20](https://huggingface.co/datasets/taufiktrf/aqua20) dataset.

## My contribution (member IT25101088)

| Part | File |
|---|---|
| Preprocessing: CLAHE colour correction | `notebooks/preprocessing/IT25101088_Color_Correction.ipynb` |
| Model: custom CNN built from scratch (TensorFlow / Keras) | `notebooks/models/IT25101088_CNN.ipynb` |
| Write-ups | `docs/03_color_correction.md`, `docs/11_cnn.md` |
| Results | `results/model_figures/IT25101088_*`, `results/outputs/*/IT25101088.csv` |

### Module result (baseline)

Best variety: 96x96 px, data augmentation, 3 conv blocks, balanced class weights.

| Metric | Test set |
|---|---|
| Macro F1 | 0.171 |
| Accuracy | 0.360 |

The network was still underfitting (about 27% training accuracy) when it stopped: a small CNN trained from
scratch on a CPU with about 5.5k images. See `docs/11_cnn.md` for the full analysis.

## Shared group code

The files in `scripts/` (data loading, preprocessing, the train/validation split, evaluation helpers) and
`data/splits/split_indices.csv` were written by the group together. They are included unchanged so the
notebooks run and use exactly the same data split as the module project. `docs/00_data_utils.md` and
`docs/08_model_shared_code.md` describe them.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

The dataset is downloaded into `data/raw/` on the first run (not committed). Preprocessed image caches are
written to `data/cache/` (not committed). Run the notebooks from inside their own folders.

## Next steps

- Transfer learning: fine-tune an ImageNet-pretrained backbone (e.g. EfficientNet) at 224 px on a GPU.
- Better imbalance handling for the rare species.
- Targeted external data for classes that remain weak (training set only).
