## Introduction
Tumor-infiltrating lymphocytes (TILs), which are lymphocytes located within and around cancer cells, have been recognized as significant factors associated with the progression, prognosis, and therapy of various cancers, including breast cancer and non-small cell lung cancer. Concurrently, advancements in deep learning algorithms have substantially improved the effectiveness and precision of pathology image analysis. We conducted a search on Google Scholar using the terms ("machine learning" OR "artificial intelligence") AND "pancreatic cancer" AND "lymphocyte to tumor ratio", but no relevant literature was found. There is a pressing need for a fully automated model to evaluate the lymphocyte to tumor area ratio(LTR) in whole slide images (WSI) for clinical practice. Therefore,We developed a deep learning model for the comprehensive automated evaluation of LTR in HE-stained WSI, and validated its prognostic value in patient populations with PDAC. 


## Prerequisites
Python 3.10

environments can be `conda` create:

`conda env create -f environments.yaml`

or `pip` install:

`pip install -r requirements.txt`

or

Install PyTorch for your CUDA or CPU environment first, then install the remaining Python dependencies:

```bash
pip install numpy pillow matplotlib scikit-learn seaborn tqdm scipy openslide-python timm huggingface_hub
```

`8class-WSI-quantify.py` also requires the system OpenSlide library. Install it with your platform package manager before using `openslide-python`.

The CONCH, TITAN, and UNI workflows require their corresponding local model files. Place them under `models/` or pass explicit paths with command-line arguments.


## Expected Tile Classes

The tile training scripts expect an ImageFolder-style directory with these class folders:

```text
ADI BAC DEB LYM MUS NOR STR TUM
```

Example layout:

```text
data/
└── 8class_patch/
    ├── train/
    │   ├── ADI/
    │   ├── BAC/
    │   └── ...
    └── test/
        ├── ADI/
        ├── BAC/
        └── ...
```



## Training A Supervised Tile Classifier

```bash
python 8class-model-train.py \
  --mode train \
  --data-root data/8class_patch/train \
  --output outputs/model_out \
  --gpu-ids 0 \
  --model-arch vit_b_16
```

Evaluate a saved checkpoint:

```bash
python 8class-model-train.py \
  --mode test \
  --data-root data/8class_patch/test \
  --output outputs/model_eval \
  --model-arch vit_b_16 \
  --model-path checkpoints/8class_vit_b_16.pth \
  --gpu-ids 0
```

## Training CONCH/TITAN/UNI Probes

Run the default CONCH frozen encoder with an MLP head:

```bash
python 8class-model-train-for-lm.py \
  --encoder conch \
  --train-data-root data/8class_patch/train \
  --test-data-root data/8class_patch/test \
  --conch-model-dir models/CONCH \
  --output-dir outputs/conch-result \
  --device cuda:0
```

Run all three frozen encoders:

```bash
python 8class-model-train-for-lm.py \
  --encoder all \
  --train-data-root data/8class_patch/train \
  --test-data-root data/8class_patch/test \
  --conch-model-dir models/CONCH \
  --titan-model-dir models/TITAN \
  --uni-model-dir models/UNI2-h \
  --output-dir outputs/lm-mlp-result \
  --device cuda:0
```

Use `--head-type linear` to train a linear probe instead of the default MLP head.

## WSI Quantification

Run WSI inference with a trained ViT-B/16 checkpoint:

```bash
python 8class-WSI-quantify.py \
  --path slides/example.svs \
  --out outputs/wsi \
  --model_type vit_b_16 \
  --model checkpoints/8class_vit_b_16.pth \
  --gpu_id 0
```

Run WSI inference with a trained CONCH MLP probe:

```bash
python 8class-WSI-quantify.py \
  --path slides/example.svs \
  --out outputs/wsi \
  --model_type conch \
  --model checkpoints/conch_mlp_probe.pt \
  --conch_model_dir models/CONCH \
  --gpu_id 0
```

The WSI script writes visualization images and a metrics CSV unless `--no_save_vis` is set.

## Path Configuration

All local absolute paths were removed. You can configure paths with command-line arguments or environment variables:

- `WSI_DATA_ROOT`
- `WSI_OUTPUT_DIR`
- `WSI_TRAIN_ROOT`
- `WSI_TEST_ROOT`
- `RESNET50_MODEL_PATH`
- `VIT_B_16_MODEL_PATH`
- `CONCH_MODEL_DIR`
- `CONCH_HEAD_PATH`
- `TITAN_MODEL_DIR`
- `UNI_MODEL_DIR`


