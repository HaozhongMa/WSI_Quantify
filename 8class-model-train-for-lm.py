from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


ENCODER_CHOICES = ("conch", "titan", "uni")
DEFAULT_MODEL_DIRS = {
    "conch": os.environ.get("CONCH_MODEL_DIR", "models/CONCH"),
    "titan": os.environ.get("TITAN_MODEL_DIR", "models/TITAN"),
    "uni": os.environ.get("UNI_MODEL_DIR", "models/UNI2-h"),
}
DEFAULT_OUTPUT_DIRS = {
    "conch": "outputs/lm-mlp-result/conch",
    "titan": "outputs/lm-mlp-result/titan",
    "uni": "outputs/lm-mlp-result/uni",
}
DEFAULT_ALL_OUTPUT_ROOT = "outputs/lm-mlp-result"
DEFAULT_IMAGE_BATCH_SIZES = {
    "conch": 64,
    "titan": 16,
    "uni": 64,
}
DEFAULT_NORMALIZE_FEATURES = {
    "conch": False,
    "titan": False,
    "uni": True,
}
DEFAULT_TRAIN_ROOT = os.environ.get("WSI_TRAIN_ROOT", "data/8class_patch/train")
DEFAULT_TEST_ROOT = os.environ.get("WSI_TEST_ROOT", "data/8class_patch/test")
EXPECTED_CLASSES = ["ADI", "BAC", "DEB", "LYM", "MUS", "NOR", "STR", "TUM"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen CONCH/TITAN/UNI feature probes for 8-class tile evaluation."
    )
    parser.add_argument(
        "--encoder",
        choices=[*ENCODER_CHOICES, "all"],
        default="all",
        help="Encoder to run. Use 'all' to run CONCH, TITAN, and UNI sequentially.",
    )
    parser.add_argument("--conch-model-dir", default=DEFAULT_MODEL_DIRS["conch"])
    parser.add_argument("--titan-model-dir", default=DEFAULT_MODEL_DIRS["titan"])
    parser.add_argument("--uni-model-dir", default=DEFAULT_MODEL_DIRS["uni"])
    parser.add_argument("--train-data-root", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--test-data-root", default=DEFAULT_TEST_ROOT)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to the original per-encoder result directory "
            "for single encoder runs, or lm-mlp-result/{encoder} for --encoder all."
        ),
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help=(
            "Directory containing train/test feature caches to reuse. Defaults to the "
            "resolved output directory. With --encoder all, an explicit cache dir is "
            "split into cache-dir/{encoder}."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=None,
        help="Image batch size. Defaults are conch=64, titan=16, uni=64.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument(
        "--normalize-features",
        dest="normalize_features",
        action="store_true",
        help="Force L2 normalization of extracted features before probe training.",
    )
    parser.add_argument(
        "--no-normalize-features",
        dest="normalize_features",
        action="store_false",
        help="Disable feature normalization even when the encoder default enables it.",
    )
    parser.set_defaults(normalize_features=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-search-trials", type=int, default=5000)
    parser.add_argument("--split-local-search-max-iter", type=int, default=1000)
    parser.add_argument(
        "--smoke-samples-per-class",
        type=int,
        default=0,
        help="Use only the first N images per class from train and test roots.",
    )
    parser.add_argument(
        "--force-recompute-features",
        action="store_true",
        help="Ignore existing train/test feature caches in the output directory.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scan_image_folder(root: Path, limit_per_class: int = 0) -> tuple[list[tuple[str, int]], list[str], dict[str, int]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    class_names = sorted([entry.name for entry in root.iterdir() if entry.is_dir()])
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class order under {root}: {class_names}; expected {EXPECTED_CLASSES}"
        )
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    samples: list[tuple[str, int]] = []
    for class_name in class_names:
        class_dir = root / class_name
        paths = sorted(
            str(path)
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit_per_class > 0:
            paths = paths[:limit_per_class]
        samples.extend((path, class_to_idx[class_name]) for path in paths)

    if not samples:
        raise RuntimeError(f"No image files found under {root}")
    return samples, class_names, class_to_idx


def extract_group_from_tile_path(tile_path: str) -> str:
    filename = os.path.basename(tile_path)
    return filename.split(" [", 1)[0]


def strip_module_prefix(key: object) -> str:
    text = str(key)
    prefix = "module."
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def split_score_tuple(
    val_class_counts: np.ndarray,
    total_class_counts: np.ndarray,
    val_fraction: float,
) -> tuple[int, float, float, float]:
    safe_total_class_counts = np.maximum(total_class_counts, 1)
    val_class_fractions = val_class_counts / safe_total_class_counts
    class_errors = np.abs(val_class_fractions - val_fraction)
    total_fraction_error = abs((val_class_counts.sum() / total_class_counts.sum()) - val_fraction)
    missing_classes = int(np.sum(val_class_counts == 0))
    return (
        missing_classes,
        float(class_errors.max()),
        float(class_errors.mean()),
        float(total_fraction_error),
    )


def split_score_value(
    val_class_counts: np.ndarray,
    total_class_counts: np.ndarray,
    val_fraction: float,
) -> float:
    missing_classes, max_error, mean_error, total_error = split_score_tuple(
        val_class_counts,
        total_class_counts,
        val_fraction,
    )
    return missing_classes * 100.0 + max_error * 10.0 + mean_error + total_error


def split_score_values(
    candidate_counts: np.ndarray,
    total_class_counts: np.ndarray,
    val_fraction: float,
) -> np.ndarray:
    safe_total_class_counts = np.maximum(total_class_counts, 1)
    class_errors = np.abs((candidate_counts / safe_total_class_counts) - val_fraction)
    total_fraction_errors = np.abs(
        (candidate_counts.sum(axis=1) / total_class_counts.sum()) - val_fraction
    )
    missing_classes = np.sum(candidate_counts == 0, axis=1)
    return missing_classes * 100.0 + class_errors.max(axis=1) * 10.0 + class_errors.mean(axis=1) + total_fraction_errors


def improve_split_by_flips(
    initial_mask: np.ndarray,
    group_class_counts: np.ndarray,
    total_class_counts: np.ndarray,
    val_fraction: float,
    max_iter: int,
) -> np.ndarray:
    val_mask = initial_mask.copy()
    val_class_counts = group_class_counts[val_mask].sum(axis=0)
    current_score = split_score_value(val_class_counts, total_class_counts, val_fraction)

    for _ in range(max_iter):
        best_score = current_score
        best_action: str | None = None
        best_group_index: int | None = None

        add_indices = np.flatnonzero(~val_mask)
        if len(add_indices) > 0:
            add_counts = val_class_counts + group_class_counts[add_indices]
            add_scores = split_score_values(add_counts, total_class_counts, val_fraction)
            add_position = int(np.argmin(add_scores))
            add_group_index = int(add_indices[add_position])
            would_leave_no_train = int(np.sum(val_mask)) + 1 >= len(val_mask)
            if float(add_scores[add_position]) < best_score and not would_leave_no_train:
                best_score = float(add_scores[add_position])
                best_action = "add"
                best_group_index = add_group_index

        remove_indices = np.flatnonzero(val_mask)
        if len(remove_indices) > 0:
            remove_counts = val_class_counts - group_class_counts[remove_indices]
            remove_scores = split_score_values(remove_counts, total_class_counts, val_fraction)
            remove_position = int(np.argmin(remove_scores))
            remove_group_index = int(remove_indices[remove_position])
            would_leave_no_val = int(np.sum(val_mask)) - 1 <= 0
            if float(remove_scores[remove_position]) < best_score and not would_leave_no_val:
                best_score = float(remove_scores[remove_position])
                best_action = "remove"
                best_group_index = remove_group_index

        if best_action is None or best_group_index is None:
            break

        if best_action == "add":
            val_mask[best_group_index] = True
            val_class_counts += group_class_counts[best_group_index]
        else:
            val_mask[best_group_index] = False
            val_class_counts -= group_class_counts[best_group_index]
        current_score = best_score

    return val_mask


def build_grouped_split(
    samples: Sequence[tuple[str, int]],
    num_classes: int,
    val_fraction: float,
    split_search_trials: int,
    split_local_search_max_iter: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    paths = np.array([path for path, _ in samples])
    labels = np.array([label for _, label in samples])
    groups = np.array([extract_group_from_tile_path(path) for path in paths])

    unique_groups, group_ids = np.unique(groups, return_inverse=True)
    group_class_counts = np.zeros((len(unique_groups), num_classes), dtype=np.int64)
    for group_id, label in zip(group_ids, labels):
        group_class_counts[group_id, label] += 1

    total_class_counts = np.bincount(labels, minlength=num_classes)
    rng = np.random.default_rng(seed)
    best_score = None
    best_val_group_mask = None

    for _ in range(split_search_trials):
        val_group_mask = rng.random(len(unique_groups)) < val_fraction
        if not val_group_mask.any() or val_group_mask.all():
            continue
        val_class_counts = group_class_counts[val_group_mask].sum(axis=0)
        score = split_score_tuple(val_class_counts, total_class_counts, val_fraction)
        if best_score is None or score < best_score:
            best_score = score
            best_val_group_mask = val_group_mask.copy()

    if best_val_group_mask is None:
        raise RuntimeError("Failed to generate a valid group-level train/val split.")

    best_val_group_mask = improve_split_by_flips(
        best_val_group_mask,
        group_class_counts,
        total_class_counts,
        val_fraction,
        split_local_search_max_iter,
    )

    val_sample_mask = best_val_group_mask[group_ids]
    train_indices = np.flatnonzero(~val_sample_mask)
    val_indices = np.flatnonzero(val_sample_mask)

    train_groups = set(groups[train_indices])
    val_groups = set(groups[val_indices])
    overlap_groups = train_groups.intersection(val_groups)
    if overlap_groups:
        raise RuntimeError(f"Group leakage detected between train/val: {sorted(overlap_groups)[:5]}")

    log(
        "Split train={} tiles/{} groups, val={} tiles/{} groups".format(
            len(train_indices),
            len(train_groups),
            len(val_indices),
            len(val_groups),
        )
    )
    log(f"Train class counts: {np.bincount(labels[train_indices], minlength=num_classes).tolist()}")
    log(f"Val class counts: {np.bincount(labels[val_indices], minlength=num_classes).tolist()}")
    return train_indices.tolist(), val_indices.tolist()


class ImagePathDataset(Dataset):
    def __init__(self, samples: Sequence[tuple[str, int]], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(label), path


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, indices: Sequence[int]):
        self.features = features
        self.labels = labels
        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample_index = int(self.indices[index])
        feature = torch.from_numpy(np.array(self.features[sample_index], dtype=np.float32, copy=True))
        label = int(self.labels[sample_index])
        return feature, label


class ConchImageEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images, proj_contrast=False, normalize=False)


class TitanPatchEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def resolve_conch_weights(model_dir: Path) -> Path:
    if model_dir.is_file():
        return model_dir
    weights_path = model_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"CONCH model path must point to pytorch_model.bin or contain it: {model_dir}")
    return weights_path


def load_conch_model(model_dir: Path, device: torch.device):
    from conch.open_clip_custom import create_model_from_pretrained

    weights_path = resolve_conch_weights(model_dir)
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        str(weights_path),
    )
    model = ConchImageEncoder(model).to(device)
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            dummy_features = model(dummy)
    if isinstance(dummy_features, (tuple, list)):
        dummy_features = dummy_features[0]
    feature_dim = int(dummy_features.shape[1])
    if feature_dim <= 0:
        raise RuntimeError("Could not determine CONCH feature dimension.")

    model_info = {
        "encoder": "conch",
        "architecture": "conch_ViT-B-16",
        "weights_path": str(weights_path),
        "feature_dim": feature_dim,
        "feature_call": "encode_image(proj_contrast=False, normalize=False)",
    }
    log(f"Loaded CONCH from {weights_path}; feature_dim={feature_dim}")
    return model, preprocess, feature_dim, model_info


def resolve_titan_conch_weights(model_dir: Path) -> Path:
    weights_path = model_dir / "conch_v1_5_pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"TITAN model dir must contain conch_v1_5_pytorch_model.bin: {model_dir}")
    return weights_path


def load_titan_patch_encoder(model_dir: Path, device: torch.device):
    import huggingface_hub

    if not model_dir.is_dir():
        raise FileNotFoundError(f"TITAN model dir does not exist: {model_dir}")
    weights_path = resolve_titan_conch_weights(model_dir)

    def local_hf_hub_download(repo_id, filename, *args, **kwargs):  # noqa: ANN001
        local_path = model_dir / filename
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        return str(local_path)

    huggingface_hub.hf_hub_download = local_hf_hub_download

    package_name = "titan_local_eval"
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    from titan_local_eval.configuration_titan import TitanConfig
    from titan_local_eval.conch_v1_5 import build_conch

    config = TitanConfig.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    model, preprocess = build_conch(config.conch_config)
    model = TitanPatchEncoder(model).to(device)
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros((1, 3, 448, 448), dtype=torch.float32, device=device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            dummy_features = model(dummy)
    if isinstance(dummy_features, (tuple, list)):
        dummy_features = dummy_features[0]
    feature_dim = int(dummy_features.shape[1])
    if feature_dim <= 0:
        raise RuntimeError("Could not determine TITAN patch encoder feature dimension.")

    model_info = {
        "encoder": "titan",
        "architecture": "TITAN bundled CONCH v1.5 patch encoder",
        "weights_path": str(weights_path),
        "feature_dim": feature_dim,
        "feature_call": "titan.return_conch-equivalent patch_encoder(image)",
        "input_size": 448,
    }
    log(f"Loaded TITAN patch encoder from {weights_path}; feature_dim={feature_dim}")
    return model, preprocess, feature_dim, model_info


def load_uni_model(model_dir: Path, device: torch.device):
    import timm
    from timm.data import create_transform
    from timm.models.vision_transformer import VisionTransformer

    try:
        from timm.layers import SwiGLUPacked
    except ImportError:
        from timm.models.vision_transformer import SwiGLUPacked

    config_path = model_dir / "config.json"
    weights_path = model_dir / "pytorch_model.bin"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"UNI model dir must contain config.json and pytorch_model.bin: {model_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architecture = config.get("architecture", "vit_giant_patch14_224")
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected UNI weight file format: {weights_path}")
    state = OrderedDict((strip_module_prefix(key), value) for key, value in state.items())
    embed_dim = int(state["patch_embed.proj.weight"].shape[0])
    patch_size = int(state["patch_embed.proj.weight"].shape[-1])
    depth = max(int(key.split(".")[1]) for key in state if key.startswith("blocks.")) + 1
    num_heads = int(state["blocks.0.attn.qkv.weight"].shape[0] / (3 * embed_dim))
    reg_tokens = int(state.get("reg_token", torch.empty(1, 0, embed_dim)).shape[1])
    patch_count = int(state["pos_embed"].shape[1])
    img_size = int(round(patch_count ** 0.5) * patch_size)
    mlp_fc1_out = int(state["blocks.0.mlp.fc1.weight"].shape[0])
    mlp_ratio = mlp_fc1_out / float(embed_dim)
    init_values = 1e-5 if any(".ls1.gamma" in key for key in state) else None

    try:
        model = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=int(config.get("num_classes", 0)),
            global_pool=config.get("global_pool", "token"),
        )
        model_shape = tuple(model.patch_embed.proj.weight.shape)
        state_shape = tuple(state["patch_embed.proj.weight"].shape)
        if model_shape != state_shape:
            raise RuntimeError(f"timm default shape {model_shape} does not match UNI shape {state_shape}")
    except Exception as exc:  # noqa: BLE001
        log(f"Building explicit UNI2-h VisionTransformer because timm default is incompatible: {exc}")
        model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=int(config.get("num_classes", 0)),
            global_pool=config.get("global_pool", "token"),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            init_values=init_values,
            class_token=True,
            no_embed_class=True,
            reg_tokens=reg_tokens,
            mlp_layer=SwiGLUPacked,
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log(f"UNI load missing keys: {len(missing)}")
    if unexpected:
        log(f"UNI load unexpected keys: {len(unexpected)}")

    pretrained_cfg = config.get("pretrained_cfg", {})
    transform_cfg = {
        "input_size": tuple(pretrained_cfg.get("input_size", (3, 224, 224))),
        "interpolation": pretrained_cfg.get("interpolation", "bilinear"),
        "mean": tuple(pretrained_cfg.get("mean", (0.485, 0.456, 0.406))),
        "std": tuple(pretrained_cfg.get("std", (0.229, 0.224, 0.225))),
        "crop_pct": float(pretrained_cfg.get("crop_pct", 1.0)),
    }
    transform = create_transform(**transform_cfg, is_training=False)
    feature_dim = int(config.get("num_features", getattr(model, "num_features", 0) or embed_dim))
    if feature_dim <= 0:
        raise RuntimeError("Could not determine UNI feature dimension.")

    model = model.to(device)
    model.eval()
    model_info = {
        "encoder": "uni",
        "architecture": architecture,
        "config_path": str(config_path),
        "weights_path": str(weights_path),
        "feature_dim": feature_dim,
        "input_size": list(transform_cfg["input_size"]),
        "interpolation": transform_cfg["interpolation"],
        "mean": list(transform_cfg["mean"]),
        "std": list(transform_cfg["std"]),
        "crop_pct": transform_cfg["crop_pct"],
        "config": config,
    }
    log(f"Loaded UNI2-h from {model_dir}; architecture={architecture}; feature_dim={feature_dim}")
    return model, transform, feature_dim, model_info


def load_encoder_model(encoder: str, model_dir: Path, device: torch.device):
    if encoder == "conch":
        return load_conch_model(model_dir, device)
    if encoder == "titan":
        return load_titan_patch_encoder(model_dir, device)
    if encoder == "uni":
        return load_uni_model(model_dir, device)
    raise ValueError(f"Unsupported encoder: {encoder}")


def metadata_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return current == expected


def samples_digest(samples: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for path, label in samples:
        digest.update(str(path).encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def extract_or_load_features(
    name: str,
    samples: Sequence[tuple[str, int]],
    model: nn.Module,
    transform,
    device: torch.device,
    feature_dim: int,
    output_dir: Path,
    feature_cache_dir: Path,
    image_batch_size: int,
    num_workers: int,
    force_recompute: bool,
    normalize_features: bool,
    encoder: str,
    model_info: dict,
) -> np.ndarray:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{name}_features.npy"
    meta_path = output_dir / f"{name}_features_meta.json"
    cache_feature_path = feature_cache_dir / f"{name}_features.npy"
    cache_meta_path = feature_cache_dir / f"{name}_features_meta.json"
    expected_meta = {
        "name": name,
        "encoder": encoder,
        "feature_dim": feature_dim,
        "normalize_features": bool(normalize_features),
        "sample_count": len(samples),
        "samples_sha256": samples_digest(samples),
        "model_info": model_info,
    }
    if (
        not force_recompute
        and cache_feature_path.exists()
        and metadata_matches(cache_meta_path, expected_meta)
    ):
        log(f"Loading cached {encoder} {name} features from {cache_feature_path}")
        return np.load(cache_feature_path, mmap_mode="r")

    if (
        not force_recompute
        and feature_path.exists()
        and metadata_matches(meta_path, expected_meta)
    ):
        log(f"Loading cached {encoder} {name} features from {feature_path}")
        return np.load(feature_path, mmap_mode="r")

    log(f"Extracting {encoder} {name} features for {len(samples)} images")
    dataset = ImagePathDataset(samples, transform)
    loader = DataLoader(
        dataset,
        batch_size=image_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(samples), feature_dim),
    )

    offset = 0
    with torch.no_grad():
        for batch_index, (images, _labels, _paths) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            outputs = outputs.float()
            if normalize_features:
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
            batch_features = outputs.detach().cpu().numpy().astype(np.float32)
            features[offset : offset + len(batch_features)] = batch_features
            offset += len(batch_features)
            if batch_index % 20 == 0 or offset == len(samples):
                log(f"{encoder} {name} features: {offset}/{len(samples)}")

    features.flush()
    meta_path.write_text(json.dumps(expected_meta, ensure_ascii=False), encoding="utf-8")
    return np.load(feature_path, mmap_mode="r")


def build_probe_head(
    feature_dim: int,
    num_classes: int,
    args: argparse.Namespace,
) -> nn.Module:
    if args.head_type == "linear":
        return nn.Linear(feature_dim, num_classes)
    if args.head_type == "mlp":
        return nn.Sequential(
            nn.Linear(feature_dim, args.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(args.mlp_dropout),
            nn.Linear(args.mlp_hidden_dim, num_classes),
        )
    raise ValueError(f"Unsupported head type: {args.head_type}")


def evaluate_head(
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = FeatureDataset(features, labels, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true: list[int] = []
    y_pred: list[int] = []
    head.eval()
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            logits = head(batch_features)
            predicted = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            y_pred.extend(predicted)
            y_true.extend(batch_labels.numpy().tolist())
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def train_probe(
    train_features: np.ndarray,
    labels: np.ndarray,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    feature_dim: int,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    head = build_probe_head(feature_dim, num_classes, args).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_dataset = FeatureDataset(train_features, labels, train_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.feature_batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    best_state = None
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_labels.size(0)
            predicted = torch.argmax(logits, dim=1)
            correct += int((predicted == batch_labels).sum().item())
            total += int(batch_labels.size(0))

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_true, val_pred = evaluate_head(
            head,
            train_features,
            labels,
            val_indices,
            args.feature_batch_size,
            device,
        )
        val_macro_f1 = f1_score(
            val_true,
            val_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
        val_weighted_f1 = f1_score(
            val_true,
            val_pred,
            labels=list(range(num_classes)),
            average="weighted",
            zero_division=0,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_macro_f1": float(val_macro_f1),
            "val_weighted_f1": float(val_weighted_f1),
        }
        history.append(row)
        log(
            "Epoch {:03d}: train_loss={:.6f}, train_acc={:.4f}, val_macro_f1={:.6f}, val_weighted_f1={:.6f}".format(
                epoch,
                train_loss,
                train_acc,
                val_macro_f1,
                val_weighted_f1,
            )
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = float(val_macro_f1)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                log(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_val_macro_f1={best_macro_f1:.6f}")
                break

    if best_state is None:
        raise RuntimeError(f"{args.head_type} probe training did not produce a best checkpoint.")
    head.load_state_dict(best_state)
    return head, history


def save_labeled_matrix_csv(path: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_label\\pred_label"] + list(class_names))
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row.tolist())


def save_confusion_matrix_png(
    path: Path,
    matrix: np.ndarray,
    class_names: Sequence[str],
    title: str,
    fmt: str,
    cmap: str = "Blues",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Labels",
        ylabel="True Labels",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = float(matrix.max()) / 2.0 if matrix.size and matrix.max() > 0 else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = format(matrix[i, j], fmt)
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_predictions_csv(
    path: Path,
    samples: Sequence[tuple[str, int]],
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path", "true_label", "pred_label", "true_index", "pred_index"])
        for (sample_path, true_label), pred_label in zip(samples, y_pred):
            writer.writerow(
                [
                    sample_path,
                    class_names[int(true_label)],
                    class_names[int(pred_label)],
                    int(true_label),
                    int(pred_label),
                ]
            )


def save_history_csv(path: Path, history: Sequence[dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_evaluation_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
    prefix: str,
) -> dict[str, float]:
    label_indices = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_indices,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, labels=label_indices, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=label_indices, average="weighted", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    per_class_path = output_dir / f"{prefix}_per_class_metrics.csv"
    with per_class_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for class_name, p_value, r_value, f1_value, support_value in zip(
            class_names,
            precision,
            recall,
            f1,
            support,
        ):
            writer.writerow([class_name, p_value, r_value, f1_value, int(support_value)])

    summary_path = output_dir / f"{prefix}_summary_metrics.csv"
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["macro_f1", macro_f1])
        writer.writerow(["weighted_f1", weighted_f1])
        writer.writerow(["balanced_accuracy", balanced_acc])

    raw_cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    row_sums = raw_cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(
        raw_cm.astype(float),
        row_sums,
        out=np.zeros_like(raw_cm, dtype=float),
        where=row_sums != 0,
    )

    save_labeled_matrix_csv(output_dir / f"{prefix}_confusion_matrix_raw_counts.csv", raw_cm, class_names)
    save_labeled_matrix_csv(output_dir / f"{prefix}_confusion_matrix_normalized.csv", normalized_cm, class_names)
    save_confusion_matrix_png(
        output_dir / f"{prefix}_confusion_matrix_raw_counts.png",
        raw_cm,
        class_names,
        f"{prefix.capitalize()} Raw Count Confusion Matrix",
        "d",
    )
    save_confusion_matrix_png(
        output_dir / f"{prefix}_confusion_matrix_normalized.png",
        normalized_cm,
        class_names,
        f"{prefix.capitalize()} Normalized Confusion Matrix",
        ".3f",
    )

    metrics = {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_acc),
    }
    log(
        f"{prefix} metrics: "
        + "macro_f1={macro_f1:.6f}, weighted_f1={weighted_f1:.6f}, balanced_accuracy={balanced_accuracy:.6f}".format(
            **metrics
        )
    )
    return metrics


def selected_encoders(args: argparse.Namespace) -> list[str]:
    if args.encoder == "all":
        return list(ENCODER_CHOICES)
    return [args.encoder]


def resolve_model_dir(args: argparse.Namespace, encoder: str) -> Path:
    value = getattr(args, f"{encoder}_model_dir")
    return Path(value)


def resolve_output_dir(args: argparse.Namespace, encoder: str, multiple_encoders: bool) -> Path:
    if args.output_dir:
        base = Path(args.output_dir)
        return base / encoder if multiple_encoders else base
    if multiple_encoders:
        return Path(DEFAULT_ALL_OUTPUT_ROOT) / encoder
    return Path(DEFAULT_OUTPUT_DIRS[encoder])


def resolve_feature_cache_dir(args: argparse.Namespace, encoder: str, output_dir: Path, multiple_encoders: bool) -> Path:
    if not args.feature_cache_dir:
        return output_dir
    base = Path(args.feature_cache_dir)
    return base / encoder if multiple_encoders else base


def resolve_image_batch_size(args: argparse.Namespace, encoder: str) -> int:
    if args.image_batch_size is not None:
        return int(args.image_batch_size)
    return DEFAULT_IMAGE_BATCH_SIZES[encoder]


def resolve_normalize_features(args: argparse.Namespace, encoder: str) -> bool:
    if args.normalize_features is not None:
        return bool(args.normalize_features)
    return DEFAULT_NORMALIZE_FEATURES[encoder]


def run_single_encoder(
    encoder: str,
    args: argparse.Namespace,
    device: torch.device,
    train_samples: Sequence[tuple[str, int]],
    test_samples: Sequence[tuple[str, int]],
    class_names: Sequence[str],
    class_to_idx: dict[str, int],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    multiple_encoders: bool,
) -> None:
    set_seed(args.seed)
    output_dir = resolve_output_dir(args, encoder, multiple_encoders)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir = resolve_feature_cache_dir(args, encoder, output_dir, multiple_encoders)
    image_batch_size = resolve_image_batch_size(args, encoder)
    normalize_features = resolve_normalize_features(args, encoder)
    model_dir = resolve_model_dir(args, encoder)

    log("=" * 80)
    log(f"Starting encoder={encoder}, head_type={args.head_type}, output_dir={output_dir}")
    log(f"model_dir={model_dir}")
    log(f"image_batch_size={image_batch_size}, normalize_features={normalize_features}")

    model, transform, feature_dim, model_info = load_encoder_model(encoder, model_dir, device)
    train_features = extract_or_load_features(
        "train",
        train_samples,
        model,
        transform,
        device,
        feature_dim,
        output_dir,
        feature_cache_dir,
        image_batch_size,
        args.num_workers,
        args.force_recompute_features,
        normalize_features,
        encoder,
        model_info,
    )
    test_features = extract_or_load_features(
        "test",
        test_samples,
        model,
        transform,
        device,
        feature_dim,
        output_dir,
        feature_cache_dir,
        image_batch_size,
        args.num_workers,
        args.force_recompute_features,
        normalize_features,
        encoder,
        model_info,
    )

    train_labels = np.array([label for _, label in train_samples], dtype=np.int64)
    test_labels = np.array([label for _, label in test_samples], dtype=np.int64)
    num_classes = len(class_names)

    head, history = train_probe(
        train_features,
        train_labels,
        train_indices,
        val_indices,
        feature_dim,
        num_classes,
        args,
        device,
    )

    checkpoint = {
        "state_dict": head.state_dict(),
        "encoder": encoder,
        "model_dir": str(model_dir),
        "model_info": model_info,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "class_to_idx": class_to_idx,
        "head_type": args.head_type,
        "mlp_hidden_dim": args.mlp_hidden_dim if args.head_type == "mlp" else None,
        "mlp_dropout": args.mlp_dropout if args.head_type == "mlp" else None,
        "normalize_features": bool(normalize_features),
    }
    checkpoint_path = output_dir / f"{args.head_type}_probe.pt"
    torch.save(checkpoint, checkpoint_path)
    save_history_csv(output_dir / "train_history.csv", history)

    val_samples = [train_samples[int(index)] for index in val_indices]
    val_true, val_pred = evaluate_head(
        head,
        train_features,
        train_labels,
        val_indices,
        args.feature_batch_size,
        device,
    )
    save_predictions_csv(output_dir / "val_predictions.csv", val_samples, val_pred, class_names)
    val_metrics = save_evaluation_outputs(val_true, val_pred, class_names, output_dir, "val")

    test_true, test_pred = evaluate_head(
        head,
        test_features,
        test_labels,
        list(range(len(test_samples))),
        args.feature_batch_size,
        device,
    )
    save_predictions_csv(output_dir / "test_predictions.csv", test_samples, test_pred, class_names)
    test_metrics = save_evaluation_outputs(test_true, test_pred, class_names, output_dir, "test")

    run_config = {
        "args": vars(args),
        "encoder": encoder,
        "model_dir": str(model_dir),
        "model_info": model_info,
        "class_names": list(class_names),
        "class_to_idx": class_to_idx,
        "feature_dim": feature_dim,
        "train_image_count": len(train_samples),
        "test_image_count": len(test_samples),
        "train_indices_count": len(train_indices),
        "val_indices_count": len(val_indices),
        "feature_cache_dir": str(feature_cache_dir),
        "image_batch_size": image_batch_size,
        "normalize_features": bool(normalize_features),
        "head_type": args.head_type,
        "train_support": np.bincount(train_labels[train_indices], minlength=num_classes).astype(int).tolist(),
        "val_support": np.bincount(train_labels[val_indices], minlength=num_classes).astype(int).tolist(),
        "test_support": np.bincount(test_labels, minlength=num_classes).astype(int).tolist(),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"Saved {encoder} outputs to {output_dir}")

    del model, head, train_features, test_features
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    encoders = selected_encoders(args)
    multiple_encoders = len(encoders) > 1
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    log(f"Using device: {device}")
    log(f"Selected encoders: {encoders}")

    train_samples, class_names, class_to_idx = scan_image_folder(
        Path(args.train_data_root),
        args.smoke_samples_per_class,
    )
    test_samples, test_class_names, test_class_to_idx = scan_image_folder(
        Path(args.test_data_root),
        args.smoke_samples_per_class,
    )
    if class_names != test_class_names or class_to_idx != test_class_to_idx:
        raise ValueError(f"Train/test class mismatch: {class_names} vs {test_class_names}")
    num_classes = len(class_names)
    log(f"Classes: {class_names}")
    log(f"Train images: {len(train_samples)}; test images: {len(test_samples)}")

    train_indices, val_indices = build_grouped_split(
        train_samples,
        num_classes,
        args.val_fraction,
        args.split_search_trials,
        args.split_local_search_max_iter,
        args.seed,
    )

    for encoder in encoders:
        run_single_encoder(
            encoder,
            args,
            device,
            train_samples,
            test_samples,
            class_names,
            class_to_idx,
            train_indices,
            val_indices,
            multiple_encoders,
        )


if __name__ == "__main__":
    main()
