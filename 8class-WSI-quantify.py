import openslide
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import argparse
import csv
import scipy.ndimage

# Optimized version generated from 8class-WSI-quantify.py.
# v2: fail-safe tissue prefilter; automatically falls back when low-resolution levels are unavailable or unreadable.

# --- Constants ---
class_map = {
    'Adipose': 0, 'Background': 1, 'Debris': 2, 'Lymphocytes': 3,
    'Muscle': 4, 'Normal': 5, 'Stroma': 6, 'Tumour': 7,
    'Tumor_Relate_Lymphocytes': 8
}

label_to_name = {v: k for k, v in class_map.items()}

EXPECTED_CONCH_CLASSES = ["ADI", "BAC", "DEB", "LYM", "MUS", "NOR", "STR", "TUM"]
DEFAULT_RESNET50_MODEL_PATH = os.environ.get("RESNET50_MODEL_PATH", "checkpoints/8class_resnet50.pth")
DEFAULT_CONCH_MODEL_DIR = os.environ.get("CONCH_MODEL_DIR", "models/CONCH")
DEFAULT_CONCH_HEAD_PATH = os.environ.get("CONCH_HEAD_PATH", "checkpoints/conch_mlp_probe.pt")
DEFAULT_VIT_B_16_MODEL_PATH = os.environ.get("VIT_B_16_MODEL_PATH", "checkpoints/8class_vit_b_16.pth")

LABEL_COLORS = {
    0: (246, 227, 109),    # Adipose
    1: (255, 255, 255),    # Background
    2: (27, 121, 175),     # Debris
    8: (164, 53, 219),     # Tumor_Relate_Lymphocytes
    4: (202, 234, 194),    # Muscle
    5: (187, 225, 115),    # Normal
    6: (101, 194, 164),    # Stroma
    7: (227, 27, 30),      # Tumour
    3: (211, 186, 217)     # Lymphocytes
}

# --- Model Definition ---
class ModifiedResNet50(models.ResNet):
    def __init__(self, num_classes=8, dropout_probability=0.5):
        super(ModifiedResNet50, self).__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3])
        self.fc = nn.Sequential(
            nn.Dropout(dropout_probability),
            nn.Linear(self.fc.in_features, num_classes)
        )


class ConchImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.encode_image(images, proj_contrast=False, normalize=False)


def build_mlp_head(feature_dim, num_classes, hidden_dim=512, dropout=0.2):
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def resolve_conch_weights(model_dir):
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"CONCH weights not found: {weights_path}")
    return weights_path


def load_resnet50_classifier(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ResNet50 checkpoint not found: {model_path}")

    model = ModifiedResNet50(num_classes=8)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded ResNet50 classifier: checkpoint={model_path}")
    return model


def normalize_state_dict_for_model(state_dict, model):
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model_is_parallel = isinstance(model, nn.DataParallel)
    has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())

    if model_is_parallel and not has_module_prefix:
        return OrderedDict((f"module.{key}", value) for key, value in state_dict.items())
    if not model_is_parallel and has_module_prefix:
        return OrderedDict((key.replace("module.", "", 1), value) for key, value in state_dict.items())
    return state_dict


def load_vit_b_16_classifier(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ViT-B/16 checkpoint not found: {model_path}")

    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 8)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = normalize_state_dict_for_model(checkpoint, model)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded ViT-B/16 classifier: checkpoint={model_path}")
    return model


def load_conch_classifier(conch_model_dir, head_path, device):
    from conch.open_clip_custom import create_model_from_pretrained

    if not os.path.exists(head_path):
        raise FileNotFoundError(f"CONCH MLP head checkpoint not found: {head_path}")

    conch_weights_path = resolve_conch_weights(conch_model_dir)
    conch_model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        conch_weights_path,
    )
    encoder = ConchImageEncoder(conch_model).to(device)
    encoder.eval()

    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    feature_dim = int(checkpoint.get("feature_dim", state_dict["0.weight"].shape[1]))
    hidden_dim = int(checkpoint.get("mlp_hidden_dim", state_dict["0.weight"].shape[0]))
    num_classes = int(checkpoint.get("num_classes", state_dict["3.weight"].shape[0]))
    dropout = float(checkpoint.get("mlp_dropout", 0.2))
    class_names = checkpoint.get("class_names", EXPECTED_CONCH_CLASSES)
    if list(class_names) != EXPECTED_CONCH_CLASSES:
        raise ValueError(f"Unexpected CONCH class order: {class_names}; expected {EXPECTED_CONCH_CLASSES}")
    if num_classes != 8:
        raise ValueError(f"Expected an 8-class CONCH head, got num_classes={num_classes}")

    head = build_mlp_head(feature_dim, num_classes, hidden_dim, dropout).to(device)
    head.load_state_dict(state_dict)
    head.eval()
    normalize_features = bool(checkpoint.get("normalize_features", False))

    print(
        "Loaded CONCH classifier: "
        f"encoder={conch_weights_path}, head={head_path}, "
        f"feature_dim={feature_dim}, hidden_dim={hidden_dim}, "
        f"normalize_features={normalize_features}"
    )
    return encoder, head, preprocess, normalize_features

# --- Optimized Dataset for Parallel I/O ---
class WSIPatchDataset(Dataset):
    def __init__(self, slide_path, coords, tile_size=224, preprocess=None, exact_white_filter=False, white_threshold=230, white_ratio=0.90):
        self.slide_path = slide_path
        self.coords = coords
        self.tile_size = tile_size
        self.preprocess = preprocess or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.exact_white_filter = exact_white_filter
        self.white_threshold = white_threshold
        self.white_ratio = white_ratio
        self.dummy_tensor = self.preprocess(Image.new("RGB", (tile_size, tile_size), (255, 255, 255)))
        self._slide = None  # Worker-specific slide handle.

    def _get_slide(self):
        # OpenSlide objects cannot be pickled, so each worker needs its own instance.
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)
        return self._slide

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        slide = self._get_slide()
        img = slide.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')

        if self.exact_white_filter:
            # Optional exact white check. This preserves the original behavior but costs CPU time.
            img_np = np.asarray(img)
            white_ratio = np.mean(np.all(img_np >= self.white_threshold, axis=-1))
            if white_ratio > self.white_ratio:
                return torch.zeros_like(self.dummy_tensor), 1, x, y
            return self.preprocess(img), 0, x, y

        return self.preprocess(img), x, y

# --- Core Processing ---
def _parse_float_property(properties, keys):
    for key in keys:
        value = properties.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None

def get_slide_mpp(slide):
    mpp_x_keys = [
        getattr(openslide, 'PROPERTY_NAME_MPP_X', 'openslide.mpp-x'),
        'openslide.mpp-x',
        'aperio.MPP',
    ]
    mpp_y_keys = [
        getattr(openslide, 'PROPERTY_NAME_MPP_Y', 'openslide.mpp-y'),
        'openslide.mpp-y',
        'aperio.MPP',
    ]
    mpp_x = _parse_float_property(slide.properties, mpp_x_keys)
    mpp_y = _parse_float_property(slide.properties, mpp_y_keys)

    if mpp_x is not None and mpp_y is not None:
        return (mpp_x + mpp_y) / 2.0
    if mpp_x is not None:
        return mpp_x
    return mpp_y


def build_all_coords(width, height, tile_size, step):
    n_rows = (height - tile_size) // step + 1
    n_cols = (width - tile_size) // step + 1
    coords = [(c * step, r * step) for r in range(n_rows) for c in range(n_cols)]
    return coords, n_rows, n_cols


def build_tissue_coords(
    slide,
    tile_size,
    step,
    mask_level=-1,
    white_threshold=230,
    tissue_ratio_threshold=0.05,
    close_radius=2,
    min_downsample=4.0,
    max_mask_pixels=25_000_000,
):
    """
    Fast low-resolution tissue prefilter.

    This function is intentionally fail-safe: if the slide has no usable
    downsampled level, if the chosen level is too large, or if OpenSlide fails
    to read that level, it returns all grid coordinates instead of crashing.
    That keeps the script usable for SVS/TIF/MRXS files with incomplete pyramid
    metadata or problematic low-resolution levels.
    """
    width, height = slide.level_dimensions[0]
    all_coords, n_rows, n_cols = build_all_coords(width, height, tile_size, step)

    try:
        level_count = int(slide.level_count)
        if level_count <= 1 and mask_level != 0:
            print(
                "Tissue prefilter skipped: slide has no downsampled pyramid level. "
                "Processing all grid tiles instead."
            )
            return all_coords, n_rows, n_cols

        if mask_level < 0:
            # Choose the first level whose downsample is large enough.
            # This avoids accidentally reading the whole level-0 WSI as a mask.
            chosen_level = None
            for level_idx in range(1, level_count):
                if float(slide.level_downsamples[level_idx]) >= float(min_downsample):
                    chosen_level = level_idx
                    break
            if chosen_level is None:
                print(
                    f"Tissue prefilter skipped: no level with downsample >= {min_downsample}. "
                    "Processing all grid tiles instead."
                )
                return all_coords, n_rows, n_cols
            mask_level = chosen_level
        else:
            mask_level = max(0, min(int(mask_level), level_count - 1))

        low_w, low_h = slide.level_dimensions[mask_level]
        downsample = float(slide.level_downsamples[mask_level])

        if mask_level == 0 and downsample < float(min_downsample):
            print(
                "Tissue prefilter skipped: selected mask level is level 0, "
                "which would require reading the full-resolution WSI. "
                "Processing all grid tiles instead."
            )
            return all_coords, n_rows, n_cols

        mask_pixels = int(low_w) * int(low_h)
        if mask_pixels > int(max_mask_pixels):
            print(
                "Tissue prefilter skipped: selected mask level is too large "
                f"({low_w}x{low_h}={mask_pixels:,} pixels > {max_mask_pixels:,}). "
                "Processing all grid tiles instead."
            )
            return all_coords, n_rows, n_cols

        low_img = slide.read_region((0, 0), mask_level, (low_w, low_h)).convert("RGB")
        low_np = np.asarray(low_img)

        # Tissue is anything not close to white in at least one channel.
        tissue_mask = np.any(low_np < white_threshold, axis=-1)

        if close_radius > 0 and tissue_mask.any():
            structure = np.ones((2 * close_radius + 1, 2 * close_radius + 1), dtype=bool)
            tissue_mask = scipy.ndimage.binary_closing(tissue_mask, structure=structure)
            tissue_mask = scipy.ndimage.binary_dilation(tissue_mask, structure=structure)

        coords = []
        for x, y in all_coords:
            x0 = max(0, int(x / downsample))
            y0 = max(0, int(y / downsample))
            x1 = min(low_w, int(np.ceil((x + tile_size) / downsample)))
            y1 = min(low_h, int(np.ceil((y + tile_size) / downsample)))
            patch_mask = tissue_mask[y0:y1, x0:x1]
            if patch_mask.size > 0 and float(patch_mask.mean()) >= tissue_ratio_threshold:
                coords.append((x, y))

        skipped = len(all_coords) - len(coords)
        print(
            "Tissue prefilter: "
            f"mask_level={mask_level}, downsample={downsample:.1f}, "
            f"kept={len(coords)}/{len(all_coords)} tiles, skipped={skipped} blank-like tiles"
        )
        return coords, n_rows, n_cols

    except Exception as exc:
        print(
            "Tissue prefilter failed and was skipped: "
            f"{type(exc).__name__}: {exc}. Processing all grid tiles instead."
        )
        return all_coords, n_rows, n_cols


def make_dataloader(dataset, batch_size, num_workers, pin_memory, prefetch_factor):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def run_model_forward(model_type, imgs, model, encoder, head, normalize_features):
    if model_type == "conch":
        features = encoder(imgs)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = features.float()
        if normalize_features:
            features = F.normalize(features, p=2, dim=1)
        return head(features)
    return model(imgs)


def generate_label_matrix(
    slide_path,
    model_type,
    model_path,
    conch_model_dir,
    tile_size,
    step,
    gpu_id,
    batch_size,
    num_workers=8,
    prefetch_factor=4,
    use_tissue_prefilter=True,
    tissue_mask_level=-1,
    tissue_white_threshold=230,
    tissue_ratio_threshold=0.05,
    tissue_close_radius=2,
    tissue_min_downsample=4.0,
    tissue_max_mask_pixels=25_000_000,
    exact_white_filter=False,
    exact_white_ratio=0.90,
    amp=True,
):
    """
    Runs inference and returns a 2D matrix of labels representing the WSI.
    Speed optimizations:
    1) low-resolution tissue prefilter to reduce level-0 reads;
    2) pinned memory + non_blocking GPU transfer;
    3) torch.inference_mode;
    4) AMP for all supported model backends;
    5) configurable dataloader workers/prefetch.
    """
    if step <= 0:
        raise ValueError("step must be positive.")

    slide = openslide.OpenSlide(slide_path)
    w, h = slide.level_dimensions[0]
    slide_mpp = get_slide_mpp(slide)

    if use_tissue_prefilter:
        coords, n_rows, n_cols = build_tissue_coords(
            slide,
            tile_size=tile_size,
            step=step,
            mask_level=tissue_mask_level,
            white_threshold=tissue_white_threshold,
            tissue_ratio_threshold=tissue_ratio_threshold,
            close_radius=tissue_close_radius,
            min_downsample=tissue_min_downsample,
            max_mask_pixels=tissue_max_mask_pixels,
        )
    else:
        coords, n_rows, n_cols = build_all_coords(w, h, tile_size, step)
        print(f"Tissue prefilter disabled: processing all {len(coords)} tiles")

    label_matrix = np.full((n_rows, n_cols), class_map['Background'], dtype=np.int8)
    if len(coords) == 0:
        print("No tissue tiles found. Returning all-background label matrix.")
        return label_matrix, (w, h), step, slide_mpp

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = None
    encoder = None
    head = None
    normalize_features = False

    if model_type == "conch":
        encoder, head, preprocess, normalize_features = load_conch_classifier(
            conch_model_dir,
            model_path,
            device,
        )
    elif model_type == "resnet50":
        model = load_resnet50_classifier(model_path, device)
        preprocess = None
    elif model_type == "vit_b_16":
        model = load_vit_b_16_classifier(model_path, device)
        preprocess = None
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    dataset = WSIPatchDataset(
        slide_path,
        coords,
        tile_size=tile_size,
        preprocess=preprocess,
        exact_white_filter=exact_white_filter,
        white_threshold=tissue_white_threshold,
        white_ratio=exact_white_ratio,
    )
    pin_memory = device.type == "cuda"
    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    print(
        f"Start inference: tiles={len(coords)}, batch_size={batch_size}, "
        f"num_workers={num_workers}, amp={amp and device.type == 'cuda'}, exact_white_filter={exact_white_filter}"
    )

    with torch.inference_mode():
        if exact_white_filter:
            for imgs, is_white_flags, xs, ys in tqdm(loader, desc="Inference"):
                valid_mask = is_white_flags == 0
                if not valid_mask.any():
                    continue
                imgs = imgs[valid_mask].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    outputs = run_model_forward(model_type, imgs, model, encoder, head, normalize_features)
                preds = outputs.argmax(1).cpu().numpy()
                row_indices = ys[valid_mask].numpy() // step
                col_indices = xs[valid_mask].numpy() // step
                label_matrix[row_indices, col_indices] = preds
        else:
            for imgs, xs, ys in tqdm(loader, desc="Inference"):
                imgs = imgs.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    outputs = run_model_forward(model_type, imgs, model, encoder, head, normalize_features)
                preds = outputs.argmax(1).cpu().numpy()
                row_indices = ys.numpy() // step
                col_indices = xs.numpy() // step
                label_matrix[row_indices, col_indices] = preds

    try:
        slide.close()
    except Exception:
        pass
    return label_matrix, (w, h), step, slide_mpp

def remove_small_components(binary_mask, min_size):
    if min_size <= 1 or not binary_mask.any():
        return binary_mask

    component_matrix, component_count = scipy.ndimage.label(binary_mask)
    if component_count == 0:
        return binary_mask

    component_sizes = np.bincount(component_matrix.ravel())
    keep_components = component_sizes >= min_size
    keep_components[0] = False
    return keep_components[component_matrix]

def define_tumor_related_lymphocytes(
    label_matrix,
    step,
    mpp,
    radius_um=500.0,
    min_tumor_component_tiles=20,
    tumor_closing_radius=1,
):
    """
    Defines Tumor_Relate_Lymphocytes by physical proximity to reliable Tumour regions.
    """
    if mpp <= 0:
        raise ValueError("mpp must be positive.")
    if radius_um <= 0:
        raise ValueError("radius_um must be positive.")

    new_matrix = label_matrix.copy()
    new_matrix[new_matrix == class_map['Tumor_Relate_Lymphocytes']] = class_map['Lymphocytes']

    tumor_mask = new_matrix == class_map['Tumour']
    lymphocyte_mask = new_matrix == class_map['Lymphocytes']
    lymphocyte_count = int(np.sum(lymphocyte_mask))

    reliable_tumor_mask = remove_small_components(tumor_mask, min_tumor_component_tiles)
    if tumor_closing_radius > 0 and reliable_tumor_mask.any():
        kernel_size = 2 * tumor_closing_radius + 1
        structure = np.ones((kernel_size, kernel_size), dtype=bool)
        reliable_tumor_mask = scipy.ndimage.binary_closing(reliable_tumor_mask, structure=structure)
        reliable_tumor_mask = remove_small_components(reliable_tumor_mask, min_tumor_component_tiles)

    reliable_tumor_count = int(np.sum(reliable_tumor_mask))
    if reliable_tumor_count == 0:
        print("No reliable Tumour region found; Tumor_Relate_Lymphocytes count is 0.")
        return new_matrix

    step_um = step * mpp
    radius_steps = max(1, int(np.ceil(radius_um / step_um)))
    distance_to_tumor_steps = scipy.ndimage.distance_transform_edt(~reliable_tumor_mask)
    tumor_related_mask = lymphocyte_mask & (distance_to_tumor_steps <= radius_steps)

    new_matrix[tumor_related_mask] = class_map['Tumor_Relate_Lymphocytes']

    tumor_related_count = int(np.sum(tumor_related_mask))
    print(
        "Tumor_Relate_Lymphocytes: "
        f"{tumor_related_count}/{lymphocyte_count} Lymphocytes within "
        f"{radius_um:.1f} um ({radius_steps} grid steps) of reliable Tumour; "
        f"mpp={mpp:.4f}, step_um={step_um:.2f}, reliable_tumor_tiles={reliable_tumor_count}"
    )
    return new_matrix

def create_vis_from_matrix(label_matrix, original_dim, step, scale_percent=10):
    """
    Vectorized visualization. 
    Map label indices directly to RGB colors using numpy fancy indexing.
    """
    h_grid, w_grid = label_matrix.shape
    
    # 1. Create a color palette array (max_label + 1, 3)
    max_label = max(LABEL_COLORS.keys())
    palette = np.zeros((max_label + 1, 3), dtype=np.uint8)
    for lbl, color in LABEL_COLORS.items():
        palette[lbl] = color
        
    # 2. Map matrix to RGB image instantly
    # This replaces the loop of drawing rectangles
    vis_array = palette[label_matrix] 
    
    # 3. Convert to Image
    vis_image = Image.fromarray(vis_array)
    
    # 4. Resize to target size
    # Note: label_matrix is already "downsampled" by step size.
    # Original logic: Image size = Original Dim / Scale
    target_w = original_dim[0] // scale_percent
    target_h = original_dim[1] // scale_percent
    
    # Use Nearest neighbor to keep sharp squares, or Bilinear/Lanczos for smooth look
    # Usually for tile maps, Nearest is more accurate to the data
    vis_image = vis_image.resize((target_w, target_h), Image.Resampling.NEAREST)
    
    return vis_image


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0


def build_count_metrics(wsi_name, model_type, counts_dict):
    class_counts = {
        class_name: int(counts_dict.get(class_idx, 0))
        for class_name, class_idx in class_map.items()
    }
    stroma_count = class_counts["Stroma"]
    lymphocytes_count = class_counts["Lymphocytes"]
    tumor_lymphocyte_count = class_counts["Tumor_Relate_Lymphocytes"]

    lsr = safe_divide(
        lymphocytes_count + tumor_lymphocyte_count,
        stroma_count + lymphocytes_count + tumor_lymphocyte_count,
    )
    new_lsr = safe_divide(
        tumor_lymphocyte_count,
        stroma_count + tumor_lymphocyte_count,
    )

    row = {
        "WSI_name": wsi_name,
        "model_type": model_type,
        "total_count": sum(class_counts.values()),
    }
    for class_name in class_map:
        row[f"{class_name}_count"] = class_counts[class_name]
    row["LSR"] = lsr
    row["NEW_LSR"] = new_lsr
    return row


def save_metrics_csv(metrics_path, metrics_row):
    fieldnames = [
        "WSI_name",
        "model_type",
        "total_count",
        *[f"{class_name}_count" for class_name in class_map],
        "LSR",
        "NEW_LSR",
    ]
    with open(metrics_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics_row)


def main():
    parser = argparse.ArgumentParser(description='Optimized WSI Processing')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='vit_b_16', choices=['resnet50', 'resnet5', 'conch', 'vit_b_16'],
                        help='Tile classifier backend. "resnet5" is accepted as an alias for resnet50.')
    parser.add_argument('--model', type=str, default=None,
                        help='Checkpoint path for the selected model_type. Defaults to the ViT-B/16 checkpoint, CONCH MLP head, or ResNet50 checkpoint.')
    parser.add_argument('--conch_model_dir', type=str, default=DEFAULT_CONCH_MODEL_DIR,
                        help='Directory containing local CONCH pytorch_model.bin weights.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Inference batch size. Defaults to 256 for all model types.')
    parser.add_argument('--step', '--step_size', dest='step_size', type=int, default=None,
                        help='Pixel stride between adjacent tiles. Default is tile_size // 2.')
    parser.add_argument('--trl_radius_um', type=float, default=500.0,
                        help='Physical distance from Tumour used to define Tumor_Relate_Lymphocytes.')
    parser.add_argument('--mpp', type=float, default=None,
                        help='Override microns-per-pixel. If omitted, read from slide metadata.')
    parser.add_argument('--default_mpp', type=float, default=0.25,
                        help='Fallback microns-per-pixel when slide metadata has no MPP.')
    parser.add_argument('--min_tumor_component_tiles', type=int, default=20,
                        help='Remove Tumour components smaller than this many grid tiles before TRL assignment.')
    parser.add_argument('--tumor_closing_radius', type=int, default=1,
                        help='Morphological closing radius, in grid steps, for reliable Tumour mask.')
    parser.add_argument('--tile_size', type=int, default=224,
                        help='Tile size in pixels. Keep 224 unless the model was trained with another size.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of DataLoader workers for parallel OpenSlide reading.')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Number of batches prefetched per worker when num_workers > 0.')
    parser.add_argument('--disable_tissue_prefilter', action='store_true',
                        help='Disable low-resolution tissue prefilter and process all grid tiles.')
    parser.add_argument('--tissue_mask_level', type=int, default=-1,
                        help='OpenSlide level used for tissue prefilter. -1 chooses automatically.')
    parser.add_argument('--tissue_white_threshold', type=int, default=230,
                        help='RGB threshold used by tissue prefilter; pixels below this are considered tissue-like.')
    parser.add_argument('--tissue_ratio_threshold', type=float, default=0.05,
                        help='Minimum low-resolution tissue fraction required to keep a tile.')
    parser.add_argument('--tissue_close_radius', type=int, default=2,
                        help='Morphological close/dilation radius for low-resolution tissue mask.')
    parser.add_argument('--tissue_min_downsample', type=float, default=4.0,
                        help='Minimum downsample required for using a pyramid level as tissue mask. Prevents reading full-resolution WSI as mask.')
    parser.add_argument('--tissue_max_mask_pixels', type=int, default=25000000,
                        help='Maximum number of pixels allowed for the tissue-mask level. Larger mask levels are skipped safely.')
    parser.add_argument('--exact_white_filter', action='store_true',
                        help='Run the original exact white-tile check after reading selected level-0 patches. More faithful but slower.')
    parser.add_argument('--exact_white_ratio', type=float, default=0.90,
                        help='White-pixel ratio threshold for --exact_white_filter.')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP mixed precision inference.')
    parser.add_argument('--no_save_vis', action='store_true',
                        help='Skip saving the visualization JPG. Useful when only metrics are needed.')
    parser.add_argument('--vis_scale_percent', type=int, default=10,
                        help='Visualization output size divisor: original_dim / vis_scale_percent.')
    parser.add_argument('--jpg_quality', type=int, default=90,
                        help='JPEG quality for visualization output.')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    pic_path = args.path
    WSI_name = os.path.splitext(os.path.basename(pic_path))[0]
    tile_size = args.tile_size
    step = args.step_size if args.step_size is not None else tile_size // 2
    if step <= 0:
        raise ValueError("--step must be positive.")
    model_type = 'resnet50' if args.model_type == 'resnet5' else args.model_type
    model_path = args.model
    if model_path is None:
        if model_type == 'conch':
            model_path = DEFAULT_CONCH_HEAD_PATH
        elif model_type == 'vit_b_16':
            model_path = DEFAULT_VIT_B_16_MODEL_PATH
        else:
            model_path = DEFAULT_RESNET50_MODEL_PATH
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 256
    print(f"Using model_type={model_type}, model={model_path}, batch_size={batch_size}")

    # 1. Generate Label Matrix (Batched & Parallel IO)
    label_matrix, original_dim, step, slide_mpp = generate_label_matrix(
        pic_path,
        model_type,
        model_path,
        args.conch_model_dir,
        tile_size,
        step,
        args.gpu_id,
        batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        use_tissue_prefilter=not args.disable_tissue_prefilter,
        tissue_mask_level=args.tissue_mask_level,
        tissue_white_threshold=args.tissue_white_threshold,
        tissue_ratio_threshold=args.tissue_ratio_threshold,
        tissue_close_radius=args.tissue_close_radius,
        tissue_min_downsample=args.tissue_min_downsample,
        tissue_max_mask_pixels=args.tissue_max_mask_pixels,
        exact_white_filter=args.exact_white_filter,
        exact_white_ratio=args.exact_white_ratio,
        amp=not args.no_amp,
    )

    if args.mpp is not None:
        effective_mpp = args.mpp
    elif slide_mpp is not None:
        effective_mpp = slide_mpp
    else:
        effective_mpp = args.default_mpp
        print(f"Slide MPP not found; using default_mpp={effective_mpp}")

    # 2. Define tumor-related lymphocytes by proximity to reliable Tumour regions.
    label_matrix = define_tumor_related_lymphocytes(
        label_matrix,
        step=step,
        mpp=effective_mpp,
        radius_um=args.trl_radius_um,
        min_tumor_component_tiles=args.min_tumor_component_tiles,
        tumor_closing_radius=args.tumor_closing_radius,
    )

    # 3. Count and Save
    unique, counts = np.unique(label_matrix, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    # Ensure all keys exist for the filename
    count_str_parts = []
    for i in range(len(class_map)):
        count_str_parts.append(str(counts_dict.get(i, 0)))
    count_suffix = "_".join(count_str_parts)

    save_path = os.path.join(args.out, f"{WSI_name}_counts_{count_suffix}.jpg")
    metrics_path = os.path.splitext(save_path)[0] + "_metrics.csv"
    
    if not args.no_save_vis:
        vis_img = create_vis_from_matrix(label_matrix, original_dim, step, scale_percent=args.vis_scale_percent)
        vis_img.save(save_path, quality=args.jpg_quality)
        print(f"Saved visualization to {save_path}")
    else:
        print("Visualization saving skipped because --no_save_vis was set.")

    metrics_row = build_count_metrics(WSI_name, model_type, counts_dict)
    save_metrics_csv(metrics_path, metrics_row)
    
    print(f"Saved metrics to {metrics_path}")
    print(f"LSR={metrics_row['LSR']:.8f}, NEW_LSR={metrics_row['NEW_LSR']:.8f}")

if __name__ == '__main__':
    main()
