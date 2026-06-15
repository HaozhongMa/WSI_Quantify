import os
import argparse
import copy
import csv
from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import lr_scheduler

# Set random seed
seed = 114514
torch.manual_seed(seed)

DEFAULT_DATA_ROOT = os.environ.get("WSI_DATA_ROOT", "data/8class_patch/train")
DEFAULT_OUTPUT_DIR = os.environ.get("WSI_OUTPUT_DIR", "outputs/model_out")
DEFAULT_GPU_IDS = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
DEFAULT_MODEL_ARCH = "vit_b_16"
DEFAULT_FREEZE_STRATEGY = "none"
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 8
DEFAULT_NUM_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_SCHEDULER_STEP_SIZE = 7
DEFAULT_SCHEDULER_GAMMA = 0.1


def parse_gpu_ids(gpu_ids_arg):
    if gpu_ids_arg.lower() in {"", "none", "cpu"}:
        return []
    return [int(gpu_id.strip()) for gpu_id in gpu_ids_arg.split(",") if gpu_id.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Train 8-class pathology tile classifier.")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Run training or evaluate a trained checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ImageFolder data.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model checkpoint and evaluation figures.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=DEFAULT_GPU_IDS,
        help="Comma-separated CUDA device IDs to use, e.g. '0'. Use 'cpu' to force CPU.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Checkpoint path to load in test mode.",
    )
    parser.add_argument(
        "--model-arch",
        choices=["resnet50", "efficientnet_b0", "efficientnet_b3", "vit_b_16", "swin_t"],
        default=DEFAULT_MODEL_ARCH,
        help="Model architecture to train or evaluate.",
    )
    parser.add_argument(
        "--freeze-strategy",
        choices=["none", "head-only", "last-block"],
        default=DEFAULT_FREEZE_STRATEGY,
        help=(
            "Fine-tuning strategy. 'none' trains all layers, 'head-only' trains only "
            "the classifier head, and 'last-block' trains the last backbone block plus the classifier head."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for train, validation, and test loaders.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stopping patience in epochs.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw"],
        default=DEFAULT_OPTIMIZER,
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=DEFAULT_MOMENTUM,
        help="Momentum used by SGD.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=DEFAULT_SCHEDULER_STEP_SIZE,
        help="StepLR step size in epochs.",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=DEFAULT_SCHEDULER_GAMMA,
        help="StepLR decay factor.",
    )
    return parser.parse_args()


args = parse_args()

# Data preprocessing and augmentation
DATA_ROOT = args.data_root
OUTPUT_DIR = args.output
GPU_IDS = parse_gpu_ids(args.gpu_ids)
MODE = args.mode
MODEL_PATH = args.model_path
MODEL_ARCH = args.model_arch
FREEZE_STRATEGY = args.freeze_strategy
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
NUM_EPOCHS = args.num_epochs
PATIENCE = args.patience
OPTIMIZER_NAME = args.optimizer
LEARNING_RATE = args.learning_rate
MOMENTUM = args.momentum
WEIGHT_DECAY = args.weight_decay
SCHEDULER_STEP_SIZE = args.scheduler_step_size
SCHEDULER_GAMMA = args.scheduler_gamma
VAL_FRACTION = 0.2
SPLIT_SEARCH_TRIALS = 5000
SPLIT_LOCAL_SEARCH_MAX_ITER = 1000

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}


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


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = normalize_state_dict_for_model(checkpoint, model)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {model_path}")


def build_optimizer(model_parameters):
    if OPTIMIZER_NAME == "sgd":
        return optim.SGD(
            model_parameters,
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
    if OPTIMIZER_NAME == "adam":
        return optim.Adam(
            model_parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    if OPTIMIZER_NAME == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")


def collect_predictions(model, data_loader, device, desc):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=desc):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return np.array(y_true), np.array(y_pred)


def save_labeled_matrix_csv(path, matrix, class_names):
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_label\\pred_label"] + list(class_names))
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row.tolist())


def save_confusion_matrix_png(path, matrix, class_names, title, fmt, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_evaluation_outputs(y_true, y_pred, class_names, output_dir, prefix, legacy_raw_png_path=None):
    os.makedirs(output_dir, exist_ok=True)
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

    per_class_path = os.path.join(output_dir, f"{prefix}_per_class_metrics.csv")
    with open(per_class_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for class_name, p_value, r_value, f1_value, support_value in zip(class_names, precision, recall, f1, support):
            writer.writerow([class_name, p_value, r_value, f1_value, int(support_value)])

    summary_path = os.path.join(output_dir, f"{prefix}_summary_metrics.csv")
    with open(summary_path, "w", newline="") as csv_file:
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

    raw_cm_csv_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_raw_counts.csv")
    normalized_cm_csv_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_normalized.csv")
    save_labeled_matrix_csv(raw_cm_csv_path, raw_cm, class_names)
    save_labeled_matrix_csv(normalized_cm_csv_path, normalized_cm, class_names)

    raw_cm_png_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_raw_counts.png")
    normalized_cm_png_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_normalized.png")
    save_confusion_matrix_png(raw_cm_png_path, raw_cm, class_names, "Raw Count Confusion Matrix", "d")
    save_confusion_matrix_png(normalized_cm_png_path, normalized_cm, class_names, "Normalized Confusion Matrix", ".3f")

    if legacy_raw_png_path is not None:
        save_confusion_matrix_png(legacy_raw_png_path, raw_cm, class_names, "Confusion Matrix", "d")

    print(f"Saved per-class metrics to: {per_class_path}")
    print(f"Saved summary metrics to: {summary_path}")
    print(f"Saved raw count confusion matrix to: {raw_cm_csv_path} and {raw_cm_png_path}")
    print(f"Saved normalized confusion matrix to: {normalized_cm_csv_path} and {normalized_cm_png_path}")
    print(f"macro-F1: {macro_f1:.6f}, weighted-F1: {weighted_f1:.6f}, balanced accuracy: {balanced_acc:.6f}")


def save_training_history_outputs(training_history, best_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    history_path = os.path.join(output_dir, "train_history.csv")
    history_fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "best_val_accuracy_so_far",
        "is_best_epoch",
        "learning_rate",
    ]
    with open(history_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=history_fields)
        writer.writeheader()
        writer.writerows(training_history)

    best_metrics_path = os.path.join(output_dir, "best_training_metrics.csv")
    with open(best_metrics_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        for metric, value in best_metrics.items():
            writer.writerow([metric, value])

    print(f"Saved training history to: {history_path}")
    print(f"Saved best training metrics to: {best_metrics_path}")


def extract_group_from_tile_path(tile_path):
    filename = os.path.basename(tile_path)
    return filename.split(" [", 1)[0]


def split_score_tuple(val_class_counts, total_class_counts, val_fraction):
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


def split_score_value(val_class_counts, total_class_counts, val_fraction):
    missing_classes, max_error, mean_error, total_error = split_score_tuple(
        val_class_counts,
        total_class_counts,
        val_fraction
    )
    return missing_classes * 100.0 + max_error * 10.0 + mean_error + total_error


def split_score_values(candidate_counts, total_class_counts, val_fraction):
    safe_total_class_counts = np.maximum(total_class_counts, 1)
    class_errors = np.abs((candidate_counts / safe_total_class_counts) - val_fraction)
    total_fraction_errors = np.abs((candidate_counts.sum(axis=1) / total_class_counts.sum()) - val_fraction)
    missing_classes = np.sum(candidate_counts == 0, axis=1)
    return missing_classes * 100.0 + class_errors.max(axis=1) * 10.0 + class_errors.mean(axis=1) + total_fraction_errors


def improve_split_by_flips(initial_mask, group_class_counts, total_class_counts, val_fraction):
    val_mask = initial_mask.copy()
    val_class_counts = group_class_counts[val_mask].sum(axis=0)
    current_score = split_score_value(val_class_counts, total_class_counts, val_fraction)

    for _ in range(SPLIT_LOCAL_SEARCH_MAX_ITER):
        best_score = current_score
        best_action = None
        best_group_index = None

        add_indices = np.flatnonzero(~val_mask)
        if len(add_indices) > 0:
            add_counts = val_class_counts + group_class_counts[add_indices]
            add_scores = split_score_values(add_counts, total_class_counts, val_fraction)
            add_position = int(np.argmin(add_scores))
            if float(add_scores[add_position]) < best_score:
                best_score = float(add_scores[add_position])
                best_action = "add"
                best_group_index = int(add_indices[add_position])

        remove_indices = np.flatnonzero(val_mask)
        if len(remove_indices) > 0:
            remove_counts = val_class_counts - group_class_counts[remove_indices]
            remove_scores = split_score_values(remove_counts, total_class_counts, val_fraction)
            remove_position = int(np.argmin(remove_scores))
            if float(remove_scores[remove_position]) < best_score:
                best_score = float(remove_scores[remove_position])
                best_action = "remove"
                best_group_index = int(remove_indices[remove_position])

        if best_action is None:
            break

        if best_action == "add":
            val_mask[best_group_index] = True
            val_class_counts += group_class_counts[best_group_index]
        else:
            val_mask[best_group_index] = False
            val_class_counts -= group_class_counts[best_group_index]
        current_score = best_score

    return val_mask

def build_grouped_split(dataset, val_fraction=VAL_FRACTION):
    paths = np.array([path for path, _ in dataset.samples])
    labels = np.array([label for _, label in dataset.samples])
    groups = np.array([extract_group_from_tile_path(path) for path in paths])

    if len(paths) != len(labels) or len(paths) != len(groups):
        raise ValueError("paths, labels and groups must have the same length.")

    unique_groups, group_ids = np.unique(groups, return_inverse=True)
    group_class_counts = np.zeros((len(unique_groups), len(dataset.classes)), dtype=np.int64)
    for group_id, label in zip(group_ids, labels):
        group_class_counts[group_id, label] += 1

    total_class_counts = np.bincount(labels, minlength=len(dataset.classes))
    rng = np.random.default_rng(seed)
    best_score = None
    best_val_group_mask = None

    for _ in range(SPLIT_SEARCH_TRIALS):
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
        val_fraction
    )

    val_sample_mask = best_val_group_mask[group_ids]
    train_indices = np.flatnonzero(~val_sample_mask)
    val_indices = np.flatnonzero(val_sample_mask)
    split_name = f"BalancedGroupShuffleSearch(n_trials={SPLIT_SEARCH_TRIALS}, local_flip=True)"

    train_groups = set(groups[train_indices])
    val_groups = set(groups[val_indices])
    overlap_groups = train_groups.intersection(val_groups)
    if overlap_groups:
        raise RuntimeError(f"Group leakage detected between train/val: {sorted(overlap_groups)[:5]}")

    print(
        f"Split with {split_name}: "
        f"train={len(train_indices)} tiles/{len(train_groups)} groups, "
        f"val={len(val_indices)} tiles/{len(val_groups)} groups"
    )
    print(f"Train class counts: {np.bincount(labels[train_indices], minlength=len(dataset.classes)).tolist()}")
    print(f"Val class counts: {np.bincount(labels[val_indices], minlength=len(dataset.classes)).tolist()}")
    val_class_counts = np.bincount(labels[val_indices], minlength=len(dataset.classes))
    val_class_fractions = np.round(val_class_counts / np.maximum(total_class_counts, 1), 4).tolist()
    print(f"Val class fractions: {val_class_fractions}")
    return train_indices.tolist(), val_indices.tolist()

full_dataset = datasets.ImageFolder(root=DATA_ROOT)
num_classes = len(full_dataset.classes)
print(f"Found {num_classes} classes: {full_dataset.classes}")

if MODE == "train":
    train_base_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=data_transforms['train'])
    val_base_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=data_transforms['val'])
    train_indices, val_indices = build_grouped_split(full_dataset)
    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
else:
    if MODEL_PATH is None:
        raise ValueError("--model-path is required when --mode test is used.")
    test_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=data_transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Build model architecture and replace its classification head.
def build_model(model_arch, num_classes, use_pretrained):
    if model_arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    if model_arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_arch == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if use_pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if use_pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    if model_arch == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if use_pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model architecture: {model_arch}")


def set_module_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def trainable_parameter_counts(model):
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total_parameters, trainable_parameters


def apply_freeze_strategy(model, model_arch, freeze_strategy):
    if freeze_strategy == "none":
        return

    set_module_trainable(model, False)

    if model_arch == "resnet50":
        set_module_trainable(model.fc, True)
        if freeze_strategy == "last-block":
            set_module_trainable(model.layer4, True)
        return

    if model_arch in {"efficientnet_b0", "efficientnet_b3"}:
        set_module_trainable(model.classifier, True)
        if freeze_strategy == "last-block":
            set_module_trainable(model.features[-1], True)
            if len(model.features) > 1:
                set_module_trainable(model.features[-2], True)
        return

    if model_arch == "vit_b_16":
        set_module_trainable(model.heads, True)
        if freeze_strategy == "last-block":
            set_module_trainable(model.encoder.layers[-1], True)
            set_module_trainable(model.encoder.ln, True)
        return

    if model_arch == "swin_t":
        set_module_trainable(model.head, True)
        if freeze_strategy == "last-block":
            set_module_trainable(model.features[-1], True)
            set_module_trainable(model.norm, True)
        return

    raise ValueError(f"Unsupported model architecture for freezing: {model_arch}")


model = build_model(MODEL_ARCH, num_classes, use_pretrained=(MODE == "train"))
if MODE == "train":
    apply_freeze_strategy(model, MODEL_ARCH, FREEZE_STRATEGY)
total_parameters, trainable_parameters = trainable_parameter_counts(model)
print(f"Model architecture: {MODEL_ARCH}")
print(
    f"Freeze strategy: {FREEZE_STRATEGY}; "
    f"trainable parameters: {trainable_parameters}/{total_parameters}"
)

# Multi-GPU and device setup
device = torch.device(f"cuda:{GPU_IDS[0]}" if torch.cuda.is_available() and GPU_IDS else "cpu")
if torch.cuda.is_available() and len(GPU_IDS) > 1:
    model = nn.DataParallel(model, device_ids=GPU_IDS)
model = model.to(device)
use_amp = device.type == "cuda"

if MODE == "test":
    load_checkpoint(model, MODEL_PATH, device)
    y_true, y_pred = collect_predictions(model, test_loader, device, "Testing")
    save_evaluation_outputs(y_true, y_pred, full_dataset.classes, OUTPUT_DIR, prefix="test")
    raise SystemExit(0)

# Optimizer, loss function, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
trainable_model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
if not trainable_model_parameters:
    raise RuntimeError("No trainable parameters found. Check --freeze-strategy and --model-arch.")
optimizer = build_optimizer(trainable_model_parameters)
scheduler = lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
print(
    f"Optimizer: {OPTIMIZER_NAME}, lr={LEARNING_RATE}, momentum={MOMENTUM}, "
    f"weight_decay={WEIGHT_DECAY}, scheduler_step_size={SCHEDULER_STEP_SIZE}, "
    f"scheduler_gamma={SCHEDULER_GAMMA}"
)

# Mixed precision training (optional)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Track training progress
num_epochs = NUM_EPOCHS
best_val_accuracy = 0.0
best_epoch = 0
best_train_accuracy_at_best_epoch = 0.0
best_train_loss_at_best_epoch = 0.0
best_val_loss_at_best_epoch = 0.0
patience = PATIENCE
epochs_without_improvement = 0
best_model_state = None
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
training_history = []

for epoch in range(num_epochs):
    # Train
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1:02d} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    is_best_epoch = val_accuracy > best_val_accuracy
    if is_best_epoch:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        best_train_accuracy_at_best_epoch = train_accuracy
        best_train_loss_at_best_epoch = train_loss
        best_val_loss_at_best_epoch = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0  # reset
    else:
        epochs_without_improvement += 1

    training_history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "best_val_accuracy_so_far": best_val_accuracy,
        "is_best_epoch": int(is_best_epoch),
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    if not is_best_epoch and epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    
    scheduler.step()

print('Finished Training.')

# Save model
os.makedirs(OUTPUT_DIR, exist_ok=True)
if best_model_state:
    checkpoint_filename = (
        "8class_best_resnet50_model_0622_128.pth"
        if MODEL_ARCH == "resnet50"
        else f"8class_best_{MODEL_ARCH}_model_0622_128.pth"
    )
    save_path = os.path.join(OUTPUT_DIR, checkpoint_filename)
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, save_path)
    save_training_history_outputs(
        training_history,
        {
            "model_arch": MODEL_ARCH,
            "freeze_strategy": FREEZE_STRATEGY,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_train_accuracy_at_best_epoch": best_train_accuracy_at_best_epoch,
            "best_train_loss_at_best_epoch": best_train_loss_at_best_epoch,
            "best_val_loss_at_best_epoch": best_val_loss_at_best_epoch,
            "checkpoint_path": save_path,
        },
        OUTPUT_DIR,
    )
    print(f"Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}")
    print(f"Saved best model to: {save_path}")
else:
    raise RuntimeError("No best model state was captured during training.")

# Clear GPU cache
torch.cuda.empty_cache()

confusion_matrix_path = os.path.join(OUTPUT_DIR, '8class_confusion_matrix_0622_128.png')
y_true, y_pred = collect_predictions(model, val_loader, device, "Evaluating best checkpoint")
save_evaluation_outputs(
    y_true,
    y_pred,
    full_dataset.classes,
    OUTPUT_DIR,
    prefix="val",
    legacy_raw_png_path=confusion_matrix_path,
)

# Visualize training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
metrics_path = os.path.join(OUTPUT_DIR, '8class_training_validation_metrics_0622_128.png')
plt.savefig(metrics_path)
plt.close()
