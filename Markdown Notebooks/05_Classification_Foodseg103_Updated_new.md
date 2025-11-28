```python
# ==================================================
# Cell 1: Setup environment, mount Drive, define paths
# ==================================================

!pip install -q timm==0.9.16

import os, random, time, pickle, copy
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm
from torchvision import transforms
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# NEW PROJECT ROOT
PROJECT_ROOT = Path("/content/drive/MyDrive/Cal_Estimation_Project")
IMG_DIR = PROJECT_ROOT / "data/raw/foodseg103/FoodSeg103/Images/img_dir"
ANN_DIR = PROJECT_ROOT / "data/raw/foodseg103/FoodSeg103/Images/ann_dir"
CAT_FILE = PROJECT_ROOT / "data/raw/foodseg103/FoodSeg103/category_id.txt"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("IMG_DIR     :", IMG_DIR, "| exists?", IMG_DIR.exists())
print("ANN_DIR     :", ANN_DIR, "| exists?", ANN_DIR.exists())
print("CAT_FILE    :", CAT_FILE, "| exists?", CAT_FILE.exists())

assert IMG_DIR.exists() and ANN_DIR.exists() and CAT_FILE.exists(), "Fix paths in Cell 1 before continuing."

CKPT_DIR = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def gpu_clear():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

```

    Device: cuda
    Mounted at /content/drive
    PROJECT_ROOT: /content/drive/MyDrive/Cal_Estimation_Project
    IMG_DIR     : /content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103/Images/img_dir | exists? True
    ANN_DIR     : /content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103/Images/ann_dir | exists? True
    CAT_FILE    : /content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103/category_id.txt | exists? True



```python
# ==================================================
# Cell 2: Cache utilities for fast re-runs (NEW)
# ==================================================

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def save_cache(obj, name="05_cache"):
    path = CACHE_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✔ Saved cache → {path}")

def load_cache(name="05_cache"):
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        print(f"✔ Loaded cache → {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

```


```python
# ==================================================
# Cell 3: Load category_id.txt and build id->name
# ==================================================

cat_id2name = {}
with open(CAT_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cid = int(parts[0])
        name = " ".join(parts[1:])
        cat_id2name[cid] = name

print(f"Loaded {len(cat_id2name)} entries from category_id.txt")
print("Examples:", list(cat_id2name.items())[:10])

all_class_ids = sorted([cid for cid in cat_id2name.keys() if cid != 0])
print("Total non-background food classes:", len(all_class_ids))

```

    Loaded 104 entries from category_id.txt
    Examples: [(0, 'background'), (1, 'candy'), (2, 'egg tart'), (3, 'french fries'), (4, 'chocolate'), (5, 'biscuit'), (6, 'popcorn'), (7, 'pudding'), (8, 'ice cream'), (9, 'cheese butter')]
    Total non-background food classes: 103



```python
# ==================================================
# Cell 4: Build and cache all usable img/mask pairs
# ==================================================

pairs_cache = PROJECT_ROOT / "foodseg103_pairs_cache.pkl"

if pairs_cache.exists():
    print("Loading usable_pairs + all_ids_per_pair from cache:", pairs_cache)
    with open(pairs_cache, "rb") as f:
        data = pickle.load(f)
    usable_pairs = data["usable_pairs"]
    all_ids_per_pair = data["all_ids_per_pair"]
else:
    print("Scanning masks to build usable_pairs (one time)...")
    split_names = ["train", "val", "test"]
    usable_pairs = []
    all_ids_per_pair = []

    for split in split_names:
        img_dir_split = IMG_DIR / split
        ann_dir_split = ANN_DIR / split
        if not img_dir_split.exists():
            continue
        for img_name in sorted(os.listdir(img_dir_split)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = img_dir_split / img_name
            ann_path = ann_dir_split / (img_name.rsplit(".", 1)[0] + ".png")
            if not ann_path.exists():
                continue
            try:
                mask = np.array(Image.open(ann_path))
            except UnidentifiedImageError:
                print("[WARN] Skipping unreadable mask:", ann_path)
                continue

            ids = set(np.unique(mask).tolist())
            ids.discard(0)
            if len(ids) == 0:
                continue

            usable_pairs.append((img_path, ann_path))
            all_ids_per_pair.append(ids)

    with open(pairs_cache, "wb") as f:
        pickle.dump(
            {
                "usable_pairs": usable_pairs,
                "all_ids_per_pair": all_ids_per_pair,
            },
            f,
        )
    print("Cached to:", pairs_cache)

print("Total usable pairs:", len(usable_pairs))
print("Example pair:", usable_pairs[0])
print("Example ids:", list(all_ids_per_pair[0])[:10])

gpu_clear()

```

    Loading usable_pairs + all_ids_per_pair from cache: /content/drive/MyDrive/Cal_Estimation_Project/foodseg103_pairs_cache.pkl
    Total usable pairs: 7118
    Example pair: (PosixPath('/content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103/Images/img_dir/train/00000000.jpg'), PosixPath('/content/drive/MyDrive/Cal_Estimation_Project/data/raw/foodseg103/FoodSeg103/Images/ann_dir/train/00000000.png'))
    Example ids: [48, 90, 66]



```python
# ==================================================
# Cell 5: Select images, enforce coverage, split train/val
# ==================================================

N_IMAGES = 3000  # or None to use all

rng = np.random.RandomState(SEED)
num_pairs = len(usable_pairs)
indices = list(range(num_pairs))

covered = set()
selected_indices = []

for cid in all_class_ids:
    candidates = [i for i, s in enumerate(all_ids_per_pair) if cid in s]
    if not candidates:
        print(f"[WARN] No images for class id {cid} ({cat_id2name.get(cid, '?')})")
        continue
    chosen = rng.choice(candidates)
    selected_indices.append(chosen)
    covered.add(cid)

selected_indices = sorted(set(selected_indices))
print(f"After coverage: {len(selected_indices)} selected (unique)")

if (N_IMAGES is None) or (N_IMAGES <= 0) or (N_IMAGES > num_pairs):
    target_n = num_pairs
else:
    target_n = N_IMAGES

remaining = [i for i in indices if i not in selected_indices]
rng.shuffle(remaining)
while len(selected_indices) < target_n and remaining:
    selected_indices.append(remaining.pop())
selected_indices = sorted(selected_indices)
print(f"Total selected images: {len(selected_indices)} (target {target_n})")

selected_pairs = [usable_pairs[i] for i in selected_indices]
selected_ids_per_pair = [all_ids_per_pair[i] for i in selected_indices]

rng.shuffle(selected_indices)
split_point = int(0.8 * len(selected_indices))
train_img_indices = set(selected_indices[:split_point])
val_img_indices   = set(selected_indices[split_point:])

print(f"Train images: {len(train_img_indices)} | Val images: {len(val_img_indices)}")

gpu_clear()

```

    After coverage: 102 selected (unique)
    Total selected images: 3000 (target 3000)
    Train images: 2400 | Val images: 600



```python
# ==================================================
# Cell 6: Build or load segment crops (masked) for training/validation
# ==================================================

MIN_AREA = 300   # min number of pixels in mask
PAD_FRAC = 0.05  # padding fraction around bbox

segments_cache = PROJECT_ROOT / "foodseg103_segments_convnextL_masked.pkl"

if segments_cache.exists():
    print(f"Loading segments from cache: {segments_cache}")
    with open(segments_cache, "rb") as f:
        data = pickle.load(f)
    segments = data["segments"]
    train_segments = data["train_segments"]
    val_segments = data["val_segments"]
    print(f"Loaded from cache -> Total segments: {len(segments)}")
    print(f"Train segments: {len(train_segments)} | Val segments: {len(val_segments)}")

else:
    print("No segments cache found. Building segments from masks (one-time heavy step)...")
    segments = []

    total_sel = len(selected_indices)
    for k, global_idx in enumerate(selected_indices):
        img_path, ann_path = usable_pairs[global_idx]
        split = "train" if global_idx in train_img_indices else "val"

        if k % 100 == 0:
            print(f"[{k}/{total_sel}] Processing idx={global_idx} -> {img_path.name}")

        try:
            img = Image.open(img_path).convert("RGB")
            mask = np.array(Image.open(ann_path))
        except UnidentifiedImageError:
            print(f"[WARN] Skipping unreadable {img_path} / {ann_path}")
            continue

        h, w = mask.shape
        ids_here = np.unique(mask)
        ids_here = ids_here[ids_here != 0]
        if ids_here.size == 0:
            continue

        for cid in ids_here:
            ys, xs = np.where(mask == cid)
            if xs.size < MIN_AREA:
                continue

            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()

            pad = int(PAD_FRAC * max(h, w))
            x0p = max(0, x0 - pad)
            y0p = max(0, y0 - pad)
            x1p = min(w - 1, x1 + pad)
            y1p = min(h - 1, y1 + pad)

            segments.append(
                {
                    "img_idx": int(global_idx),
                    "split": split,
                    "img_path": str(img_path),
                    "ann_path": str(ann_path),
                    "class_id": int(cid),
                    "class_name": cat_id2name.get(int(cid), f"class_{cid}"),
                    "bbox": (int(x0p), int(y0p), int(x1p), int(y1p)),
                }
            )

    print(f"\nFinished building segments. Total segments: {len(segments)}")

    train_segments = [s for s in segments if s["split"] == "train"]
    val_segments   = [s for s in segments if s["split"] == "val"]

    print(f"Train segments: {len(train_segments)} | Val segments: {len(val_segments)}")

    with open(segments_cache, "wb") as f:
        pickle.dump(
            {
                "segments": segments,
                "train_segments": train_segments,
                "val_segments": val_segments,
            },
            f,
        )
    print("Saved segments cache to:", segments_cache)

gpu_clear()

```

    Loading segments from cache: /content/drive/MyDrive/Cal_Estimation_Project/foodseg103_segments_convnextL_masked.pkl
    Loaded from cache -> Total segments: 10934
    Train segments: 8778 | Val segments: 2156



```python
# ==================================================
# Cell 7: Dataset, transforms, dataloaders, class mapping
# ==================================================

from torch.utils.data import Dataset, DataLoader

# Class mapping: use class names (human-readable)
all_class_names = sorted({s["class_name"] for s in segments})
class2idx = {name: i for i, name in enumerate(all_class_names)}
idx2class = {i: name for name, i in class2idx.items()}
NUM_CLASSES = len(class2idx)

print("Num classes:", NUM_CLASSES)
print("Example class2idx entries:", list(class2idx.items())[:10])

IMG_SIZE = 288

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class SegmentDataset(Dataset):
    def __init__(self, segments, class2idx, transform, use_mask=True, return_meta=False):
        self.segments = segments
        self.class2idx = class2idx
        self.transform = transform
        self.use_mask = use_mask
        self.return_meta = return_meta

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        s = self.segments[idx]

        # --- Load full image & mask ---
        img = Image.open(s["img_path"]).convert("RGB")
        img_np = np.array(img)  # (H_img, W_img, 3)

        mask_full = np.array(Image.open(s["ann_path"]))
        if mask_full.ndim == 3:
            mask_full = mask_full[:, :, 0]  # (H_mask, W_mask)

        h_img, w_img, _ = img_np.shape
        h_mask, w_mask = mask_full.shape

        x0, y0, x1, y1 = s["bbox"]

        # --- Clamp bbox to both image and mask bounds ---
        x0 = max(0, min(x0, w_img - 1, w_mask - 1))
        y0 = max(0, min(y0, h_img - 1, h_mask - 1))
        x1 = max(x0, min(x1, w_img - 1, w_mask - 1))
        y1 = max(y0, min(y1, h_img - 1, h_mask - 1))

        # --- Crop image and mask with same bbox ---
        crop_np = img_np[y0 : y1 + 1, x0 : x1 + 1, :]            # (Hc_img, Wc_img, 3)
        cid = s["class_id"]
        mask_local = (mask_full[y0 : y1 + 1, x0 : x1 + 1] == cid)  # (Hc_mask, Wc_mask)

        # --- Extra safety: force same (h, w) if shapes differ slightly ---
        if crop_np.shape[:2] != mask_local.shape[:2]:
            h = min(crop_np.shape[0], mask_local.shape[0])
            w = min(crop_np.shape[1], mask_local.shape[1])
            crop_np = crop_np[:h, :w, :]
            mask_local = mask_local[:h, :w]

        # --- Apply instance mask (zero out background) if requested ---
        if self.use_mask and mask_local.sum() > 0:
            mask_local_3 = np.repeat(mask_local[:, :, None], 3, axis=2)  # (Hc, Wc, 3)
            crop_np = np.where(mask_local_3, crop_np, 0)

        crop_img = Image.fromarray(crop_np)
        tensor = self.transform(crop_img)
        label = self.class2idx[s["class_name"]]

        if self.return_meta:
            meta = {
                "img_path": s["img_path"],
                "ann_path": s["ann_path"],
                "class_name": s["class_name"],
                "class_id": s["class_id"],
                "bbox": s["bbox"],
            }
            return tensor, label, meta
        else:
            return tensor, label


# ↓↓↓ Smaller batch sizes to avoid OOM when unfreezing ↓↓↓
BATCH_TRAIN = 24
BATCH_VAL   = 32   # val has no grads, can be a bit larger

train_ds = SegmentDataset(train_segments, class2idx, train_transform, use_mask=True, return_meta=False)
val_ds   = SegmentDataset(val_segments,   class2idx, val_transform,   use_mask=True, return_meta=False)

dl_tr = DataLoader(
    train_ds,
    batch_size=BATCH_TRAIN,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
dl_va = DataLoader(
    val_ds,
    batch_size=BATCH_VAL,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print("Train batches:", len(dl_tr), "| Val batches:", len(dl_va))

# Optional quick check
batch = next(iter(dl_tr))
imgs, labels = batch
print("Sanity check batch shapes -> imgs:", imgs.shape, "| labels:", labels.shape)

gpu_clear()


```

    Num classes: 103
    Example class2idx entries: [('French beans', 0), ('almond', 1), ('apple', 2), ('apricot', 3), ('asparagus', 4), ('avocado', 5), ('bamboo shoots', 6), ('banana', 7), ('bean sprouts', 8), ('biscuit', 9)]
    Train batches: 366 | Val batches: 68
    Sanity check batch shapes -> imgs: torch.Size([24, 3, 288, 288]) | labels: torch.Size([24])



```python
# ==================================================
# Cell 8: Build class_weights for balanced CrossEntropy
#   >>> RUN THIS BEFORE THE CONVNEXT TRAINING CELL <<<
# ==================================================

from collections import Counter

train_counts = Counter(s["class_name"] for s in train_segments)

class_freq = np.zeros(NUM_CLASSES, dtype=np.float32)
for name, cnt in train_counts.items():
    idx = class2idx[name]
    class_freq[idx] = cnt

class_freq[class_freq == 0] = 1.0
class_weights_np = 1.0 / np.sqrt(class_freq)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)

print("Built class_weights with shape:", class_weights.shape)
print("First 10 class_freq:", class_freq[:10])
print("First 10 class_weights:", class_weights[:10])

```

    Built class_weights with shape: torch.Size([103])
    First 10 class_freq: [131.  31.  50.   8.  88.  24.   4.  52.  12. 105.]
    First 10 class_weights: tensor([0.0874, 0.1796, 0.1414, 0.3536, 0.1066, 0.2041, 0.5000, 0.1387, 0.2887,
            0.0976], device='cuda:0')



```python
# ==================================================
# Cell 9: Load ConvNeXt Large checkpoint or train if missing
# ==================================================

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

CKPT_PATH = "/content/drive/MyDrive/Cal_Estimation_Project/checkpoints/convnext_large_foodseg103_masked_ema.pth"
SKIP_TRAIN_IF_CKPT = True

print("Looking for checkpoint at:", CKPT_PATH)
print("Exists:", os.path.exists(CKPT_PATH))


def build_model():
    model = timm.create_model("convnext_large", pretrained=True, num_classes=NUM_CLASSES)
    return model


def evaluate_model(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            with autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
            total_loss += float(loss.item()) * xb.size(0)

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    return acc, avg_loss


if SKIP_TRAIN_IF_CKPT and os.path.exists(CKPT_PATH):
    print("Checkpoint found. Loading EMA model. Skipping training.")
    ema_eval_model = build_model().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    if "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        raise KeyError("Checkpoint does not contain model weights.")

    ema_eval_model.load_state_dict(state_dict)
    ema_eval_model.eval()

    val_acc, val_loss = evaluate_model(ema_eval_model, dl_va)
    print("Loaded EMA checkpoint.")
    print("ValAcc:", val_acc, "| ValLoss:", val_loss)
else:
    print("No checkpoint found. Training will start.")

    EPOCHS = 20
    LR = 3e-4
    WD = 1e-2
    EARLY_PATIENCE = 6
    WARMUP_EPOCHS = 2

    model = build_model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)

    ema_model = build_model().to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.999

    def update_ema(ema_model, model, alpha):
        with torch.no_grad():
            for ep, p in zip(ema_model.parameters(), model.parameters()):
                ep.data.mul_(alpha).add_(p.data, alpha=1 - alpha)

    best_val = 0.0
    best_epoch = -1
    no_improve = 0

    def run_epoch(train=True):
        loader = dl_tr if train else dl_va
        model.train() if train else model.eval()

        correct = 0
        total = 0
        running_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                update_ema(ema_model, model, ema_decay)

            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
            running_loss += float(loss.item()) * xb.size(0)

        acc = correct / max(total, 1)
        avg_loss = running_loss / max(total, 1)
        return acc, avg_loss

    print("Starting ConvNeXt Large training...")
    for epoch in range(1, EPOCHS + 1):
        train_acc, train_loss = run_epoch(train=True)
        val_acc, val_loss = run_epoch(train=False)

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS} | TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f}")

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_epoch = epoch
            no_improve = 0

            torch.save(
                {
                    "ema_state_dict": ema_model.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val,
                    "epoch": epoch,
                },
                CKPT_PATH,
            )
            print("New best model saved.")
        else:
            no_improve += 1
            print("No improvement.")
            if no_improve >= EARLY_PATIENCE:
                print("Early stopping.")
                break

    ema_eval_model = ema_model
    ema_eval_model.eval()
    val_acc, val_loss = evaluate_model(ema_eval_model, dl_va)
    print("Training complete. Best ValAcc:", best_val, "at epoch", best_epoch)
    print("Final EMA ValAcc:", val_acc, "ValLoss:", val_loss)

```

    Looking for checkpoint at: /content/drive/MyDrive/Cal_Estimation_Project/checkpoints/convnext_large_foodseg103_masked_ema.pth
    Exists: True
    Checkpoint found. Loading EMA model. Skipping training.


    /tmp/ipython-input-2112478679.py:32: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=(DEVICE.type == "cuda")):


    Loaded EMA checkpoint.
    ValAcc: 0.9081632653061225 | ValLoss: 0.734186805909109



```python
# ==================================================
# Cell 10: Visualize 5 classified images + export PNGs & JSON
# ==================================================

import json

EXPORT_ROOT = PROJECT_ROOT / "Outputs" / "classification_visuals"
EXPORT_IMG_DIR = EXPORT_ROOT / "images"
EXPORT_JSON_DIR = EXPORT_ROOT / "json"

EXPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_JSON_DIR.mkdir(parents=True, exist_ok=True)

print("Image exports ->", EXPORT_IMG_DIR)
print("JSON exports  ->", EXPORT_JSON_DIR)


def ensure_ema_model():
    global ema_eval_model
    try:
        ema_eval_model
    except NameError:
        print("ema_eval_model not in memory, loading best EMA checkpoint...")
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        ema_eval_model_local = build_model().to(DEVICE)
        ema_eval_model_local.load_state_dict(ckpt["ema_state_dict"])
        ema_eval_model = ema_eval_model_local
    ema_eval_model.eval()
    ema_eval_model.to(DEVICE)


val_img_to_indices = defaultdict(list)
for i, s in enumerate(val_segments):
    val_img_to_indices[s["img_path"]].append(i)

val_meta_ds = SegmentDataset(
    val_segments, class2idx, val_transform, use_mask=True, return_meta=True
)


def make_segmentation_overlay(orig_np, mask_full):
    if mask_full.ndim == 3:
        mask_full = mask_full[:, :, 0]

    overlay = orig_np.copy().astype(np.float32)
    unique_ids = [cid for cid in np.unique(mask_full) if cid != 0]
    rng = np.random.RandomState(0)
    color_map = {cid: rng.randint(0, 255, size=3) for cid in unique_ids}
    alpha = 0.5
    for cid in unique_ids:
        color = color_map[cid]
        mask = (mask_full == cid)
        overlay[mask] = alpha * color.astype(np.float32) + (1 - alpha) * overlay[mask]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def visualize_full_image_predictions(num_images=5, max_foods_per_image=6):
    ensure_ema_model()

    if len(val_img_to_indices) == 0:
        print("No validation segments available.")
        return

    img_paths = list(val_img_to_indices.keys())
    num_images = min(num_images, len(img_paths))
    chosen_paths = random.sample(img_paths, k=num_images)

    for img_idx, img_path in enumerate(chosen_paths):
        seg_indices = val_img_to_indices[img_path][:max_foods_per_image]

        orig = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig)

        ann_path = val_segments[seg_indices[0]]["ann_path"]
        mask_full = np.array(Image.open(ann_path))
        if mask_full.ndim == 3:
            mask_full = mask_full[:, :, 0]

        seg_overlay = make_segmentation_overlay(orig_np, mask_full)
        n_foods = len(seg_indices)
        if n_foods == 0:
            print(f"No segments for image: {img_path}")
            continue

        n_rows = 1 + n_foods
        n_cols = 2
        plt.figure(figsize=(6 * n_cols, 3 * n_rows))

        ax0 = plt.subplot(n_rows, n_cols, 1)
        ax0.imshow(orig_np)
        ax0.set_title("Original Image")
        ax0.axis("off")

        ax1 = plt.subplot(n_rows, n_cols, 2)
        ax1.imshow(seg_overlay)
        ax1.set_title("Segmented Image (all classes)")
        ax1.axis("off")

        for row_i, seg_idx in enumerate(seg_indices, start=1):
            img_tensor, label, meta = val_meta_ds[seg_idx]
            true_name = meta["class_name"]
            cid = meta["class_id"]
            x0, y0, x1, y1 = meta["bbox"]

            img_np_crop = img_tensor.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_np_crop = std * img_np_crop + mean
            img_np_crop = np.clip(img_np_crop, 0.0, 1.0)

            mask_full_local = mask_full.copy()
            mask_instance = (mask_full_local == cid)
            mask_instance_box = np.zeros_like(mask_instance)
            mask_instance_box[y0:y1+1, x0:x1+1] = mask_instance[y0:y1+1, x0:x1+1]

            inst_overlay = orig_np.copy().astype(np.float32)
            alpha_inst = 0.6
            color_inst = np.array([255, 0, 0])
            inst_overlay[mask_instance_box] = (
                alpha_inst * color_inst + (1 - alpha_inst) * inst_overlay[mask_instance_box]
            )
            inst_overlay = np.clip(inst_overlay, 0, 255).astype(np.uint8)

            with torch.no_grad():
                inp = img_tensor.unsqueeze(0).to(DEVICE)
                with autocast(enabled=(DEVICE.type == "cuda")):
                    logits = ema_eval_model(inp)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(probs.argmax())
                pred_name = idx2class[pred_idx]
                conf = float(probs[pred_idx])

            ax_left = plt.subplot(n_rows, n_cols, row_i * n_cols + 1)
            ax_left.imshow(inst_overlay)
            ax_left.set_title(f"Food {row_i} mask\nGT: {true_name}", fontsize=9)
            ax_left.axis("off")

            ax_right = plt.subplot(n_rows, n_cols, row_i * n_cols + 2)
            ax_right.imshow(img_np_crop)
            ax_right.set_title(
                f"P: {pred_name}\nT: {true_name}\nConf: {conf:.2f}", fontsize=9
            )
            ax_right.axis("off")

        plt.tight_layout()
        plt.show()
        print(f"[VIS] Visualized image {img_idx+1}/{num_images}: {img_path}")


def export_visuals_and_json_all(
    max_foods_per_image=8,
    export_mode="all_json_png_some",
    max_png_images=50,
):
    ensure_ema_model()

    img_paths = list(val_img_to_indices.keys())
    total_images = len(img_paths)
    print(f"Found {total_images} validation images for export.")

    png_limit = total_images if export_mode == "all_json_png_all" else max_png_images
    print(f"PNG export mode: {export_mode} (up to {png_limit} PNGs)")

    for img_idx, img_path in enumerate(img_paths):
        seg_indices = val_img_to_indices[img_path][:max_foods_per_image]
        orig = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig)

        ann_path = val_segments[seg_indices[0]]["ann_path"]
        mask_full = np.array(Image.open(ann_path))
        if mask_full.ndim == 3:
            mask_full = mask_full[:, :, 0]

        seg_overlay = make_segmentation_overlay(orig_np, mask_full)
        n_foods = len(seg_indices)
        if n_foods == 0:
            print(f"[WARN] No segments for {img_path}")
            continue

        img_json = {
            "image_path": img_path,
            "annotation_path": ann_path,
            "segments": []
        }

        do_png = img_idx < png_limit
        fig = None
        if do_png:
            n_rows = 1 + n_foods
            n_cols = 2
            fig = plt.figure(figsize=(6 * n_cols, 3 * n_rows))

            ax0 = plt.subplot(n_rows, n_cols, 1)
            ax0.imshow(orig_np)
            ax0.set_title("Original Image")
            ax0.axis("off")

            ax1 = plt.subplot(n_rows, n_cols, 2)
            ax1.imshow(seg_overlay)
            ax1.set_title("Segmented Image (all classes)")
            ax1.axis("off")

        for row_i, seg_idx in enumerate(seg_indices, start=1):
            img_tensor, label, meta = val_meta_ds[seg_idx]
            true_name = meta["class_name"]
            cid = meta["class_id"]
            x0, y0, x1, y1 = meta["bbox"]

            img_np_crop = img_tensor.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_np_crop = std * img_np_crop + mean
            img_np_crop = np.clip(img_np_crop, 0.0, 1.0)

            mask_instance = (mask_full == cid)
            mask_instance_box = np.zeros_like(mask_instance)
            mask_instance_box[y0:y1+1, x0:x1+1] = mask_instance[y0:y1+1, x0:x1+1]

            inst_overlay = orig_np.copy().astype(np.float32)
            alpha_inst = 0.6
            color_inst = np.array([255, 0, 0])
            inst_overlay[mask_instance_box] = (
                alpha_inst * color_inst + (1 - alpha_inst) * inst_overlay[mask_instance_box]
            )
            inst_overlay = np.clip(inst_overlay, 0, 255).astype(np.uint8)

            with torch.no_grad():
                inp = img_tensor.unsqueeze(0).to(DEVICE)
                with autocast(enabled=(DEVICE.type == "cuda")):
                    logits = ema_eval_model(inp)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(probs.argmax())
                pred_name = idx2class[pred_idx]
                conf = float(probs[pred_idx])

            if do_png and fig is not None:
                ax_left = plt.subplot(n_rows, n_cols, row_i * n_cols + 1)
                ax_left.imshow(inst_overlay)
                ax_left.set_title(f"Food {row_i} mask\nGT: {true_name}", fontsize=9)
                ax_left.axis("off")

                ax_right = plt.subplot(n_rows, n_cols, row_i * n_cols + 2)
                ax_right.imshow(img_np_crop)
                ax_right.set_title(
                    f"P: {pred_name}\nT: {true_name}\nConf: {conf:.2f}",
                    fontsize=9,
                )
                ax_right.axis("off")

            mask_area = int(mask_instance_box.sum())
            img_json["segments"].append(
                {
                    "segment_index": int(seg_idx),
                    "class_name_pred": pred_name,
                    "class_name_true": true_name,
                    "class_id": int(cid),
                    "confidence": conf,
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "mask_area": mask_area,
                    "volume": None,
                    "mass": None,
                    "calories": None,
                }
            )

        img_stem = os.path.splitext(os.path.basename(img_path))[0]

        if do_png and fig is not None:
            plt.tight_layout()
            fig_path = EXPORT_IMG_DIR / f"{img_stem}_classification.png"
            plt.savefig(fig_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"[{img_idx+1}/{total_images}] PNG  -> {fig_path.name}")
        elif fig is not None:
            plt.close(fig)

        json_path = EXPORT_JSON_DIR / f"{img_stem}_classification.json"
        with open(json_path, "w") as f:
            json.dump(img_json, f, indent=2)
        print(f"[{img_idx+1}/{total_images}] JSON -> {json_path.name}")

    print("\nExport (all images) completed.")


# ---- RUN VISUALIZATION + EXPORT ----
visualize_full_image_predictions(num_images=5, max_foods_per_image=6)

export_visuals_and_json_all(
    max_foods_per_image=8,
    export_mode="all_json_png_some",
    max_png_images=50,
)

```


    Output hidden; open in https://colab.research.google.com to view.



```python
# ==================================================
# Cell 11: Save cache for fast re-run (NEW)
# ==================================================

cache_data = {
    "checkpoint_path": str(CKPT_PATH),
    "num_segments_total": len(segments),
    "num_train_segments": len(train_segments),
    "num_val_segments": len(val_segments),
    "num_classes": NUM_CLASSES,
    "class2idx": class2idx,
    "idx2class": idx2class,
    "export_root": str(EXPORT_ROOT),
}

save_cache(cache_data, "05_cache")

```

    ✔ Saved cache → /content/drive/MyDrive/Cal_Estimation_Project/cache/05_cache.pkl



```python

```
