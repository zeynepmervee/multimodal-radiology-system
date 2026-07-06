# -*- coding: utf-8 -*-
"""
Multimodal Fusion of Radiological Images and Clinical Reports
for Multi-Label Chest Disease Detection under Class Imbalance
--------------------------------------------------------------
Reference implementation of the PROPOSED METHOD only (no ablation baselines).

Pipeline (matches the paper, Section III):
  * Image branch : EfficientNet-B0 (ImageNet weights) -> Linear(., 512)          Eq. (2)
  * Text  branch : BERT bert-base-uncased, pooler_output(768) -> Linear(768, 512) Eq. (3)
  * Late fusion  : concat -> R^1024 -> Linear(1024, 256) -> ReLU -> Linear(256, 14) Eq. (4-6)
  * Output       : independent Sigmoid per finding (multi-label)                   Eq. (6)
  * Loss         : multi-label Focal Loss, alpha = 0.5, gamma = 2                  Eq. (7)
  * Decision rule: threshold tau = 0.30 (chosen from a 0.10..0.50 sweep)           Eq. (8)

Training configuration (Section III-G):
  Adam, lr = 1e-4, StepLR(step=10, gamma=0.5), 30 epochs, batch size 4,
  best checkpoint = maximum validation Micro-F1 (at 0.5). Fixed seed for reproducibility.

How to run:
  Just run  ->  python proposed_method.py
  Only edit the DATASET PATHS in the CONFIG block below to point at your data.
  The script trains and then evaluates on the held-out test set.
"""
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef,
)
from PIL import Image

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIG  --  edit the two DATASET PATHS below; everything else matches the paper
# ===========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

# Folder with Train.jsonl / Valid.jsonl / Test.jsonl
DATA_DIR = os.path.join(HERE, "archive", "MedViLL-master", "MedViLL-master", "data", "openi")
# Folder with the chest X-ray images
IMG_DIR = os.path.join(HERE, "archive", "openi", "home", "data_storage", "mimic-cxr",
                       "dataset", "open_i", "image_preprocessing", "512_3ch")
# Where the best model is saved
CHECKPOINT = os.path.join(HERE, "proposed_method.pth")

EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 2.0
SEED = 42
NUM_WORKERS = 4
# ===========================================================================

# The 14 thoracic findings, in a stable order (Table II).
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices',
]
LAB2IDX = {label: i for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# ImageNet statistics used for normalisation (Table III).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Threshold sweep used to select the operating point tau = 0.30 (Section III-F).
SWEEP = [0.10, 0.20, 0.30, 0.40, 0.50]
TAU = 0.30  # selected decision threshold


def set_seed(seed):
    """Make the run reproducible across Python, NumPy and PyTorch (CPU/GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_to_vec(text):
    """Turn a comma-separated label string into a 14-dim multi-hot vector."""
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for part in str(text).split(','):
        part = part.strip(" '")
        if part in LAB2IDX:
            vec[LAB2IDX[part]] = 1.0
    return vec


def load_table(split):
    """Read a {Train,Valid,Test}.jsonl split and rewrite image paths to IMG_DIR."""
    df = pd.read_json(os.path.join(DATA_DIR, f"{split}.jsonl"), lines=True)
    df['img'] = df['img'].apply(lambda p: os.path.join(IMG_DIR, os.path.basename(p)))
    return df.reset_index(drop=True)


class CxrDataset(Dataset):
    """Yields (image_tensor, report_text, label_vector) for one case."""

    def __init__(self, df, train=False):
        self.df = df
        if train:
            # Training augmentation (Table III): mild geometric + photometric jitter.
            self.tf = transforms.Compose([
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = Image.open(row['img']).convert("RGB")
        return self.tf(image), str(row['text']), label_to_vec(row['label'])


class MultiModalNet(nn.Module):
    """Late-fusion network: EfficientNet-B0 image branch + BERT text branch."""

    def __init__(self, n_classes=NUM_CLASSES):
        super().__init__()
        # Image branch: EfficientNet-B0 backbone, head replaced by a 512-d projection.
        self.img_net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.img_net.classifier[1].in_features
        self.img_net.classifier[1] = nn.Linear(in_features, 512)     # Eq. (2)

        # Text branch: BERT pooled output (768) projected to the same 512-d size.
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.txt_proj = nn.Linear(768, 512)                          # Eq. (3)

        # Shared decision head over the concatenated 1024-d joint representation.
        self.head = nn.Sequential(
            nn.Linear(512 + 512, 256),                               # Eq. (4-5)
            nn.ReLU(),
            nn.Linear(256, n_classes),                               # Eq. (6): logits
        )

    def forward(self, image, ids, mask):
        v = self.img_net(image)                                      # R^512
        u = self.txt_proj(self.bert(input_ids=ids, attention_mask=mask).pooler_output)  # R^512
        h = torch.cat((v, u), dim=1)                                 # R^1024
        return self.head(h)                                          # logits (Sigmoid applied later)


def focal_loss(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    """Multi-label sigmoid Focal Loss (paper Eq. 7).

    At alpha = 0.5 the balancing term is a constant 0.5, so this reduces exactly
    to the single-alpha form written in the paper.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)           # p_t = p if y==1 else 1-p
    loss = ce * ((1 - p_t) ** gamma)                      # focusing factor
    a_t = alpha * targets + (1 - alpha) * (1 - targets)   # balancing term
    return (a_t * loss).mean()


def infer(model, loader, tokenizer, device):
    """Run the model over a loader and return (y_true, y_prob) as arrays."""
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for images, texts, labels in loader:
            images = images.to(device)
            enc = tokenizer(list(texts), return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            logits = model(images, ids, mask)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            trues.append(labels.numpy())
    return np.concatenate(trues), np.concatenate(probs)


def report(y_true, y_prob):
    """Print the threshold sweep, the metrics at tau = 0.30 and per-class recall."""
    print("\n" + "=" * 64)
    print(" DECISION THRESHOLD SWEEP (selecting the operating point)")
    print("=" * 64)
    print(f"{'tau':>5} | {'MicroF1':>8} | {'MacroF1':>8} | {'MacroRec':>9} | {'MCC':>7}")
    print("-" * 64)
    for t in SWEEP:
        y_pred = (y_prob >= t).astype(float)
        micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
        star = "  <- selected" if abs(t - TAU) < 1e-9 else ""
        print(f"{t:>5.2f} | {micro*100:>7.2f}% | {macro*100:>7.2f}% | "
              f"{rec*100:>8.2f}% | {mcc:>7.4f}{star}")

    y_pred = (y_prob >= TAU).astype(float)
    print("\n" + "=" * 64)
    print(f" PROPOSED METHOD @ tau = {TAU}")
    print("=" * 64)
    print(f"Accuracy (subset)  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Micro  F1          : {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"Macro  F1          : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro  Recall      : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro  Precision   : {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"MCC                : {matthews_corrcoef(y_true.flatten(), y_pred.flatten()):.4f}")

    print("\n[+] Per-class performance @ %.2f\n" % TAU)
    print(f"{'Finding':<28}{'Test cases':>11}{'True pos.':>11}{'Recall':>9}")
    print("-" * 59)
    for c, label in enumerate(LABELS):
        support = int(y_true[:, c].sum())
        tp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 1)).sum())
        rec = f"{tp / support * 100:.1f}%" if support > 0 else "-"
        print(f"{label:<28}{support:>11}{tp:>11}{rec:>9}")
    print("=" * 64)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MultiModalNet(NUM_CLASSES).to(device)

    train_df = load_table("Train")
    val_df = load_table("Valid")
    test_df = load_table("Test")
    print(f"Cases -> train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(CxrDataset(train_df, train=True), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                              generator=generator)
    val_loader = DataLoader(CxrDataset(val_df), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(CxrDataset(test_df), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_f1, best_epoch, best_state = -1.0, -1, None
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for images, texts, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            enc = tokenizer(list(texts), return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            logits = model(images, ids, mask)
            loss = focal_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        # Validation: select the best epoch by Micro-F1 at 0.5 (as in the paper).
        y_val, p_val = infer(model, val_loader, tokenizer, device)
        val_f1 = f1_score(y_val, (p_val > 0.5).astype(float), average="micro", zero_division=0)
        print(f"Epoch {epoch+1:>2}/{EPOCHS} | loss {running/max(1,len(train_loader)):.4f} "
              f"| val MicroF1 {val_f1*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = float(val_f1), epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, CHECKPOINT)
        print(f"\nBest epoch {best_epoch} (val MicroF1 {best_val_f1*100:.2f}%) "
              f"saved to {CHECKPOINT}")

    # Final evaluation on the held-out test set.
    y_true, y_prob = infer(model, test_loader, tokenizer, device)
    report(y_true, y_prob)


if __name__ == "__main__":
    main()
