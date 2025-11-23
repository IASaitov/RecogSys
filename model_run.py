import cv2
import logging
from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import albumentations as A

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ====================== CONFIG ======================
EPOCHS = 2
DATA_FOLDER = '../data/PetImages'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================== MODEL ======================
def get_resnet50_binary_sigmoid(pretrained: bool = False):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(2048, 1)
    return model.to(DEVICE)

# ====================== DATASET WRAPPER ======================
class AlWrapper(datasets.ImageFolder):
    def __init__(self, root, albumentations_transform=None):
        super().__init__(root)
        self.albumentations_transform = albumentations_transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, label

# ====================== METRIC CLASS ======================
class Metric:
    def __init__(self):
        self.labels: List[torch.Tensor] = []
        self.logits: List[torch.Tensor] = []
        self._cache = None

    def add(self, labels: torch.Tensor, logits: torch.Tensor):
        labels = labels.detach().cpu()
        logits = logits.detach().cpu() #.squeeze(1)
        self.labels.append(labels)
        self.logits.append(logits)
        self._cache = None

    def _materialize(self):
        if self._cache is None:
            if not self.labels:
                y_true = torch.empty(0, dtype=torch.long)
                y_pred = torch.empty(0, dtype=torch.long)
            else:
                y_true = torch.cat(self.labels, dim=0)
                y_pred = (torch.cat(self.logits, dim=0) > 0).long()
            self._cache = y_true, y_pred
        return self._cache

    def accuracy(self):
        y_true, y_pred = self._materialize()
        return (y_true == y_pred).float().mean().item() if len(y_true) > 0 else 0.0

    def precision(self):
        y_true, y_pred = self._materialize()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        return tp.float() / (tp + fp + 1e-8)

    def recall(self):
        y_true, y_pred = self._materialize()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        return tp.float() / (tp + fn + 1e-8)

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r + 1e-8) if (p + r) > 0 else 0.0

    def reset(self):
        self.labels.clear()
        self.logits.clear()
        self._cache = None

# ====================== PLOTTING FUNCTION ======================
def plot_training_history(df: pd.DataFrame, title: str):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=20, fontweight='bold')

    metrics = ['Loss', 'accuracy', 'precision', 'recall', 'f1_score']
    titles  = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

    palette = sns.color_palette("husl", 2)
    colors = {'Untrained ResNet-50': palette[0], 'Pre-trained ResNet-50': palette[1]}

    for i, (col, t) in enumerate(zip(metrics, titles)):
        ax = axes[i//3, i%3]
        for model in df['Model'].unique():
            data = df[df['Model'] == model]
            ax.plot(data['epoch'], data[col], label=model, color=colors[model], marker='o', linewidth=2.5)
        ax.set_title(t, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend()

    ax = axes[1, 2]
    for model in df['Model'].unique():
        data = df[df['Model'] == model]
        c = colors[model]
        ax.plot(data['epoch'], data['accuracy'], label=f'{model} Acc', color=c, linewidth=3)
        ax.plot(data['epoch'], data['f1_score'], label=f'{model} F1', color=c, linestyle='--', linewidth=2.5)
    ax.set_title('Accuracy & F1-Score', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


# ====================== MAIN EXECUTION ======================
if __name__ == '__main__':   # This is the CRITICAL fix for Windows!
    log.info(f"Using device: {DEVICE}")

    # Models
    resnet50_untrained = get_resnet50_binary_sigmoid(pretrained=False)
    resnet50_pretrained = get_resnet50_binary_sigmoid(pretrained=True)
    log.info("Models loaded and moved to device.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer_untrained = torch.optim.Adam(resnet50_untrained.parameters(), lr=0.0001)
    optimizer_pretrained = torch.optim.Adam(resnet50_pretrained.parameters(), lr=0.0001)

    # Transforms
    train_transforms = A.Compose([
        A.RandomResizedCrop(size=(224,224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])

    # Dataset & loaders
    dataset = AlWrapper(root=DATA_FOLDER, albumentations_transform=train_transforms)
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_ds, valid_ds, test_ds = random_split(
        dataset, [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    log.info(f"Dataset: {len(dataset)} images → Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    # Metrics storage
    train_metrics = []
    valid_metrics = []
    metric_untrained = Metric()
    metric_pretrained = Metric()

    log.info("Starting training...")

    for epoch in range(1, EPOCHS + 1):
        # Training
        resnet50_untrained.train()
        resnet50_pretrained.train()
        metric_untrained.reset()
        metric_pretrained.reset()

        for batch_data, batch_label in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            batch_data = batch_data.to(DEVICE, non_blocking=True)
            batch_label = batch_label.float().to(DEVICE, non_blocking=True)

            # Untrained
            optimizer_untrained.zero_grad()
            logits1 = resnet50_untrained(batch_data).squeeze(1)
            loss1 = criterion(logits1, batch_label)
            loss1.backward()
            optimizer_untrained.step()
            metric_untrained.add(batch_label, logits1)

            # Pretrained
            optimizer_pretrained.zero_grad()
            logits2 = resnet50_pretrained(batch_data).squeeze(1)
            loss2 = criterion(logits2, batch_label)
            loss2.backward()
            optimizer_pretrained.step()
            metric_pretrained.add(batch_label, logits2)

        # Log train
        train_metrics.extend([
            {'Model': 'Untrained ResNet-50', 'Loss': loss1.item(), 'accuracy': metric_untrained.accuracy(),
             'precision': metric_untrained.precision(), 'recall': metric_untrained.recall(),
             'f1_score': metric_untrained.f1_score(), 'epoch': epoch},
            {'Model': 'Pre-trained ResNet-50', 'Loss': loss2.item(), 'accuracy': metric_pretrained.accuracy(),
             'precision': metric_pretrained.precision(), 'recall': metric_pretrained.recall(),
             'f1_score': metric_pretrained.f1_score(), 'epoch': epoch}
        ])

        log.info(f"Epoch {epoch} [Train] Untrained Acc: {metric_untrained.accuracy():.4f} | Pretrained Acc: {metric_pretrained.accuracy():.4f}")

        # Validation
        resnet50_untrained.eval()
        resnet50_pretrained.eval()
        metric_untrained.reset()
        metric_pretrained.reset()

        with torch.no_grad():
            for batch_data, batch_label in tqdm(valid_loader, desc=f"Epoch {epoch}/{EPOCHS} [Valid]"):
                batch_data = batch_data.to(DEVICE, non_blocking=True)
                batch_label = batch_label.float().to(DEVICE, non_blocking=True)
                metric_untrained.add(batch_label, resnet50_untrained(batch_data).squeeze(1))
                metric_pretrained.add(batch_label, resnet50_pretrained(batch_data).squeeze(1))

        valid_metrics.extend([
            {'Model': 'Untrained ResNet-50', 'Loss': loss1.item(), 'accuracy': metric_untrained.accuracy(),
             'precision': metric_untrained.precision(), 'recall': metric_untrained.recall(),
             'f1_score': metric_untrained.f1_score(), 'epoch': epoch},
            {'Model': 'Pre-trained ResNet-50', 'Loss': loss2.item(), 'accuracy': metric_pretrained.accuracy(),
             'precision': metric_pretrained.precision(), 'recall': metric_pretrained.recall(),
             'f1_score': metric_pretrained.f1_score(), 'epoch': epoch}
        ])

        log.info(f"Epoch {epoch} [Valid] Untrained Acc: {metric_untrained.accuracy():.4f} | Pretrained Acc: {metric_pretrained.accuracy():.4f}")

    # Plot results
    plot_training_history(pd.DataFrame(train_metrics), "Training Progress")
    plot_training_history(pd.DataFrame(valid_metrics), "Validation Progress")

    log.info("Training completed successfully!")
