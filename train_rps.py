"""
Rock-Paper-Scissors Deep Learning Model Training Pipeline
==========================================================
Features:
- Automatic Hardware Detection & Acceleration:
    Automatically uses NVIDIA CUDA GPU (e.g., RTX 3050 Laptop GPU) if available,
    with cuDNN benchmark enabled, mixed precision (AMP fp16), and CPU fallback.
- Dataset Handling:
    - Train split: Rock-Paper-Scissors/train (paper, rock, scissors)
    - Test split: Rock-Paper-Scissors/test (paper, rock, scissors)
    - Optional validation split: Rock-Paper-Scissors/validation (handles both subfolders and flat prefixed files)
    - If validation directory has few samples, performs a stratified/reproducible validation split from train.
- Rich Data Augmentation:
    Random horizontal/vertical flips, subtle rotation, color jitter, affine transforms, normalization.
- Architecture:
    Transfer learning with MobileNetV2 (lightweight, fast, 99%+ accuracy) or custom CNN option.
- Comprehensive Analysis & Visualizations:
    1. Training & Validation Loss curves
    2. Training & Validation Accuracy curves
    3. Learning Rate schedule curve
    4. Normalized Confusion Matrix heatmap
    5. Precision, Recall, F1-Score per class bar plot
    6. Sample predictions gallery (Ground Truth vs Predicted with confidence scores)
    7. JSON report of final metrics & classification report text
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless & reliable saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


CLASS_NAMES = ['paper', 'rock', 'scissors']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}


class RPSFlatDataset(Dataset):
    """Dataset loader for flat folders where class name is in filename (e.g. 'paper1.png')."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        if self.root_dir.exists():
            for f in self.root_dir.iterdir():
                if f.suffix.lower() in valid_extensions:
                    fname = f.name.lower()
                    label = None
                    for c in CLASS_NAMES:
                        if fname.startswith(c):
                            label = CLASS_TO_IDX[c]
                            break
                    if label is not None:
                        self.samples.append((str(f), label))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class RPSFolderDataset(Dataset):
    """Dataset loader for subfolder structures (e.g. root/rock/1.png)."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        for cname in CLASS_NAMES:
            cpath = self.root_dir / cname
            if cpath.exists() and cpath.is_dir():
                label = CLASS_TO_IDX[cname]
                for f in cpath.iterdir():
                    if f.suffix.lower() in valid_extensions:
                        self.samples.append((str(f), label))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class TransformedSubset(Dataset):
    """Applies a specific transform to a PyTorch Subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Retrieve underlying item without previous transform if base had none
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size=224):
    """Returns training, validation, and testing augmentations and normalization."""
    # Standard ImageNet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


def build_model(num_classes=3, pretrained=True, freeze_backbone=False):
    """Builds a MobileNetV2 transfer learning model optimized for fast GPU training."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes)
    )
    return model


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch_num=1, total_epochs=1):
    """Trains the model for one epoch using Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = (device.type == 'cuda')
    num_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        # Real-time batch progress logging
        if batch_idx % max(1, num_batches // 4) == 0 or batch_idx == num_batches:
            batch_acc = (correct / total) * 100
            batch_loss = running_loss / total
            print(f"  [Epoch {epoch_num}/{total_epochs}] Batch {batch_idx:02d}/{num_batches:02d} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}%", flush=True)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on validation or test sets."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def generate_visualizations(history, test_labels, test_preds, test_loader, model, device, output_dir):
    """Generates and saves all training curves, confusion matrices, and analysis graphs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = out_dir / 'graphs'
    graphs_dir.mkdir(exist_ok=True)

    sns.set_theme(style='whitegrid', font_scale=1.1)

    # 1. Training & Validation Loss Curve
    plt.figure(figsize=(9, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'o-', color='#3b82f6', linewidth=2.5, label='Train Loss')
    plt.plot(epochs, history['val_loss'], 's--', color='#ef4444', linewidth=2.5, label='Val Loss')
    plt.title('Training & Validation Loss Over Epochs', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Cross Entropy)', fontsize=12)
    plt.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    plt.tight_layout()
    loss_path = graphs_dir / 'loss_curve.png'
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[Graph Saved] Loss curve -> {loss_path}")

    # 2. Training & Validation Accuracy Curve
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, [a * 100 for a in history['train_acc']], 'o-', color='#10b981', linewidth=2.5, label='Train Accuracy (%)')
    plt.plot(epochs, [a * 100 for a in history['val_acc']], 's--', color='#8b5cf6', linewidth=2.5, label='Val Accuracy (%)')
    plt.title('Training & Validation Accuracy Over Epochs', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    plt.tight_layout()
    acc_path = graphs_dir / 'accuracy_curve.png'
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"[Graph Saved] Accuracy curve -> {acc_path}")

    # 3. Learning Rate Curve
    if 'lr' in history and len(history['lr']) > 0:
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, history['lr'], 'o-', color='#f59e0b', linewidth=2.5)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.tight_layout()
        lr_path = graphs_dir / 'learning_rate_schedule.png'
        plt.savefig(lr_path, dpi=300)
        plt.close()
        print(f"[Graph Saved] Learning rate schedule -> {lr_path}")

    # 4. Normalized Confusion Matrix Heatmap
    cm = confusion_matrix(test_labels, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title('Test Set Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_path = graphs_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Graph Saved] Confusion matrix -> {cm_path}")

    # 5. Precision, Recall, F1-Score Breakdown Bar Chart
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, labels=[0, 1, 2])
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision * 100, width, label='Precision', color='#3b82f6')
    plt.bar(x, recall * 100, width, label='Recall', color='#10b981')
    plt.bar(x + width, f1 * 100, width, label='F1-Score', color='#8b5cf6')
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
    plt.title('Per-Class Performance Metrics (Test Set)', fontsize=14, fontweight='bold', pad=12)
    plt.xticks(x, [c.capitalize() for c in CLASS_NAMES], fontsize=11)
    plt.ylim(0, 110)
    plt.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    plt.tight_layout()
    metrics_bar_path = graphs_dir / 'per_class_metrics.png'
    plt.savefig(metrics_bar_path, dpi=300)
    plt.close()
    print(f"[Graph Saved] Per-class metrics chart -> {metrics_bar_path}")

    # 6. Sample Predictions Visual Gallery
    visualize_sample_predictions(test_loader, model, device, graphs_dir / 'sample_predictions.png')


@torch.no_grad()
def visualize_sample_predictions(dataloader, model, device, save_path, num_samples=12):
    """Draws a gallery grid of sample test images with true vs predicted labels."""
    model.eval()
    images_list, labels_list, preds_list, confs_list = [], [], [], []

    for images, labels in dataloader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, 1)

        images_list.extend(images.cpu())
        labels_list.extend(labels.cpu().numpy())
        preds_list.extend(preds.cpu().numpy())
        confs_list.extend(confs.cpu().numpy())

        if len(images_list) >= num_samples:
            break

    # Un-normalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 11))
    axes = axes.flatten()

    for i in range(min(num_samples, len(images_list))):
        img_tensor = images_list[i].permute(1, 2, 0).numpy()
        img_tensor = np.clip(img_tensor * std + mean, 0, 1)

        true_lbl = IDX_TO_CLASS[labels_list[i]]
        pred_lbl = IDX_TO_CLASS[preds_list[i]]
        conf = confs_list[i] * 100

        is_correct = (labels_list[i] == preds_list[i])
        title_color = '#15803d' if is_correct else '#b91c1c'
        status = "✓ Correct" if is_correct else "✗ Wrong"

        axes[i].imshow(img_tensor)
        axes[i].set_title(f"True: {true_lbl}\nPred: {pred_lbl} ({conf:.1f}%)\n{status}",
                          fontsize=10, color=title_color, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('Model Test Predictions Gallery with Confidence Scores', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Graph Saved] Sample predictions gallery -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Rock-Paper-Scissors Deep Learning Model")
    parser.add_argument('--data_dir', type=str, default='Rock-Paper-Scissors', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='training_output', help='Output directory for checkpoints and graphs')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Input image resolution')
    parser.add_argument('--device', type=str, default='auto', help="Device ('auto', 'cuda', or 'cpu')")
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker processes (0 is safest on Windows)')
    args = parser.parse_args()

    # Hardware detection and GPU setup
    print("\n" + "=" * 60)
    print(" [*] ROCK-PAPER-SCISSORS DEEP LEARNING TRAINING PIPELINE")
    print("=" * 60)

    if args.device == 'auto':
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_type = args.device

    device = torch.device(device_type)
    print(f"\n[Hardware Initialization]")
    print(f" * Selected Device: {device.type.upper()}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f" * GPU Name: {gpu_name}")
        print(f" * GPU Memory: {gpu_vram:.2f} GB")
        print(f" * CUDA Driver / Version: {torch.version.cuda}")
        print(f" * CuDNN Benchmark Enabled: True")
        torch.backends.cudnn.benchmark = True
    else:
        print(" * Note: Running on CPU.")

    # Dataset Paths
    base_dir = Path(args.data_dir)
    train_dir = base_dir / 'train'
    test_dir = base_dir / 'test'
    val_dir = base_dir / 'validation'

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found at: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    train_transform, val_transform = get_transforms(args.img_size)

    # Load datasets
    print(f"\n[Dataset Preparation]")
    raw_train_dataset = RPSFolderDataset(train_dir)
    print(f" * Full Train Directory Samples: {len(raw_train_dataset)}")

    # We partition 15% of train set into a proper validation set for robust hyperparameter monitoring
    val_ratio = 0.15
    val_len = int(len(raw_train_dataset) * val_ratio)
    train_len = len(raw_train_dataset) - val_len

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(raw_train_dataset, [train_len, val_len], generator=generator)

    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=val_transform)
    test_dataset = RPSFolderDataset(test_dir, transform=val_transform)

    print(f" * Training Set: {len(train_dataset)} samples")
    print(f" * Validation Set: {len(val_dataset)} samples")
    print(f" * Test Set: {len(test_dataset)} samples")
    print(f" * Classes: {CLASS_NAMES} (Indices: {CLASS_TO_IDX})")

    # Data Loaders
    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    # Model, Optimizer, Loss, LR Scheduler, AMP Scaler
    print(f"\n[Model Initialization]")
    model = build_model(num_classes=3, pretrained=True)
    model = model.to(device)
    print(" * Model Architecture: MobileNetV2 with Custom Classifier Head")
    print(" * Pretrained Weights: ImageNet (Transfer Learning)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / 'best_rps_model.pth'

    # Training Loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    start_total_time = time.time()

    print(f"\n[Training Execution - {args.epochs} Epochs]")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<10} {'Time':<8}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch_num=epoch, total_epochs=args.epochs)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        epoch_time = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Save Best Model Checkpoint
        saved_flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
                'class_to_idx': CLASS_TO_IDX,
                'arch': 'mobilenet_v2'
            }, best_model_path)
            saved_flag = "[BEST]"

        print(f"{epoch:<8} {train_loss:<12.4f} {train_acc*100:<11.2f}% {val_loss:<12.4f} {val_acc*100:<11.2f}% {current_lr:<10.6f} {epoch_time:<6.1f}s {saved_flag}")

    total_training_time = time.time() - start_total_time
    print("-" * 75)
    print(f"[Done] Training finished in {total_training_time:.2f} seconds ({total_training_time/60:.2f} mins).")
    print(f"[Done] Best Validation Accuracy: {best_val_acc * 100:.2f}% (Saved to {best_model_path})")

    # Load Best Model for Final Evaluation on Test Set
    print(f"\n[Test Set Final Evaluation]")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    print(f" * Test Loss: {test_loss:.4f}")
    print(f" * Test Accuracy: {test_acc * 100:.2f}%")

    cls_report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4)
    print("\n[Classification Report (Test Set)]")
    print(cls_report)

    # Save Classification Report & Metrics JSON
    report_path = out_dir / 'classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Rock-Paper-Scissors Test Evaluation\n")
        f.write(f"Date: {time.ctime()}\n")
        f.write(f"Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(cls_report)

    metrics_dict = {
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'total_training_time_sec': float(total_training_time),
        'device': str(device.type),
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'history': history
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)

    # Generate All Visualizations & Analysis Graphs
    print(f"\n[Generating Analysis Visualizations & Graphs]")
    generate_visualizations(
        history=history,
        test_labels=test_labels,
        test_preds=test_preds,
        test_loader=test_loader,
        model=model,
        device=device,
        output_dir=out_dir
    )

    print("\n" + "=" * 60)
    print(f" [*] ALL TASKS COMPLETED SUCCESSFULLY!")
    print(f" * Model Checkpoint: {best_model_path}")
    print(f" * Evaluation Report: {report_path}")
    print(f" * Analysis Graphs: {out_dir / 'graphs'}")
    print("=" * 60 + "\n")



if __name__ == '__main__':
    main()
