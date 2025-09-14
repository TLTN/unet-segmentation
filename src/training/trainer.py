import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datetime import datetime


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
                predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        total = (predictions + targets).sum()
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        return self.dice_weight * dice + self.bce_weight * bce


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate segmentation metrics"""
    pred_binary = (predictions > threshold).float()
    targets = targets.float()

    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)

    # Calculate metrics
    intersection = (pred_flat * target_flat).sum()

    # Dice coefficient
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)

    # IoU
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = intersection / (union + 1e-6)

    # Pixel accuracy
    accuracy = (pred_flat == target_flat).float().mean()

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'accuracy': accuracy.item()
    }


class UNetTrainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Create directories
        self.model_dir = config['paths']['model_dir']
        self.results_dir = config['paths']['results_dir']
        self.logs_dir = config['paths']['logs_dir']

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Setup loss function and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.2, patience=10, verbose=True
        )

        # TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(self.logs_dir, f'tensorboard_{timestamp}')
        try:
            self.writer = SummaryWriter(log_dir=log_dir)
        except Exception as e:
            print(f"Warning: TensorBoard not available: {e}")
            self.writer = None

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        self.train_ious = []
        self.val_ious = []

        self.best_val_dice = 0.0
        self.early_stop_counter = 0
        self.early_stop_patience = config['callbacks']['early_stopping']['patience']

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(outputs, masks)

            running_loss += loss.item()
            running_dice += metrics['dice']
            running_iou += metrics['iou']

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{metrics["dice"]:.4f}',
                'IoU': f'{metrics["iou"]:.4f}'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_dice = running_dice / len(train_loader)
        epoch_iou = running_iou / len(train_loader)

        return epoch_loss, epoch_dice, epoch_iou

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                metrics = calculate_metrics(outputs, masks)

                running_loss += loss.item()
                running_dice += metrics['dice']
                running_iou += metrics['iou']

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{metrics["dice"]:.4f}',
                    'IoU': f'{metrics["iou"]:.4f}'
                })

        epoch_loss = running_loss / len(val_loader)
        epoch_dice = running_dice / len(val_loader)
        epoch_iou = running_iou / len(val_loader)

        return epoch_loss, epoch_dice, epoch_iou

    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_dice': val_dice,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.model_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved with Dice: {val_dice:.4f}")

    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        print("🚀 Starting training...")

        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # Train
            train_loss, train_dice, train_iou = self.train_epoch(train_loader)

            # Validate
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dices.append(train_dice)
            self.val_dices.append(val_dice)
            self.train_ious.append(train_iou)
            self.val_ious.append(val_iou)

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                self.writer.add_scalar('Dice/Train', train_dice, epoch)
                self.writer.add_scalar('Dice/Val', val_dice, epoch)
                self.writer.add_scalar('IoU/Train', train_iou, epoch)
                self.writer.add_scalar('IoU/Val', val_iou, epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch results
            print(f"📊 Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            print(f"📊 Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

            # Save checkpoint
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            self.save_checkpoint(epoch + 1, val_dice, is_best)

            # Early stopping
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"⏹️ Early stopping triggered after {epoch + 1} epochs")
                break

        if self.writer is not None:
            self.writer.close()
        print("✅ Training completed!")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious
        }

    def plot_training_history(self, history):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(history['train_losses'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_losses'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Dice Coefficient
        axes[0, 1].plot(history['train_dices'], label='Training Dice', linewidth=2)
        axes[0, 1].plot(history['val_dices'], label='Validation Dice', linewidth=2)
        axes[0, 1].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # IoU Score
        axes[1, 0].plot(history['train_ious'], label='Training IoU', linewidth=2)
        axes[1, 0].plot(history['val_ious'], label='Validation IoU', linewidth=2)
        axes[1, 0].set_title('IoU Score', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        epochs = range(1, len(history['train_losses']) + 1)
        current_lr = self.optimizer.param_groups[0]['lr']
        lrs = [current_lr] * len(epochs)
        axes[1, 1].plot(epochs, lrs, 'g-', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training plots saved to {plot_path}")
        plt.show()