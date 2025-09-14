import sys
import os

# Add src to path
sys.path.append('src')

import torch
from torch.utils.data import DataLoader
import yaml

from model.unet import UNet
from data.dataset import SegmentationDataset
from training.trainer import Trainer

def main():
    print("🚀 Starting PyTorch U-Net Training")
    print("=" * 50)

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU (will be very slow)")
        response = input("Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            return

    # Dataset
    train_dataset = SegmentationDataset(
        'data/processed/train/images',
        'data/processed/train/masks'
    )
    val_dataset = SegmentationDataset(
        'data/processed/val/images', 
        'data/processed/val/masks'
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return

    # Data loaders
    batch_size = config['training']['batch_size']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = UNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Trainer
    trainer = Trainer(model, device)

    # Training
    epochs = config['training']['epochs']
    print(f"Training for {epochs} epochs...")

    try:
        history = trainer.train(train_loader, val_loader, epochs)
        trainer.plot_history(history)

        print("\n🎉 Training completed successfully!")
        print("📊 Best model saved to models/best_model.pth")
        print("📈 Training plot saved to results/training_history.png")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
