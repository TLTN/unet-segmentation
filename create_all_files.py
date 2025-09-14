import os
import yaml


def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'src',
        'src/model',
        'src/data',
        'src/training',
        'src/utils',
        'scripts',
        'data',
        'data/raw',
        'data/raw/images',
        'data/raw/masks',
        'data/processed',
        'data/processed/train/images',
        'data/processed/train/masks',
        'data/processed/val/images',
        'data/processed/val/masks',
        'data/processed/test/images',
        'data/processed/test/masks',
        'data/sample_data',
        'models',
        'models/checkpoints',
        'results',
        'logs',
        'tests',
        'notebooks'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}")


def create_config_yaml():
    """Create config.yaml file"""
    config = {
        'data': {
            'raw_data_dir': 'data/raw',
            'processed_data_dir': 'data/processed',
            'image_dir': 'images',
            'mask_dir': 'masks'
        },
        'model': {
            'input_size': [256, 256, 3],
            'architecture': 'unet'
        },
        'training': {
            'batch_size': 8,
            'epochs': 50,
            'learning_rate': 0.0001,
            'validation_split': 0.2,
            'test_split': 0.1
        },
        'augmentation': {
            'rotation_range': 0.2,
            'width_shift_range': 0.05,
            'height_shift_range': 0.05,
            'shear_range': 0.05,
            'zoom_range': 0.05,
            'horizontal_flip': True,
            'vertical_flip': False,
            'fill_mode': 'nearest'
        },
        'callbacks': {
            'early_stopping': {
                'patience': 20,
                'monitor': 'val_dice',
                'mode': 'max'
            },
            'reduce_lr': {
                'factor': 0.2,
                'patience': 10,
                'min_lr': 1e-7
            }
        },
        'paths': {
            'model_dir': 'models',
            'results_dir': 'results',
            'logs_dir': 'logs'
        }
    }

    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print("📝 Created: config.yaml")


def create_init_files():
    """Create all __init__.py files"""
    init_files = [
        'src/__init__.py',
        'src/model/__init__.py',
        'src/data/__init__.py',
        'src/training/__init__.py',
        'src/utils/__init__.py',
        'tests/__init__.py'
    ]

    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('"""Package initialization"""\n')
        print(f"📄 Created: {init_file}")


def create_unet_model():
    """Create src/model/unet.py"""
    content = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block used in U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture for image segmentation"""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (down sampling path)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (up sampling path)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Up-sampling
            skip_connection = skip_connections[idx // 2]

            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)  # Double conv

        return torch.sigmoid(self.final_conv(x))
'''

    with open('src/model/unet.py', 'w') as f:
        f.write(content)

    print("📄 Created: src/model/unet.py")


def create_trainer():
    """Create src/training/trainer.py"""
    content = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

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
        if TENSORBOARD_AVAILABLE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(self.logs_dir, f'tensorboard_{timestamp}')
            try:
                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                self.writer = None
        else:
            self.writer = None

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []

        self.best_val_dice = 0.0
        self.early_stop_counter = 0
        self.early_stop_patience = config['callbacks']['early_stopping']['patience']

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0

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

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{metrics["dice"]:.4f}'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_dice = running_dice / len(train_loader)

        return epoch_loss, epoch_dice

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0

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

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{metrics["dice"]:.4f}'
                })

        epoch_loss = running_loss / len(val_loader)
        epoch_dice = running_dice / len(val_loader)

        return epoch_loss, epoch_dice

    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_dice': val_dice,
            'config': self.config
        }

        # Save best model
        if is_best:
            best_path = os.path.join(self.model_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved with Dice: {val_dice:.4f}")

    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        print("🚀 Starting training...")

        for epoch in range(epochs):
            print(f"\\n📍 Epoch {epoch+1}/{epochs}")
            print("-" * 50)

            # Train
            train_loss, train_dice = self.train_epoch(train_loader)

            # Validate
            val_loss, val_dice = self.validate_epoch(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dices.append(train_dice)
            self.val_dices.append(val_dice)

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                self.writer.add_scalar('Dice/Train', train_dice, epoch)
                self.writer.add_scalar('Dice/Val', val_dice, epoch)

            # Print epoch results
            print(f"📊 Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"📊 Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

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
            'val_dices': self.val_dices
        }

    def plot_training_history(self, history):
        """Plot and save training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss
        axes[0].plot(history['train_losses'], label='Training Loss', linewidth=2)
        axes[0].plot(history['val_losses'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Dice Coefficient
        axes[1].plot(history['train_dices'], label='Training Dice', linewidth=2)
        axes[1].plot(history['val_dices'], label='Validation Dice', linewidth=2)
        axes[1].set_title('Dice Coefficient', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training plots saved to {plot_path}")
        plt.show()
'''

    with open('src/training/trainer.py', 'w') as f:
        f.write(content)

    print("📄 Created: src/training/trainer.py")


def create_dataset():
    """Create src/data/dataset.py"""
    content = '''import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size

        self.images = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        # Find corresponding mask
        img_name = os.path.splitext(self.images[index])[0]
        mask_extensions = ['.png', '.jpg', '.jpeg']
        mask_path = None

        for ext in mask_extensions:
            potential_mask = os.path.join(self.mask_dir, img_name + ext)
            if os.path.exists(potential_mask):
                mask_path = potential_mask
                break

        if mask_path is None:
            # Try with same extension as image
            img_ext = os.path.splitext(self.images[index])[1]
            mask_path = os.path.join(self.mask_dir, img_name + img_ext)

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize
        image = image.resize(self.image_size)
        mask = mask.resize(self.image_size)

        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        return image, mask

def get_train_transform(image_size=(256, 256)):
    """Get training transforms"""
    return None  # Basic version without augmentation

def get_val_transform(image_size=(256, 256)):
    """Get validation transforms"""
    return None  # Basic version without augmentation
'''

    with open('src/data/dataset.py', 'w') as f:
        f.write(content)

    print("📄 Created: src/data/dataset.py")


def create_config_utils():
    """Create src/utils/config.py"""
    content = '''import yaml
import os

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        raise

def create_directories(config):
    """Create necessary directories"""
    dirs = [
        config['paths']['model_dir'],
        config['paths']['results_dir'],
        config['paths']['logs_dir'],
        os.path.join(config['paths']['model_dir'], 'checkpoints'),
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
'''

    with open('src/utils/config.py', 'w') as f:
        f.write(content)

    print("📄 Created: src/utils/config.py")


def create_train_script():
    """Create scripts/train.py"""
    content = '''import argparse
import os
import sys
import logging
from datetime import datetime

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

def setup_logging():
    """Setup simple logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()

    parser = argparse.ArgumentParser(description='Train PyTorch U-Net model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    try:
        logger.info("🎯 Starting PyTorch U-Net Training")
        logger.info("=" * 50)

        # Import modules
        logger.info("📦 Loading modules...")
        import torch
        from torch.utils.data import DataLoader

        from model.unet import UNet
        from data.dataset import SegmentationDataset, get_train_transform, get_val_transform
        from training.trainer import UNetTrainer
        from utils.config import load_config, create_directories

        # Load config
        logger.info("📋 Loading configuration...")
        config = load_config(args.config)

        # Override config with command line args
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size

        # Create directories
        create_directories(config)

        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {device}")

        # Check data
        train_img_dir = os.path.join(config['data']['processed_data_dir'], 'train', 'images')
        val_img_dir = os.path.join(config['data']['processed_data_dir'], 'val', 'images')

        if not os.path.exists(train_img_dir) or len(os.listdir(train_img_dir)) == 0:
            logger.error(f"❌ No training data found in {train_img_dir}")
            logger.info("Run data preparation first!")
            return 1

        # Create datasets
        logger.info("📊 Creating datasets...")
        input_size = tuple(config['model']['input_size'][:2])

        train_dataset = SegmentationDataset(
            image_dir=os.path.join(config['data']['processed_data_dir'], 'train', 'images'),
            mask_dir=os.path.join(config['data']['processed_data_dir'], 'train', 'masks'),
            transform=get_train_transform(input_size),
            image_size=input_size
        )

        val_dataset = SegmentationDataset(
            image_dir=os.path.join(config['data']['processed_data_dir'], 'val', 'images'),
            mask_dir=os.path.join(config['data']['processed_data_dir'], 'val', 'masks'),
            transform=get_val_transform(input_size),
            image_size=input_size
        )

        logger.info(f"📊 Training samples: {len(train_dataset)}")
        logger.info(f"📊 Validation samples: {len(val_dataset)}")

        if len(train_dataset) == 0:
            logger.error("❌ No training data found!")
            return 1

        # Create data loaders
        batch_size = config['training']['batch_size']

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )

        # Create model
        logger.info("🏗️ Building U-Net model...")
        in_channels = config['model']['input_size'][2]
        model = UNet(in_channels=in_channels, out_channels=1)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🔢 Model parameters: {total_params:,}")

        # Create trainer
        logger.info("🏃 Initializing trainer...")
        trainer = UNetTrainer(model, config, device)

        # Start training
        epochs = config['training']['epochs']
        logger.info(f"🚀 Starting training for {epochs} epochs...")

        history = trainer.train(train_loader, val_loader, epochs)

        # Plot results
        logger.info("📈 Generating plots...")
        trainer.plot_training_history(history)

        # Training summary
        best_dice = max(history['val_dices']) if history['val_dices'] else 0
        logger.info("\\n🎉 Training Summary:")
        logger.info(f"   📊 Best Validation Dice: {best_dice:.4f}")
        logger.info(f"   💾 Model saved: models/best_model.pth")
        logger.info(f"   📈 Plots saved: results/training_history.png")

        logger.info("✅ Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
'''

    with open('scripts/train.py', 'w') as f:
        f.write(content)

    print("📄 Created: scripts/train.py")


def create_synthetic_data_generator():
    """Create data/sample_data/synthetic_data_generator.py"""
    content = '''import numpy as np
import cv2
import os
from tqdm import tqdm
import random

def generate_synthetic_segmentation_data(num_samples=1000, image_size=(256, 256)):
    """Generate synthetic images and masks for segmentation"""

    # Create directories
    os.makedirs("data/raw/images", exist_ok=True)
    os.makedirs("data/raw/masks", exist_ok=True)

    print(f"🎨 Generating {num_samples} synthetic samples...")

    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Create base image
        image = np.random.randint(50, 150, (*image_size, 3), dtype=np.uint8)

        # Add gradient background
        x_grad = np.linspace(0.5, 1.0, image_size[1])
        for c in range(3):
            image[:, :, c] = (image[:, :, c] * x_grad).astype(np.uint8)

        # Create mask
        mask = np.zeros(image_size, dtype=np.uint8)

        # Add 1-3 random shapes
        num_shapes = random.randint(1, 3)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'ellipse'])

            if shape_type == 'circle':
                center = (random.randint(40, image_size[1]-40), 
                         random.randint(40, image_size[0]-40))
                radius = random.randint(25, 70)
                color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))

                cv2.circle(image, center, radius, color, -1)
                cv2.circle(mask, center, radius, 255, -1)

            elif shape_type == 'rectangle':
                pt1 = (random.randint(20, image_size[1]//2), 
                       random.randint(20, image_size[0]//2))
                pt2 = (random.randint(image_size[1]//2, image_size[1]-20), 
                       random.randint(image_size[0]//2, image_size[0]-20))
                color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))

                cv2.rectangle(image, pt1, pt2, color, -1)
                cv2.rectangle(mask, pt1, pt2, 255, -1)

            elif shape_type == 'ellipse':
                center = (random.randint(60, image_size[1]-60), 
                         random.randint(60, image_size[0]-60))
                axes = (random.randint(30, 80), random.randint(30, 80))
                angle = random.randint(0, 180)
                color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))

                cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)
                cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        # Add some noise
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Save files
        cv2.imwrite(f"data/raw/images/sample_{i:04d}.jpg", image)
        cv2.imwrite(f"data/raw/masks/sample_{i:04d}.png", mask)

    print(f"✅ Generated {num_samples} samples!")
    print(f"📁 Images: data/raw/images/")
    print(f"📁 Masks: data/raw/masks/")

if __name__ == "__main__":
    generate_synthetic_segmentation_data(num_samples=1000, image_size=(256, 256))
    print("\\nNext steps:")
    print("1. python scripts/data_preparation.py")
    print("2. python scripts/train.py")
'''

    with open('data/sample_data/synthetic_data_generator.py', 'w') as f:
        f.write(content)

    print("📄 Created: data/sample_data/synthetic_data_generator.py")


def create_data_preparation_script():
    """Create scripts/data_preparation.py"""
    content = '''import os
import sys
import traceback
import yaml
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        raise

def prepare_dataset(config):
    """Prepare dataset by splitting into train/val/test"""
    raw_img_dir = os.path.join(config['data']['raw_data_dir'], config['data']['image_dir'])
    raw_mask_dir = os.path.join(config['data']['raw_data_dir'], config['data']['mask_dir'])

    # Get all image files
    image_files = sorted([f for f in os.listdir(raw_img_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(raw_mask_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Found {len(image_files)} images and {len(mask_files)} masks")

    # Match files by name
    image_names = [os.path.splitext(f)[0] for f in image_files]
    mask_names = [os.path.splitext(f)[0] for f in mask_files]

    pairs = []
    for img_name, img_file in zip(image_names, image_files):
        if img_name in mask_names:
            mask_idx = mask_names.index(img_name)
            pairs.append((img_file, mask_files[mask_idx]))

    print(f"Matched {len(pairs)} image-mask pairs")

    # Split data
    random.shuffle(pairs)
    val_split = config['training']['validation_split']
    test_split = config['training']['test_split']

    # First split: train+val vs test
    train_val, test = train_test_split(pairs, test_size=test_split, random_state=42)

    # Second split: train vs val
    val_size = val_split / (1 - test_split)
    train, val = train_test_split(train_val, test_size=val_size, random_state=42)

    print(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    # Copy files to processed directories
    processed_dir = config['data']['processed_data_dir']
    splits = [('train', train), ('val', val), ('test', test)]

    for split_name, file_pairs in splits:
        img_dst = os.path.join(processed_dir, split_name, 'images')
        mask_dst = os.path.join(processed_dir, split_name, 'masks')

        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(mask_dst, exist_ok=True)

        for img_file, mask_file in tqdm(file_pairs, desc=f"Copying {split_name}"):
            # Copy image
            src_img = os.path.join(raw_img_dir, img_file)
            dst_img = os.path.join(img_dst, img_file)
            shutil.copy2(src_img, dst_img)

            # Copy mask
            src_mask = os.path.join(raw_mask_dir, mask_file)
            dst_mask = os.path.join(mask_dst, mask_file)
            shutil.copy2(src_mask, dst_mask)

    print("Dataset preparation completed!")

def main():
    print("🔄 Starting data preparation...")

    try:
        # Load config
        config = load_config("config.yaml")

        # Check if raw data exists
        raw_img_dir = os.path.join(config['data']['raw_data_dir'], config['data']['image_dir'])
        raw_mask_dir = os.path.join(config['data']['raw_data_dir'], config['data']['mask_dir'])

        if not os.path.exists(raw_img_dir) or not os.path.exists(raw_mask_dir):
            print("❌ Raw data not found!")
            print("Run: python data/sample_data/synthetic_data_generator.py")
            return 1

        # Check if directories contain files
        img_files = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(img_files) == 0:
            print("❌ No image files found!")
            return 1

        # Prepare dataset
        prepare_dataset(config)

        print("✅ Data preparation completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''

    with open('scripts/data_preparation.py', 'w') as f:
        f.write(content)

    print("📄 Created: scripts/data_preparation.py")


def create_requirements_txt():
    """Create requirements.txt"""
    requirements = '''torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.5.0
matplotlib>=3.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
pyyaml>=6.0
pillow>=8.3.0
tqdm>=4.62.0
tensorboard>=2.8.0
'''

    with open('requirements.txt', 'w') as f:
        f.write(requirements)

    print("📄 Created: requirements.txt")


def create_readme():
    """Create README.md"""
    content = '''# PyTorch U-Net Image Segmentation

A complete implementation of U-Net for image segmentation using PyTorch.

## Quick Start

1. **Generate synthetic data:**
   ```bash
   python data/sample_data/synthetic_data_generator.py
   ```

2. **Prepare dataset:**
   ```bash
   python scripts/data_preparation.py
   ```

3. **Train model:**
   ```bash
   python scripts/train.py
   ```

## Project Structure

```
├── data/
│   ├── raw/              # Raw images and masks
│   ├── processed/        # Train/val/test splits
│   └── sample_data/      # Synthetic data generator
├── src/
│   ├── model/           # U-Net architecture
│   ├── data/            # Dataset classes
│   ├── training/        # Training pipeline
│   └── utils/           # Utilities
├── scripts/             # Training and data scripts
├── models/              # Saved models
├── results/             # Training plots
└── logs/                # TensorBoard logs
```

## Features

- ✅ PyTorch U-Net implementation
- ✅ Synthetic data generation
- ✅ Data augmentation
- ✅ Training with validation
- ✅ TensorBoard logging
- ✅ Model checkpointing
- ✅ Training visualization

## Requirements

- Python 3.8+
- PyTorch 1.12+
- OpenCV
- matplotlib
- scikit-learn
- tqdm
- pyyaml

Install all requirements:
```bash
pip install -r requirements.txt
```

## Configuration

Modify `config.yaml` to adjust:
- Model parameters
- Training settings
- Data paths
- Augmentation options

## GPU Support

The model automatically detects and uses CUDA if available.
'''

    with open('README.md', 'w') as f:
        f.write(content)

    print("📄 Created: README.md")


def main():
    print("🚀 Creating Complete PyTorch U-Net Project")
    print("=" * 60)

    print("\\n📁 Creating directory structure...")
    create_directory_structure()

    print("\\n📝 Creating configuration files...")
    create_config_yaml()
    create_requirements_txt()
    create_readme()

    print("\\n📄 Creating Python files...")
    create_init_files()
    create_unet_model()
    create_trainer()
    create_dataset()
    create_config_utils()
    create_train_script()
    create_synthetic_data_generator()
    create_data_preparation_script()

    print("\\n🎉 PROJECT CREATION COMPLETE!")
    print("=" * 60)
    print("\\n📋 Next steps:")
    print("1. python data/sample_data/synthetic_data_generator.py")
    print("2. python scripts/data_preparation.py")
    print("3. python scripts/train.py")
    print("\\n🎯 Your complete PyTorch U-Net project is ready!")


if __name__ == "__main__":
    main()