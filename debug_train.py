import os
import sys
import traceback


def check_environment():
    """Check if all dependencies are available"""
    print("🔍 Checking Environment...")
    print("=" * 50)

    # Check Python version
    print(f"🐍 Python: {sys.version}")

    # Check PyTorch
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"🖥️  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ PyTorch not found: {e}")
        return False

    # Check other dependencies
    dependencies = [
        'torchvision', 'numpy', 'cv2', 'matplotlib', 'tqdm', 'sklearn', 'yaml', 'PIL'
    ]

    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif dep == 'yaml':
                import yaml
                print(f"✅ PyYAML: Available")
            elif dep == 'PIL':
                from PIL import Image
                print(f"✅ Pillow: Available")
            else:
                module = __import__(dep)
                if hasattr(module, '__version__'):
                    print(f"✅ {dep}: {module.__version__}")
                else:
                    print(f"✅ {dep}: Available")
        except ImportError as e:
            print(f"❌ {dep}: Missing")
            return False

    return True


def check_project_structure():
    """Check if project structure is correct"""
    print("\n📁 Checking Project Structure...")
    print("=" * 50)

    required_files = [
        'config.yaml',
        'src/model/unet.py',
        'src/training/trainer.py',
        'src/data/dataset.py',
        'src/utils/config.py',
        'scripts/train.py'
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)

    return len(missing_files) == 0


def check_data():
    """Check if data is available"""
    print("\n📊 Checking Data...")
    print("=" * 50)

    data_dirs = [
        'data/processed/train/images',
        'data/processed/train/masks',
        'data/processed/val/images',
        'data/processed/val/masks'
    ]

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"✅ {data_dir}: {len(img_files)} files")

            if len(img_files) == 0:
                print(f"⚠️  Directory is empty!")
        else:
            print(f"❌ {data_dir}: Directory not found")

    return True


def test_imports():
    """Test importing all required modules"""
    print("\n🧪 Testing Imports...")
    print("=" * 50)

    try:
        # Add src to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        sys.path.insert(0, src_path)

        # Test basic imports
        print("Testing basic imports...")
        import torch
        from torch.utils.data import DataLoader
        print("✅ PyTorch imports OK")

        # Test project imports
        print("Testing project imports...")
        from model.unet import UNet
        print("✅ UNet import OK")

        from training.trainer import UNetTrainer
        print("✅ UNetTrainer import OK")

        from data.dataset import SegmentationDataset
        print("✅ SegmentationDataset import OK")

        from utils.config import load_config
        print("✅ Config utils import OK")

        print("🎉 All imports successful!")
        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False


def run_minimal_train():
    """Run a minimal version of training to debug"""
    print("\n🚀 Running Minimal Training Test...")
    print("=" * 50)

    try:
        # Add src to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        sys.path.insert(0, src_path)

        import torch
        from torch.utils.data import DataLoader
        import yaml

        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print(f"📋 Config loaded: {len(config)} sections")

        # Import modules
        from model.unet import UNet
        from data.dataset import SegmentationDataset, get_train_transform, get_val_transform
        from training.trainer import UNetTrainer

        print("✅ All modules imported successfully")

        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")

        # Create model
        model = UNet(in_channels=3, out_channels=1)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🏗️  Model created with {total_params:,} parameters")

        # Test dataset creation
        train_dataset = SegmentationDataset(
            image_dir='data/processed/train/images',
            mask_dir='data/processed/train/masks',
            transform=get_train_transform((256, 256))
        )

        print(f"📊 Training dataset: {len(train_dataset)} samples")

        if len(train_dataset) == 0:
            print("❌ No training data found!")
            return False

        # Test dataloader
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)

        # Test one batch
        for images, masks in train_loader:
            print(f"✅ Data batch: images {images.shape}, masks {masks.shape}")
            break

        print("🎉 Minimal test completed successfully!")
        print("The training script should work now.")

        return True

    except Exception as e:
        print(f"❌ Minimal training test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("🔧 PyTorch U-Net Training Debugger")
    print("=" * 70)

    # Step 1: Check environment
    if not check_environment():
        print("\n💡 Install missing dependencies:")
        print("pip install torch torchvision opencv-python matplotlib tqdm scikit-learn pyyaml pillow")
        return

    # Step 2: Check project structure
    if not check_project_structure():
        print("\n💡 Some files are missing. Make sure you created all required files.")
        return

    # Step 3: Check data
    check_data()

    # Step 4: Test imports
    if not test_imports():
        print("\n💡 Fix import issues before proceeding.")
        return

    # Step 5: Run minimal test
    if not run_minimal_train():
        print("\n💡 There are issues with the training pipeline.")
        return

    print("\n🎯 DIAGNOSIS COMPLETE!")
    print("=" * 50)
    print("✅ Everything looks good. Try running the training script again:")
    print("   python scripts/train.py")


if __name__ == "__main__":
    main()