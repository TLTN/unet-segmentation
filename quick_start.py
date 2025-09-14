import os
import subprocess
import sys
import argparse


def run_command(cmd, description):
    """Run shell command with description"""
    print(f"\n{description}")
    print("=" * 50)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def check_pytorch():
    """Check if PyTorch is installed"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("CUDA not available - will use CPU")
        return True
    except ImportError:
        print("PyTorch not installed!")
        return False


def main():
    parser = argparse.ArgumentParser(description='PyTorch U-Net Quick Start')
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip package installation')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Install CPU-only version of PyTorch')
    args = parser.parse_args()

    print("PyTorch U-Net Segmentation Project Quick Start")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists('config.yaml'):
        print("Error: config.yaml not found. Make sure you're in the project root directory.")
        return

    # Step 1: Install requirements
    if not args.skip_install:
        print("Step 1: Installing requirements...")
        if args.cpu_only:
            # Install CPU-only PyTorch
            torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            if not run_command(torch_cmd, "Installing PyTorch (CPU only)"):
                return
        else:
            # Install PyTorch with CUDA support
            torch_cmd = "pip install torch torchvision torchaudio"
            if not run_command(torch_cmd, "Installing PyTorch"):
                return

        # Install other requirements
        other_cmd = "pip install opencv-python matplotlib scikit-learn pillow tqdm pyyaml jupyter seaborn albumentations tensorboard pynvml"
        if not run_command(other_cmd, "Installing other requirements"):
            return

    # Check PyTorch installation
    if not check_pytorch():
        print("Please install PyTorch first!")
        return

    # Step 2: Generate sample data
    print("\nStep 2: Generating sample dataset...")
    if not run_command("python data/sample_data/synthetic_data_generator.py",
                       "Creating synthetic segmentation dataset"):
        return

    # Step 3: Prepare dataset
    print("\nStep 3: Preparing dataset...")
    if not run_command("python scripts/data_preparation.py",
                       "Splitting data into train/val/test"):
        return

    # Step 4: Start training
    print("\nStep 4: Starting training...")
    print("Note: Training will start now. You can stop it anytime with Ctrl+C")
    input("Press Enter to continue or Ctrl+C to exit...")

    run_command("python scripts/train.py", "Training PyTorch U-Net model")

    print("\nQuick start completed!")
    print("Check the 'models/' directory for saved models.")
    print("Check the 'results/' directory for training plots and metrics.")


if __name__ == "__main__":
    main()