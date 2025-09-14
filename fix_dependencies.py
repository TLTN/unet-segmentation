import subprocess
import sys


def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing PyTorch U-Net Dependencies")
    print("=" * 50)

    # Basic requirements
    basic_deps = [
        "torch",
        "torchvision",
        "opencv-python",
        "matplotlib",
        "tqdm",
        "scikit-learn",
        "pyyaml",
        "pillow",
        "numpy"
    ]

    # Optional but recommended
    optional_deps = [
        "albumentations",
        "tensorboard",
    ]

    print("Installing basic dependencies...")
    for dep in basic_deps:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")

    print("\nInstalling optional dependencies...")
    for dep in optional_deps:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {dep} (optional)")

    print("\n🎉 Dependencies installation complete!")
    print("Now try running the training script again.")


if __name__ == "__main__":
    install_dependencies()

print("\n" + "=" * 80)
print("🔧 DEBUG TOOLS CREATED!")
print("=" * 80)
print("\n📋 What to do:")
print("\n1️⃣  First, run diagnostics:")
print("   python debug_train.py")
print("\n2️⃣  If dependencies are missing:")
print("   python fix_dependencies.py")
print("\n3️⃣  Try simplified training:")
print("   python simple_train.py")
print("\n4️⃣  Then run full training:")
print("   python scripts/train.py")
print("\n💡 The pynvml warning is harmless and can be ignored.")
print("🎯 These tools will help identify and fix the issue!")