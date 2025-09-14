# ==============================================================================
# python tests/test_predict.py --interactive
# ==============================================================================
import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import argparse

# Add src to path - điều chỉnh cho cấu trúc của bạn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Lên 1 cấp từ tests/
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)


def load_model(model_path, device):
    """Load trained model"""
    try:
        from model.unet import UNet
        model = UNet()

        # Kiểm tra xem file model có tồn tại không
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            # Thử tìm ở các vị trí khác
            alternative_paths = [
                os.path.join(project_root, 'models', 'best_model.pth'),
                os.path.join(project_root, 'scripts', 'models', 'best_model.pth'),
                'models/best_model.pth',
                'scripts/models/best_model.pth'
            ]

            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"✅ Found model at: {model_path}")
                    break
            else:
                print("❌ Model not found in any expected location!")
                return None

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess input image"""
    try:
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path

        # Store original size for later
        original_size = image.size
        original_image = np.array(image.resize(target_size))

        # Resize and normalize
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        return image_tensor, original_size, original_image
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None, None, None


def postprocess_prediction(prediction, original_size, threshold=0.5):
    """Postprocess prediction to original size"""
    try:
        # Remove batch dimension and convert to numpy
        pred_np = prediction.squeeze().cpu().numpy()

        # Apply threshold
        pred_binary = (pred_np > threshold).astype(np.uint8) * 255

        # Resize back to original size
        pred_resized = cv2.resize(pred_binary, original_size, interpolation=cv2.INTER_NEAREST)

        return pred_np, pred_binary, pred_resized
    except Exception as e:
        print(f"❌ Error postprocessing: {e}")
        return None, None, None


def predict_single_image(image_path, model_path=None, output_dir=None, threshold=0.5, show_plot=True):
    """Predict segmentation mask for a single image"""

    # Setup paths
    if model_path is None:
        model_path = os.path.join(project_root, 'scripts', 'models', 'best_model.pth')

    if output_dir is None:
        output_dir = os.path.join(project_root, 'tests', 'results')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # Load model
    model = load_model(model_path, device)
    if model is None:
        return False

    # Preprocess image
    print(f"📷 Processing image: {image_path}")
    image_tensor, original_size, image_display = preprocess_image(image_path)
    if image_tensor is None:
        return False

    # Make prediction
    print("🔮 Making prediction...")
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)

    # Postprocess
    pred_prob, pred_binary, pred_resized = postprocess_prediction(
        prediction, original_size, threshold
    )
    if pred_prob is None:
        return False

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save probability map
    prob_path = os.path.join(output_dir, f"{base_name}_probability.png")
    cv2.imwrite(prob_path, (pred_prob * 255).astype(np.uint8))

    # Save binary mask
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, pred_binary)  # Use 256x256 version

    # Save resized mask (original size)
    mask_original_path = os.path.join(output_dir, f"{base_name}_mask_original_size.png")
    cv2.imwrite(mask_original_path, pred_resized)

    print(f"💾 Results saved:")
    print(f"   📄 Probability map: {prob_path}")
    print(f"   📄 Binary mask (256x256): {mask_path}")
    print(f"   📄 Binary mask (original size): {mask_original_path}")

    # Show results
    if show_plot:
        plot_results(image_display, pred_prob, pred_binary, base_name, threshold)

    return True


def plot_results(original_image, pred_prob, pred_binary, title, threshold):
    """Plot prediction results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')

    # Prediction probability
    im1 = axes[0, 1].imshow(pred_prob, cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].set_title('Prediction Probability', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Binary prediction
    axes[1, 0].imshow(pred_binary, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Binary Mask (threshold={threshold})', fontweight='bold')
    axes[1, 0].axis('off')

    # Overlay
    overlay = original_image.copy().astype(np.float32)
    mask_colored = np.zeros_like(overlay)
    mask_colored[:, :, 0] = pred_binary  # Red channel
    combined = (overlay * 0.7 + mask_colored * 0.3)
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    axes[1, 1].imshow(combined)
    axes[1, 1].set_title('Overlay (Red = Mask)', fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle(f'Segmentation Results: {title}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def find_sample_image():
    """Tìm một ảnh mẫu để test"""
    # Các vị trí có thể có ảnh test
    possible_dirs = [
        os.path.join(project_root, 'data', 'processed', 'test', 'images'),
        os.path.join(project_root, 'data', 'processed', 'val', 'images'),
        os.path.join(project_root, 'data', 'raw', 'images'),
        os.path.join(project_root, 'tests', 'test_images')
    ]

    for test_dir in possible_dirs:
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                sample_path = os.path.join(test_dir, files[0])
                print(f"📷 Found sample image: {sample_path}")
                return sample_path

    return None


def create_sample_test_image():
    """Tạo một ảnh test mẫu"""
    print("🎨 Creating sample test image...")

    # Create output directory
    test_img_dir = os.path.join(project_root, 'tests', 'test_images')
    os.makedirs(test_img_dir, exist_ok=True)

    # Create a simple test image
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)

    # Add gradient background
    grad = np.linspace(0.3, 1.0, 256)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * grad).astype(np.uint8)

    # Add some distinctive shapes
    cv2.circle(img, (128, 128), 50, (255, 255, 255), -1)  # White circle
    cv2.rectangle(img, (50, 50), (100, 100), (255, 100, 100), -1)  # Light red rectangle
    cv2.ellipse(img, (200, 80), (40, 25), 45, 0, 360, (100, 255, 100), -1)  # Light green ellipse

    # Save
    sample_path = os.path.join(test_img_dir, 'sample_test_image.jpg')
    cv2.imwrite(sample_path, img)

    print(f"✅ Sample image created: {sample_path}")
    return sample_path


def interactive_test():
    """Interactive testing mode"""
    print("🎮 Interactive Testing Mode")
    print("=" * 50)

    while True:
        print("\\n📋 Options:")
        print("1. Test with sample image")
        print("2. Test with custom image path")
        print("3. Create and test sample image")
        print("4. Exit")

        choice = input("\\nChoose option (1-4): ").strip()

        if choice == '1':
            sample_path = find_sample_image()
            if sample_path:
                threshold = float(input("Enter threshold (0.0-1.0, default 0.5): ") or "0.5")
                predict_single_image(sample_path, threshold=threshold)
            else:
                print("❌ No sample images found!")

        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                threshold = float(input("Enter threshold (0.0-1.0, default 0.5): ") or "0.5")
                predict_single_image(image_path, threshold=threshold)
            else:
                print(f"❌ File not found: {image_path}")

        elif choice == '3':
            sample_path = create_sample_test_image()
            threshold = float(input("Enter threshold (0.0-1.0, default 0.5): ") or "0.5")
            predict_single_image(sample_path, threshold=threshold)

        elif choice == '4':
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid option!")


def main():
    parser = argparse.ArgumentParser(description='Test U-Net model prediction')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, help='Path to model (default: scripts/models/best_model.pth)')
    parser.add_argument('--output', type=str, help='Output directory (default: tests/results)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    print("🔮 PyTorch U-Net Prediction Test")
    print("=" * 50)
    print(f"📁 Project root: {project_root}")
    print(f"📁 Source path: {src_path}")

    if args.interactive:
        interactive_test()
    elif args.image:
        if os.path.exists(args.image):
            predict_single_image(args.image, args.model, args.output, args.threshold)
        else:
            print(f"❌ Image not found: {args.image}")
    else:
        # Default: try to find sample image
        print("🔍 Looking for sample images...")
        sample_path = find_sample_image()

        if sample_path:
            predict_single_image(sample_path, args.model, args.output, args.threshold)
        else:
            print("🎨 No sample images found, creating test image...")
            sample_path = create_sample_test_image()
            predict_single_image(sample_path, args.model, args.output, args.threshold)

        print("\\n💡 For more options, run with --interactive flag")


if __name__ == "__main__":
    main()

# ==============================================================================
# create_test_structure.py - Script để tạo cấu trúc test hoàn chỉnh
# ==============================================================================
