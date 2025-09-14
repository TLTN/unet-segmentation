import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from utils.config import load_config
    from model.unet import UNet

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, self.image_paths[idx]


def get_inference_transform(image_size=(256, 256)):
    return A.Compose([
        A.Resize(*image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained U-Net model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='results/predictions',
                        help='Output directory for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    in_channels = config['model']['input_size'][2]
    model = UNet(in_channels=in_channels, out_channels=1)

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get image paths
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    else:
        print(f"Input path not found: {args.input}")
        return 1

    if len(image_paths) == 0:
        print("No valid image files found")
        return 1

    print(f"Found {len(image_paths)} images to process")

    # Create dataset and dataloader
    input_size = tuple(config['model']['input_size'][:2])
    transform = get_inference_transform(input_size)

    dataset = InferenceDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Process images
    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)

            # Predict
            outputs = model(images)

            for i, path in enumerate(paths):
                prediction = outputs[i, 0].cpu().numpy()

                # Create binary mask
                binary_mask = (prediction > args.threshold).astype(np.uint8) * 255

                # Save results
                base_name = os.path.splitext(os.path.basename(path))[0]

                # Save probability map
                prob_path = os.path.join(args.output, f"{base_name}_probability.png")
                prob_map = (prediction * 255).astype(np.uint8)
                cv2.imwrite(prob_path, prob_map)

                # Save binary mask
                mask_path = os.path.join(args.output, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, binary_mask)

                print(f"Processed: {os.path.basename(path)}")

    print(f"Predictions saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)