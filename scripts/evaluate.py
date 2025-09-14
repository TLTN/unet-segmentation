import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

try:
    import torch
    from torch.utils.data import DataLoader

    from utils.config import load_config
    from model.unet import UNet
    from data.dataset import SegmentationDataset, get_val_transform
    from training.trainer import calculate_metrics
    from utils.visualization import predict_and_visualize

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Calculate metrics for each sample in batch
            for i in range(images.shape[0]):
                metrics = calculate_metrics(outputs[i:i + 1], masks[i:i + 1])
                all_metrics.append(metrics)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate PyTorch U-Net model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'],
                        default='test')
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
    print("Model loaded successfully")

    # Create dataset
    input_size = tuple(config['model']['input_size'][:2])

    dataset = SegmentationDataset(
        image_dir=os.path.join(config['data']['processed_data_dir'], args.split, 'images'),
        mask_dir=os.path.join(config['data']['processed_data_dir'], args.split, 'masks'),
        transform=get_val_transform(input_size)
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    print(f"Evaluating on {len(dataset)} {args.split} samples")

    # Evaluate
    print("Running evaluation...")
    metrics = evaluate_model(model, dataloader, device)

    # Convert to DataFrame
    df = pd.DataFrame(metrics)

    # Print results
    print(f"\nEvaluation Results on {args.split} set:")
    print("=" * 50)
    print(df.describe())

    # Plot metrics distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_names = ['dice', 'iou', 'accuracy']
    for i, metric in enumerate(metrics_names):
        values = df[metric]

        axes[i].hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].axvline(values.mean(), color='red', linestyle='--',
                        label=f'Mean: {values.mean():.3f}')
        axes[i].set_title(f'{metric.capitalize()} Distribution')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save results
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    plt.savefig(os.path.join(results_dir, f'evaluation_{args.split}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Save metrics to CSV
    df.to_csv(os.path.join(results_dir, f'evaluation_{args.split}.csv'), index=False)

    # Visualize predictions
    print("Generating prediction visualizations...")
    predict_and_visualize(
        model, dataloader, device,
        num_samples=5,
        save_path=os.path.join(results_dir, f'predictions_{args.split}.png')
    )

    print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
