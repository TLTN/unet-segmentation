import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def predict_and_visualize(model, dataloader, device, num_samples=5, save_path=None):
    """Predict and visualize results"""
    model.eval()

    images_list = []
    masks_list = []
    predictions_list = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if len(images_list) >= num_samples:
                break

            images = images.to(device)
            predictions = model(images)

            # Move to CPU and convert to numpy
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            predictions_np = predictions.cpu().numpy()

            for j in range(images.shape[0]):
                if len(images_list) >= num_samples:
                    break

                # Denormalize image for visualization
                img = images_np[j].transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                images_list.append(img)
                masks_list.append(masks_np[j, 0])
                predictions_list.append(predictions_np[j, 0])

    # Plot results
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images_list[i])
        axes[i, 0].set_title('Original Image', fontweight='bold')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(masks_list[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth', fontweight='bold')
        axes[i, 1].axis('off')

        # Prediction probability
        axes[i, 2].imshow(predictions_list[i], cmap='viridis', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction Probability', fontweight='bold')
        axes[i, 2].axis('off')

        # Binary prediction
        pred_binary = (predictions_list[i] > 0.5).astype(np.float32)
        axes[i, 3].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title('Binary Prediction', fontweight='bold')
        axes[i, 3].axis('off')

        # Calculate and display metrics for this sample
        dice = calculate_dice_coefficient(masks_list[i], pred_binary)
        iou = calculate_iou(masks_list[i], pred_binary)

        # Add metrics text
        metrics_text = f'Dice: {dice:.3f}\nIoU: {iou:.3f}'
        axes[i, 3].text(0.02, 0.98, metrics_text, transform=axes[i, 3].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round',
                                                           facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def calculate_dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def calculate_iou(y_true, y_pred, smooth=1e-6):
    """Calculate IoU"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)
