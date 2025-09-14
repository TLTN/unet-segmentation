import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️ Albumentations not available, using basic transforms")


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

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
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Normalize mask to [0, 1]
        mask = mask / 255.0

        if self.transform is not None:
            if ALBUMENTATIONS_AVAILABLE:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            else:
                # Basic transform without albumentations
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask).float()

        if len(mask.shape) == 2:  # Add channel dimension if needed
            mask = mask.unsqueeze(0)

        return image, mask


def get_train_transform(image_size=(256, 256)):
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Resize(*image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(10, 15, 10),
                A.CLAHE(clip_limit=3),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        # Basic transform without albumentations
        return None


def get_val_transform(image_size=(256, 256)):
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Resize(*image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return None