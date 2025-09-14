# ==============================================================================
# download_kvasir_dataset.py - Download và xử lý Kvasir-SEG dataset
# ==============================================================================
import os
import requests
import zipfile
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


def download_kvasir_dataset():
    """Download Kvasir-SEG polyp segmentation dataset"""
    print("Downloading Kvasir-SEG dataset...")

    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    download_dir = "data/raw/"
    zip_path = os.path.join(download_dir, "kvasir-seg.zip")

    # Create download directory
    os.makedirs(download_dir, exist_ok=True)

    # Check if already downloaded
    if os.path.exists(zip_path):
        print(f"Dataset already downloaded: {zip_path}")
        return zip_path

    # Download with progress bar
    print(f"Downloading from: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            pbar.update(size)

    print(f"Download completed: {zip_path}")
    return zip_path


def extract_dataset(zip_path):
    """Extract the downloaded dataset"""
    print("Extracting dataset...")

    extract_dir = "data/raw/"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Extraction completed!")

    # Find extracted directory
    kvasir_dir = None
    for item in os.listdir(extract_dir):
        item_path = os.path.join(extract_dir, item)
        if os.path.isdir(item_path) and 'kvasir' in item.lower():
            kvasir_dir = item_path
            break

    if kvasir_dir is None:
        print("Warning: Could not find Kvasir directory")
        kvasir_dir = os.path.join(extract_dir, "Kvasir-SEG")

    print(f"Dataset extracted to: {kvasir_dir}")
    return kvasir_dir


def organize_kvasir_data(kvasir_dir):
    """Organize Kvasir data into images and masks"""
    print("Organizing Kvasir dataset...")

    # Expected structure: Kvasir-SEG/images/ and Kvasir-SEG/masks/
    images_dir = os.path.join(kvasir_dir, "images")
    masks_dir = os.path.join(kvasir_dir, "masks")

    # Create organized structure
    organized_images = "data/raw/images"
    organized_masks = "data/raw/masks"

    os.makedirs(organized_images, exist_ok=True)
    os.makedirs(organized_masks, exist_ok=True)

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Error: Expected structure not found in {kvasir_dir}")
        print("Looking for alternative structure...")

        # Try to find images and masks in subdirectories
        for root, dirs, files in os.walk(kvasir_dir):
            if 'images' in dirs and 'masks' in dirs:
                images_dir = os.path.join(root, 'images')
                masks_dir = os.path.join(root, 'masks')
                print(f"Found images at: {images_dir}")
                print(f"Found masks at: {masks_dir}")
                break

    # Copy and rename files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(image_files)} images")

    for i, img_file in enumerate(tqdm(image_files, desc="Organizing files")):
        # Copy image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(organized_images, f"kvasir_{i:04d}.jpg")

        # Convert and resize image
        img = Image.open(src_img).convert('RGB')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img.save(dst_img, 'JPEG', quality=95)

        # Copy corresponding mask
        mask_file = os.path.splitext(img_file)[0] + '.jpg'  # Kvasir masks are usually .jpg
        src_mask = os.path.join(masks_dir, mask_file)

        # Try different mask extensions
        if not os.path.exists(src_mask):
            for ext in ['.png', '.jpeg']:
                alt_mask = os.path.join(masks_dir, os.path.splitext(img_file)[0] + ext)
                if os.path.exists(alt_mask):
                    src_mask = alt_mask
                    break

        if os.path.exists(src_mask):
            dst_mask = os.path.join(organized_masks, f"kvasir_{i:04d}.png")

            # Convert mask to binary
            mask = Image.open(src_mask).convert('L')
            mask = mask.resize((256, 256), Image.Resampling.NEAREST)

            # Ensure binary mask (0 or 255)
            mask_array = np.array(mask)
            mask_array = (mask_array > 128).astype(np.uint8) * 255
            mask_binary = Image.fromarray(mask_array)
            mask_binary.save(dst_mask, 'PNG')

    print(f"Dataset organized:")
    print(f"  Images: {len(os.listdir(organized_images))}")
    print(f"  Masks: {len(os.listdir(organized_masks))}")

    return organized_images, organized_masks


def prepare_kvasir_splits():
    """Split Kvasir dataset into train/val/test"""
    print("Preparing train/val/test splits...")

    # Load config for split ratios
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        val_split = config['training']['validation_split']
        test_split = config['training']['test_split']
    except:
        val_split = 0.2
        test_split = 0.1

    images_dir = "data/raw/images"
    masks_dir = "data/raw/masks"

    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Create pairs
    pairs = []
    for img_file in image_files:
        mask_file = os.path.splitext(img_file)[0] + '.png'
        if os.path.exists(os.path.join(masks_dir, mask_file)):
            pairs.append((img_file, mask_file))

    print(f"Found {len(pairs)} valid image-mask pairs")

    # Split data
    train_val, test = train_test_split(pairs, test_size=test_split, random_state=42)
    val_size = val_split / (1 - test_split)
    train, val = train_test_split(train_val, test_size=val_size, random_state=42)

    print(f"Dataset splits:")
    print(f"  Train: {len(train)} samples")
    print(f"  Validation: {len(val)} samples")
    print(f"  Test: {len(test)} samples")

    # Create processed directories
    splits = [('train', train), ('val', val), ('test', test)]

    for split_name, split_pairs in splits:
        split_img_dir = f"data/processed/{split_name}/images"
        split_mask_dir = f"data/processed/{split_name}/masks"

        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)

        for img_file, mask_file in tqdm(split_pairs, desc=f"Creating {split_name} split"):
            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(split_img_dir, img_file)
            shutil.copy2(src_img, dst_img)

            # Copy mask
            src_mask = os.path.join(masks_dir, mask_file)
            dst_mask = os.path.join(split_mask_dir, mask_file)
            shutil.copy2(src_mask, dst_mask)

    print("Dataset preparation completed!")


def analyze_dataset():
    """Analyze the prepared dataset"""
    print("Analyzing dataset...")

    splits = ['train', 'val', 'test']

    for split in splits:
        img_dir = f"data/processed/{split}/images"
        mask_dir = f"data/processed/{split}/masks"

        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            img_count = len([f for f in os.listdir(img_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            mask_count = len([f for f in os.listdir(mask_dir)
                              if f.lower().endswith(('.png',))])

            print(f"{split.capitalize()}: {img_count} images, {mask_count} masks")

    # Sample mask coverage analysis
    sample_masks_dir = "data/processed/train/masks"
    if os.path.exists(sample_masks_dir):
        mask_files = os.listdir(sample_masks_dir)[:10]  # Sample 10 masks
        coverages = []

        for mask_file in mask_files:
            mask_path = os.path.join(sample_masks_dir, mask_file)
            mask = np.array(Image.open(mask_path))
            coverage = np.sum(mask > 128) / mask.size
            coverages.append(coverage)

        if coverages:
            print(f"Average polyp coverage: {np.mean(coverages):.1%}")
            print(f"Coverage range: {np.min(coverages):.1%} - {np.max(coverages):.1%}")


def create_kvasir_config():
    """Create/update config for Kvasir dataset"""
    config = {
        'data': {
            'raw_data_dir': 'data/raw',
            'processed_data_dir': 'data/processed',
            'image_dir': 'images',
            'mask_dir': 'masks',
            'dataset_name': 'Kvasir-SEG',
            'dataset_description': 'Polyp segmentation dataset'
        },
        'model': {
            'input_size': [256, 256, 3],
            'architecture': 'unet',
            'task': 'polyp_segmentation'
        },
        'training': {
            'batch_size': 8,
            'epochs': 50,
            'learning_rate': 0.001,
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
                'patience': 15,
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

    with open('config_kvasir.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print("Created config_kvasir.yaml for Kvasir dataset")


def main():
    print("Kvasir-SEG Dataset Setup")
    print("=" * 50)
    print("This dataset contains colonoscopy images with polyp segmentation masks")
    print("Perfect for medical image segmentation research!")

    try:
        # Step 1: Download
        zip_path = download_kvasir_dataset()

        # Step 2: Extract
        kvasir_dir = extract_dataset(zip_path)

        # Step 3: Organize
        organize_kvasir_data(kvasir_dir)

        # Step 4: Create splits
        prepare_kvasir_splits()

        # Step 5: Analyze
        analyze_dataset()

        # Step 6: Create config
        create_kvasir_config()

        print("\n" + "=" * 50)
        print("Kvasir-SEG dataset setup completed!")
        print("=" * 50)
        print("\nDataset ready for training:")
        print("  python train.py --config config_kvasir.yaml")
        print("\nOr test predictions:")
        print("  python tests/test_predict.py")
        print("\nDataset info:")
        print("  - Medical polyp segmentation")
        print("  - Real colonoscopy images")
        print("  - High-quality annotations")
        print("  - Perfect for medical AI research")

    except Exception as e:
        print(f"Error setting up dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()