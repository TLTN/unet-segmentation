# ==============================================================================
# scripts/data_preparation.py (FIXED VERSION)
# ==============================================================================
import os
import sys
import traceback
import yaml

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)


def create_default_config():
    """Create default configuration if config file doesn't exist"""
    default_config = {
        'data': {
            'raw_data_dir': "data/raw",
            'processed_data_dir': "data/processed",
            'image_dir': "images",
            'mask_dir': "masks"
        },
        'model': {
            'input_size': [256, 256, 3],
            'architecture': "unet"
        },
        'training': {
            'batch_size': 8,
            'epochs': 100,
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
            'fill_mode': "nearest"
        },
        'callbacks': {
            'early_stopping': {
                'patience': 20,
                'monitor': "val_dice",
                'mode': "max"
            },
            'reduce_lr': {
                'factor': 0.2,
                'patience': 10,
                'min_lr': 0.0000001
            }
        },
        'paths': {
            'model_dir': "models",
            'results_dir': "results",
            'logs_dir': "logs"
        }
    }

    # Save default config
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(default_config, file, default_flow_style=False, indent=2)

    print("✅ Default config.yaml created successfully!")
    return default_config


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return create_default_config()


def create_directories(config):
    """Create necessary directories"""
    dirs = [
        config['data']['processed_data_dir'],
        config['paths']['model_dir'],
        config['paths']['results_dir'],
        config['paths']['logs_dir'],
        os.path.join(config['data']['processed_data_dir'], 'train', 'images'),
        os.path.join(config['data']['processed_data_dir'], 'train', 'masks'),
        os.path.join(config['data']['processed_data_dir'], 'val', 'images'),
        os.path.join(config['data']['processed_data_dir'], 'val', 'masks'),
        os.path.join(config['data']['processed_data_dir'], 'test', 'images'),
        os.path.join(config['data']['processed_data_dir'], 'test', 'masks'),
        os.path.join(config['paths']['model_dir'], 'checkpoints'),
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def main():
    print("🔄 Starting data preparation...")

    try:
        # Load configuration
        config_path = os.path.join(project_root, "config.yaml")
        if not os.path.exists(config_path):
            print("❌ Config file not found!")
            print("📋 Creating default config...")
            config = create_default_config()
        else:
            print("📋 Loading configuration...")
            config = load_config(config_path)
            print("✅ Configuration loaded successfully")

        # Create directories first
        print("📁 Creating directories...")
        create_directories(config)

        # Check if raw data exists
        raw_img_dir = os.path.join(config['data']['raw_data_dir'], config['data']['image_dir'])
        raw_mask_dir = os.path.join(config['data']['raw_data_dir'], config['data']['mask_dir'])

        print(f"🔍 Checking raw data directories:")
        print(f"   📁 Images: {raw_img_dir}")
        print(f"   📁 Masks: {raw_mask_dir}")

        if not os.path.exists(raw_img_dir) or not os.path.exists(raw_mask_dir):
            print("❌ Raw data directories not found!")
            print("\n🎨 To generate synthetic data, run:")
            print("   python data/sample_data/synthetic_data_generator.py")
            print("\n📋 Or place your data in:")
            print(f"   📁 Images: {raw_img_dir}")
            print(f"   📁 Masks: {raw_mask_dir}")
            return 1

        # Check if directories contain files
        img_files = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        mask_files = [f for f in os.listdir(raw_mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(img_files) == 0 or len(mask_files) == 0:
            print(f"❌ No image files found!")
            print(f"   📁 Images found: {len(img_files)}")
            print(f"   📁 Masks found: {len(mask_files)}")
            return 1

        print(f"✅ Found {len(img_files)} images and {len(mask_files)} masks")

        # Import and use DatasetPreparer
        try:
            from data.preprocessor import DatasetPreparer
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Creating inline DatasetPreparer...")

            # Inline DatasetPreparer if import fails
            import shutil
            import random
            from sklearn.model_selection import train_test_split
            from tqdm import tqdm

            class DatasetPreparer:
                def __init__(self, config):
                    self.config = config
                    self.raw_data_dir = config['data']['raw_data_dir']
                    self.processed_data_dir = config['data']['processed_data_dir']
                    self.image_dir = config['data']['image_dir']
                    self.mask_dir = config['data']['mask_dir']

                def get_image_mask_pairs(self):
                    """Get matching image-mask pairs"""
                    image_path = os.path.join(self.raw_data_dir, self.image_dir)
                    mask_path = os.path.join(self.raw_data_dir, self.mask_dir)

                    image_files = sorted([f for f in os.listdir(image_path)
                                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    mask_files = sorted([f for f in os.listdir(mask_path)
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

                    # Match files by name
                    image_names = [os.path.splitext(f)[0] for f in image_files]
                    mask_names = [os.path.splitext(f)[0] for f in mask_files]

                    pairs = []
                    for img_name, img_file in zip(image_names, image_files):
                        if img_name in mask_names:
                            mask_idx = mask_names.index(img_name)
                            pairs.append((img_file, mask_files[mask_idx]))

                    return pairs

                def split_data(self, pairs):
                    """Split data into train/val/test"""
                    random.shuffle(pairs)

                    val_split = self.config['training']['validation_split']
                    test_split = self.config['training']['test_split']

                    # First split: train+val vs test
                    train_val, test = train_test_split(pairs, test_size=test_split, random_state=42)

                    # Second split: train vs val
                    val_size = val_split / (1 - test_split)
                    train, val = train_test_split(train_val, test_size=val_size, random_state=42)

                    return train, val, test

                def copy_files(self, pairs, split_name):
                    """Copy files to processed directory"""
                    src_img_dir = os.path.join(self.raw_data_dir, self.image_dir)
                    src_mask_dir = os.path.join(self.raw_data_dir, self.mask_dir)

                    dst_img_dir = os.path.join(self.processed_data_dir, split_name, 'images')
                    dst_mask_dir = os.path.join(self.processed_data_dir, split_name, 'masks')

                    os.makedirs(dst_img_dir, exist_ok=True)
                    os.makedirs(dst_mask_dir, exist_ok=True)

                    for img_file, mask_file in tqdm(pairs, desc=f"Copying {split_name} data"):
                        src_img = os.path.join(src_img_dir, img_file)
                        dst_img = os.path.join(dst_img_dir, img_file)
                        shutil.copy2(src_img, dst_img)

                        src_mask = os.path.join(src_mask_dir, mask_file)
                        dst_mask = os.path.join(dst_mask_dir, mask_file)
                        shutil.copy2(src_mask, dst_mask)

                def prepare_dataset(self):
                    """Main method to prepare dataset"""
                    print("Preparing dataset...")

                    pairs = self.get_image_mask_pairs()
                    print(f"Found {len(pairs)} image-mask pairs")

                    train_pairs, val_pairs, test_pairs = self.split_data(pairs)
                    print(f"Split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

                    self.copy_files(train_pairs, 'train')
                    self.copy_files(val_pairs, 'val')
                    self.copy_files(test_pairs, 'test')

                    print("Dataset preparation complete!")

        # Prepare dataset
        print("🔄 Preparing dataset...")
        preparer = DatasetPreparer(config)
        preparer.prepare_dataset()

        print("✅ Data preparation completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you're in the project root directory")
        print("   2. Check if config.yaml exists and is valid")
        print("   3. Run synthetic data generator first")
        print("   4. Verify file permissions")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
