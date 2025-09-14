import yaml
import os


def create_clean_config():
    """Create a clean config.yaml file"""
    config = {
        'data': {
            'raw_data_dir': 'data/raw',
            'processed_data_dir': 'data/processed',
            'image_dir': 'images',
            'mask_dir': 'masks'
        },
        'model': {
            'input_size': [256, 256, 3],
            'architecture': 'unet'
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
            'fill_mode': 'nearest'
        },
        'callbacks': {
            'early_stopping': {
                'patience': 20,
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

    # Remove old config if exists
    if os.path.exists('config.yaml'):
        os.remove('config.yaml')

    # Create new config with UTF-8 encoding
    with open('config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2, allow_unicode=True)

    print("✅ Clean config.yaml created successfully!")
    print("📄 Config contents:")
    print("-" * 30)
    with open('config.yaml', 'r', encoding='utf-8') as file:
        print(file.read())


if __name__ == "__main__":
    create_clean_config()
