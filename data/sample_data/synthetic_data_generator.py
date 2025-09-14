import yaml
import os


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file with proper encoding"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except UnicodeDecodeError:
        with open(config_path, 'r', encoding='cp1252') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        raise


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
