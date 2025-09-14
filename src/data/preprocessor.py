import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


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
