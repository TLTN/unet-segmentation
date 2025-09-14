from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


class DataAugmentation:
    def __init__(self, augmentation_config):
        self.config = augmentation_config

    def create_generators(self, X_train, y_train, X_val, y_val, batch_size=8):
        """Create augmented data generators"""

        # Training data generator with augmentation
        image_datagen = ImageDataGenerator(
            rotation_range=self.config.get('rotation_range', 0.2),
            width_shift_range=self.config.get('width_shift_range', 0.05),
            height_shift_range=self.config.get('height_shift_range', 0.05),
            shear_range=self.config.get('shear_range', 0.05),
            zoom_range=self.config.get('zoom_range', 0.05),
            horizontal_flip=self.config.get('horizontal_flip', True),
            vertical_flip=self.config.get('vertical_flip', False),
            fill_mode=self.config.get('fill_mode', 'nearest'),
            preprocessing_function=self._augment_brightness_contrast
        )

        mask_datagen = ImageDataGenerator(
            rotation_range=self.config.get('rotation_range', 0.2),
            width_shift_range=self.config.get('width_shift_range', 0.05),
            height_shift_range=self.config.get('height_shift_range', 0.05),
            shear_range=self.config.get('shear_range', 0.05),
            zoom_range=self.config.get('zoom_range', 0.05),
            horizontal_flip=self.config.get('horizontal_flip', True),
            vertical_flip=self.config.get('vertical_flip', False),
            fill_mode=self.config.get('fill_mode', 'nearest')
        )

        # Use same seed for synchronized augmentation
        seed = 1
        image_datagen.fit(X_train, augment=True, seed=seed)
        mask_datagen.fit(y_train, augment=True, seed=seed)

        # Create generators
        train_image_generator = image_datagen.flow(
            X_train, batch_size=batch_size, seed=seed, shuffle=True)
        train_mask_generator = mask_datagen.flow(
            y_train, batch_size=batch_size, seed=seed, shuffle=True)

        # Validation generators (no augmentation)
        val_image_datagen = ImageDataGenerator()
        val_mask_datagen = ImageDataGenerator()

        val_image_generator = val_image_datagen.flow(
            X_val, batch_size=batch_size, shuffle=False)
        val_mask_generator = val_mask_datagen.flow(
            y_val, batch_size=batch_size, shuffle=False)

        # Combine generators
        train_generator = self._combine_generators(train_image_generator, train_mask_generator)
        val_generator = self._combine_generators(val_image_generator, val_mask_generator)

        return train_generator, val_generator

    def _combine_generators(self, img_gen, mask_gen):
        """Combine image and mask generators"""
        while True:
            img_batch = next(img_gen)
            mask_batch = next(mask_gen)

            # Ensure masks are binary
            mask_batch = (mask_batch > 0.5).astype(np.float32)

            yield img_batch, mask_batch

    def _augment_brightness_contrast(self, image):
        """Apply brightness and contrast augmentation"""
        # Random brightness
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = image * brightness_factor

        # Random contrast
        contrast_factor = np.random.uniform(0.8, 1.2)
        image = (image - 0.5) * contrast_factor + 0.5

        # Clip values
        image = np.clip(image, 0.0, 1.0)

        return image
