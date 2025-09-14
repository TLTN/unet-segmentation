import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split


class UNetModel:
    def __init__(self, input_size=(256, 256, 3)):
        self.input_size = input_size
        self.model = None

    def build_unet(self):
        """Build U-Net architecture"""
        inputs = Input(self.input_size)

        # Encoder
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bottom
        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Decoder
        up6 = Conv2D(512, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)

        # Output layer
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)
        return self.model

    def compile_model(self, learning_rate=1e-4):
        """Compile the model"""
        if self.model is None:
            self.build_unet()

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', self.dice_coefficient, self.iou_score]
        )

    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        """Dice coefficient metric"""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    @staticmethod
    def iou_score(y_true, y_pred, smooth=1e-6):
        """IoU (Intersection over Union) metric"""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)


class DataPreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def preprocess_image(self, img_path):
        """Load and preprocess image"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        return img

    def preprocess_mask(self, mask_path):
        """Load and preprocess masks"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load masks: {mask_path}")

        mask = cv2.resize(mask, self.target_size)
        mask = mask.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Convert to binary masks
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        return mask

    def load_data(self, image_dir, mask_dir):
        """Load all images and masks from directories"""
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if len(image_files) != len(mask_files):
            raise ValueError("Number of images and masks must be equal")

        images = []
        masks = []

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            img = self.preprocess_image(img_path)
            mask = self.preprocess_mask(mask_path)

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)


class DataAugmentation:
    def __init__(self):
        self.data_gen_args = dict(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def create_generators(self, X_train, y_train, X_val, y_val, batch_size=8):
        """Create data generators for training and validation"""
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        mask_datagen = ImageDataGenerator(**self.data_gen_args)

        # Fit generators on training data
        seed = 1
        image_datagen.fit(X_train, augment=True, seed=seed)
        mask_datagen.fit(y_train, augment=True, seed=seed)

        # Create training generator
        image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
        mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)
        train_generator = zip(image_generator, mask_generator)

        # Create validation generator (no augmentation)
        val_image_datagen = ImageDataGenerator()
        val_mask_datagen = ImageDataGenerator()
        val_image_generator = val_image_datagen.flow(X_val, batch_size=batch_size, seed=seed)
        val_mask_generator = val_mask_datagen.flow(y_val, batch_size=batch_size, seed=seed)
        val_generator = zip(val_image_generator, val_mask_generator)

        return train_generator, val_generator


class UNetTrainer:
    def __init__(self, model, save_dir='./models'):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.save_dir, 'best_model.h5'),
                monitor='val_dice_coefficient',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_dice_coefficient',
                mode='max',
                patience=20,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks

    def train(self, train_generator, val_generator, steps_per_epoch,
              validation_steps, epochs=100):
        """Train the model"""
        callbacks = self.create_callbacks()

        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()

        # Dice Coefficient
        axes[1, 0].plot(history.history['dice_coefficient'], label='Training Dice')
        axes[1, 0].plot(history.history['val_dice_coefficient'], label='Validation Dice')
        axes[1, 0].set_title('Dice Coefficient')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Coefficient')
        axes[1, 0].legend()

        # IoU Score
        axes[1, 1].plot(history.history['iou_score'], label='Training IoU')
        axes[1, 1].plot(history.history['val_iou_score'], label='Validation IoU')
        axes[1, 1].set_title('IoU Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU Score')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()


def predict_and_visualize(model, test_images, test_masks=None, num_samples=5):
    """Predict and visualize results"""
    predictions = model.predict(test_images)

    fig, axes = plt.subplots(num_samples, 4 if test_masks is not None else 3,
                             figsize=(15, 4 * num_samples))

    for i in range(min(num_samples, len(test_images))):
        # Original image
        axes[i, 0].imshow(test_images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Prediction
        pred_binary = (predictions[i, :, :, 0] > 0.5).astype(np.uint8)
        axes[i, 1].imshow(pred_binary, cmap='gray')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')

        # Prediction probability
        axes[i, 2].imshow(predictions[i, :, :, 0], cmap='viridis')
        axes[i, 2].set_title('Prediction Probability')
        axes[i, 2].axis('off')

        # Ground truth (if available)
        if test_masks is not None:
            axes[i, 3].imshow(test_masks[i, :, :, 0], cmap='gray')
            axes[i, 3].set_title('Ground Truth')
            axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize components
    unet = UNetModel(input_size=(256, 256, 3))
    preprocessor = DataPreprocessor(target_size=(256, 256))
    augmentation = DataAugmentation()

    # Build and compile model
    model = unet.build_unet()
    unet.compile_model(learning_rate=1e-4)

    print("Model Summary:")
    model.summary()

    # Load data (replace with your data paths)
    # X, y = preprocessor.load_data('path/to/images', 'path/to/masks')

    # Split data
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create data generators
    # batch_size = 8
    # train_gen, val_gen = augmentation.create_generators(X_train, y_train, X_val, y_val, batch_size)

    # Train model
    # trainer = UNetTrainer(model)
    # steps_per_epoch = len(X_train) // batch_size
    # validation_steps = len(X_val) // batch_size
    # history = trainer.train(train_gen, val_gen, steps_per_epoch, validation_steps, epochs=100)

    # Plot training history
    # trainer.plot_training_history(history)

    # Make predictions
    # predict_and_visualize(model, X_val, y_val, num_samples=5)

    print("U-Net model setup complete!")