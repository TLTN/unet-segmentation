import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-6
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

def dice_coefficient(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_dice = 0

        pbar = tqdm(train_loader, desc='Training')
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            dice = dice_coefficient(outputs, masks)

            total_loss += loss.item()
            total_dice += dice.item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}'
            })

        return total_loss / len(train_loader), total_dice / len(train_loader)

    def val_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_dice = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice = dice_coefficient(outputs, masks)

                total_loss += loss.item()
                total_dice += dice.item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.item():.4f}'
                })

        return total_loss / len(val_loader), total_dice / len(val_loader)

    def train(self, train_loader, val_loader, epochs):
        best_dice = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')

            train_loss, train_dice = self.train_epoch(train_loader)
            val_loss, val_dice = self.val_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dices.append(train_dice)
            self.val_dices.append(val_dice)

            print(f'Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}')

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(self.model.state_dict(), 'models/best_model.pth')
                print(f'New best model saved! Dice: {best_dice:.4f}')

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices
        }

    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['val_losses'], label='Val Loss')
        ax1.set_title('Loss')
        ax1.legend()

        ax2.plot(history['train_dices'], label='Train Dice')
        ax2.plot(history['val_dices'], label='Val Dice')
        ax2.set_title('Dice Score')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('results/training_history.png')
        plt.show()
