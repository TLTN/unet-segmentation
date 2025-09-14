import unittest
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

import torch
import numpy as np
from model.unet import UNet, DoubleConv
from training.trainer import calculate_metrics, DiceLoss, IoULoss


class TestUNet(unittest.TestCase):

    def setUp(self):
        self.model = UNet(in_channels=3, out_channels=1)
        self.device = torch.device('cpu')

    def test_model_creation(self):
        """Test if model can be created"""
        self.assertIsInstance(self.model, UNet)

        # Check if model has the right number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0)

    def test_forward_pass(self):
        """Test forward pass with dummy input"""
        batch_size = 2
        channels = 3
        height, width = 256, 256

        dummy_input = torch.randn(batch_size, channels, height, width)

        with torch.no_grad():
            output = self.model(dummy_input)

        # Check output shape
        expected_shape = (batch_size, 1, height, width)
        self.assertEqual(output.shape, expected_shape)

        # Check output values are in [0, 1] due to sigmoid
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_double_conv_block(self):
        """Test DoubleConv block"""
        block = DoubleConv(3, 64)
        dummy_input = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            output = block(dummy_input)

        expected_shape = (1, 64, 64, 64)
        self.assertEqual(output.shape, expected_shape)


class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()

    def test_dice_loss(self):
        """Test Dice loss calculation"""
        # Perfect prediction
        pred = torch.ones(1, 1, 10, 10)
        target = torch.ones(1, 1, 10, 10)

        loss = self.dice_loss(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

        # Worst prediction
        pred = torch.zeros(1, 1, 10, 10)
        target = torch.ones(1, 1, 10, 10)

        loss = self.dice_loss(pred, target)
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_iou_loss(self):
        """Test IoU loss calculation"""
        # Perfect prediction
        pred = torch.ones(1, 1, 10, 10)
        target = torch.ones(1, 1, 10, 10)

        loss = self.iou_loss(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class TestMetrics(unittest.TestCase):

    def test_calculate_metrics(self):
        """Test metrics calculation"""
        # Perfect prediction
        pred = torch.ones(1, 1, 10, 10)
        target = torch.ones(1, 1, 10, 10)

        metrics = calculate_metrics(pred, target)
        self.assertAlmostEqual(metrics['dice'], 1.0, places=5)
        self.assertAlmostEqual(metrics['iou'], 1.0, places=5)
        self.assertAlmostEqual(metrics['accuracy'], 1.0, places=5)

        # No overlap
        pred = torch.zeros(1, 1, 10, 10)
        target = torch.ones(1, 1, 10, 10)

        metrics = calculate_metrics(pred, target)
        self.assertAlmostEqual(metrics['dice'], 0.0, places=5)
        self.assertAlmostEqual(metrics['iou'], 0.0, places=5)


if __name__ == '__main__':
    unittest.main()