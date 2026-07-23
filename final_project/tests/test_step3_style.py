"""
Step 3 Style Checks — architecture recommendations (advisory, non-blocking).

These tests check whether your model follows the recommended architecture
from the hints. They will NOT block your CI from passing — they are
informational only. If any fail, you'll see a warning but your Step 3
job still shows green.

Recommended architecture:
    - self.features as nn.Sequential with 3 conv blocks
    - self.pool as nn.AdaptiveAvgPool2d
    - self.classifier as nn.Sequential
    - BatchNorm2d after each Conv2d
"""
import torch

from steel_defect.utils import NUM_CLASSES
from steel_defect.model import SteelCNN

import pytest


@pytest.fixture
def model():
    return SteelCNN(num_classes=NUM_CLASSES)


class TestModelStyle:
    """Advisory checks — recommended but not required."""

    def test_has_features_attribute(self, model):
        assert hasattr(model, "features"), (
            "Hint: the recommended architecture uses self.features "
            "as an nn.Sequential for the convolutional blocks"
        )

    def test_has_pool_attribute(self, model):
        assert hasattr(model, "pool"), (
            "Hint: the recommended architecture uses self.pool "
            "as nn.AdaptiveAvgPool2d(1)"
        )

    def test_has_classifier_attribute(self, model):
        assert hasattr(model, "classifier"), (
            "Hint: the recommended architecture uses self.classifier "
            "as an nn.Sequential for the fully connected layers"
        )

    def test_uses_batchnorm(self, model):
        bn_layers = [m for m in model.modules()
                     if isinstance(m, torch.nn.BatchNorm2d)]
        assert len(bn_layers) >= 1, (
            "Hint: BatchNorm2d after each Conv2d helps training stability"
        )

    def test_uses_three_conv_blocks(self, model):
        conv_layers = [m for m in model.modules()
                       if isinstance(m, torch.nn.Conv2d)]
        assert len(conv_layers) >= 3, (
            f"Hint: the recommended architecture uses 3 conv blocks, "
            f"found {len(conv_layers)}"
        )

    def test_uses_dropout(self, model):
        dropout_layers = [m for m in model.modules()
                          if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d))]
        assert len(dropout_layers) >= 1, (
            "Hint: Dropout in the classifier helps prevent overfitting"
        )
