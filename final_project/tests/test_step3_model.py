"""
Step 3 Tests — MODEL-1 and MODEL-2.

Run with:
    pytest tests/test_step3_model.py -v

Prerequisites: None.

These tests verify the *contract* of your model (correct input/output shapes,
trainable parameters, raw logits) without enforcing a specific architecture.
The hints in model.py guide you toward a 3-block CNN, but you are free to
explore deeper or more complex designs as long as the contract holds.
"""
import pytest
import torch

from steel_defect.utils import NUM_CLASSES, IMAGE_SIZE
from steel_defect.model import SteelCNN


@pytest.fixture
def model():
    """Create a SteelCNN instance."""
    return SteelCNN(num_classes=NUM_CLASSES)


@pytest.fixture
def dummy_batch():
    """Create a random input batch matching expected dimensions."""
    return torch.randn(2, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])


class TestModel1Init:
    """MODEL-1: __init__ layer definitions."""

    def test_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_has_trainable_parameters(self, model):
        assert model.num_parameters > 0, "Model should have trainable parameters"

    def test_reasonable_parameter_count(self, model):
        """Guard against degenerate models (too tiny or too huge)."""
        assert model.num_parameters > 100, (
            f"Model has only {model.num_parameters} parameters — "
            "too few for a useful image classifier"
        )
        assert model.num_parameters < 50_000_000, (
            f"Model has {model.num_parameters:,} parameters — "
            "over 50M is excessive for this task"
        )

    def test_has_conv_layers(self, model):
        """Model should contain at least one Conv2d layer."""
        conv_layers = [m for m in model.modules()
                       if isinstance(m, torch.nn.Conv2d)]
        assert len(conv_layers) >= 1, (
            "A CNN should contain at least one Conv2d layer"
        )

    def test_accepts_expected_input(self, model, dummy_batch):
        """Model must accept (batch, 3, 256, 256) input without error."""
        try:
            model(dummy_batch)
        except Exception as e:
            pytest.fail(
                f"Model failed on input shape {tuple(dummy_batch.shape)}: {e}"
            )


class TestModel2Forward:
    """MODEL-2: forward pass."""

    def test_output_shape(self, model, dummy_batch):
        output = model(dummy_batch)
        assert output.shape == (2, NUM_CLASSES), (
            f"Expected output shape (2, {NUM_CLASSES}), got {tuple(output.shape)}"
        )

    def test_output_dtype(self, model, dummy_batch):
        output = model(dummy_batch)
        assert output.dtype == torch.float32

    def test_single_image(self, model):
        """Forward pass with batch size 1."""
        single = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        output = model(single)
        assert output.shape == (1, NUM_CLASSES)

    def test_no_softmax_applied(self, model, dummy_batch):
        """Output should be raw logits — NOT softmax probabilities."""
        output = model(dummy_batch)
        row_sums = output.sum(dim=1)
        is_probability = torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=0.01
        )
        assert not is_probability, (
            "Output rows sum to ~1.0 — looks like softmax was applied. "
            "forward() should return raw logits, not probabilities."
        )

    def test_custom_num_classes(self):
        """Model should work with a different number of classes."""
        custom_model = SteelCNN(num_classes=3)
        x = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        output = custom_model(x)
        assert output.shape == (1, 3)

    def test_gradients_flow(self, model, dummy_batch):
        """Gradients should flow from output back to input layers."""
        output = model(dummy_batch)
        loss = output.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, (
            "No gradients reached model parameters — check that all layers "
            "are connected in the forward pass"
        )
