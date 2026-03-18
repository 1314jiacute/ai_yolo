"""Unit tests for Discriminator model in gan_augment_longtail_smallobj.py"""
import pytest
import torch
import torch.nn as nn
from gan_augment_longtail_smallobj import Discriminator, ResBlock


class TestResBlock:
    """Test cases for ResBlock component"""

    def test_resblock_initialization(self):
        """Test ResBlock initialization with different channel sizes"""
        for channels in [32, 64, 128, 256, 512]:
            block = ResBlock(channels)
            assert block.conv1.in_channels == channels
            assert block.conv1.out_channels == channels
            assert block.conv2.in_channels == channels
            assert block.conv2.out_channels == channels

    def test_resblock_forward_shape(self):
        """Test ResBlock forward pass maintains tensor shape"""
        for channels in [32, 64, 128]:
            block = ResBlock(channels)
            input_tensor = torch.randn(2, channels, 16, 16)
            output = block(input_tensor)
            assert output.shape == input_tensor.shape

    def test_resblock_forward_values(self):
        """Test ResBlock forward produces valid output values"""
        block = ResBlock(64)
        input_tensor = torch.randn(1, 64, 8, 8)
        output = block(input_tensor)
        # Check that output is finite (no NaN or Inf)
        assert torch.isfinite(output).all()

    def test_resbatch_gradients(self):
        """Test ResBlock gradients flow correctly"""
        block = ResBlock(128)
        input_tensor = torch.randn(2, 128, 8, 8, requires_grad=True)
        output = block(input_tensor)
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None
        assert input_tensor.grad.abs().sum() > 0


class TestDiscriminatorInitialization:
    """Test cases for Discriminator initialization"""

    def test_default_initialization(self):
        """Test Discriminator initialization with default parameters"""
        disc = Discriminator()
        assert disc.down1[0].in_channels == 3

    def test_custom_channels_initialization(self):
        """Test Discriminator initialization with custom channel count"""
        disc = Discriminator(channels=1)
        assert disc.down1[0].in_channels == 1
        disc_4ch = Discriminator(channels=4)
        assert disc_4ch.down1[0].in_channels == 4

    def test_layer_structure(self):
        """Test that all layers are properly initialized"""
        disc = Discriminator()
        # Check all downsampling layers exist
        assert hasattr(disc, 'down1')
        assert hasattr(disc, 'down2')
        assert hasattr(disc, 'down3')
        assert hasattr(disc, 'down4')
        assert hasattr(disc, 'down5')
        assert hasattr(disc, 'down6')
        assert hasattr(disc, 'final')

    def test_down1_structure(self):
        """Test down1 layer structure: 256x256 -> 128x128"""
        disc = Discriminator()
        assert isinstance(disc.down1[0], nn.Conv2d)
        assert disc.down1[0].kernel_size == (4, 4)
        assert disc.down1[0].stride == (2, 2)
        assert disc.down1[0].padding == (1, 1)
        assert disc.down1[0].bias is False

    def test_down2_structure(self):
        """Test down2 layer structure: 128x128 -> 64x64"""
        disc = Discriminator()
        assert disc.down2[0].in_channels == 32
        assert disc.down2[0].out_channels == 64
        assert len(disc.down2) == 4  # Conv, BN, LeakyReLU, ResBlock

    def test_down3_structure(self):
        """Test down3 layer structure: 64x64 -> 32x32"""
        disc = Discriminator()
        assert disc.down3[0].in_channels == 64
        assert disc.down3[0].out_channels == 128

    def test_down4_structure(self):
        """Test down4 layer structure: 32x32 -> 16x16"""
        disc = Discriminator()
        assert disc.down4[0].in_channels == 128
        assert disc.down4[0].out_channels == 256

    def test_down5_structure(self):
        """Test down5 layer structure: 16x16 -> 8x8"""
        disc = Discriminator()
        assert disc.down5[0].in_channels == 256
        assert disc.down5[0].out_channels == 512

    def test_down6_structure(self):
        """Test down6 layer structure: 8x8 -> 4x4"""
        disc = Discriminator()
        assert disc.down6[0].in_channels == 512
        assert disc.down6[0].out_channels == 1024

    def test_final_structure(self):
        """Test final layer structure: 4x4 -> 1x1"""
        disc = Discriminator()
        assert disc.final[0].in_channels == 1024
        assert disc.final[0].out_channels == 1
        assert isinstance(disc.final[1], nn.Sigmoid)


class TestDiscriminatorForward:
    """Test cases for Discriminator forward pass"""

    def test_forward_single_image(self):
        """Test forward pass with single image"""
        disc = Discriminator()
        input_tensor = torch.randn(1, 3, 256, 256)
        output = disc(input_tensor)
        assert output.shape == (1, 1, 1, 1)

    def test_forward_batch(self):
        """Test forward pass with batch of images"""
        disc = Discriminator()
        input_tensor = torch.randn(4, 3, 256, 256)
        output = disc(input_tensor)
        assert output.shape == (4, 1, 1, 1)

    def test_forward_output_range(self):
        """Test that output is in range [0, 1] due to sigmoid"""
        disc = Discriminator()
        input_tensor = torch.randn(2, 3, 256, 256)
        output = disc(input_tensor)
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_forward_custom_channels(self):
        """Test forward pass with custom input channels"""
        disc = Discriminator(channels=1)
        input_tensor = torch.randn(2, 1, 256, 256)
        output = disc(input_tensor)
        assert output.shape == (2, 1, 1, 1)

    def test_forward_finite_values(self):
        """Test that forward pass produces finite values (no NaN or Inf)"""
        disc = Discriminator()
        input_tensor = torch.randn(3, 3, 256, 256)
        output = disc(input_tensor)
        assert torch.isfinite(output).all()

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic given same input"""
        disc = Discriminator()
        disc.eval()  # Set to eval mode for deterministic behavior
        input_tensor = torch.randn(2, 3, 256, 256)
        output1 = disc(input_tensor)
        output2 = disc(input_tensor)
        assert torch.allclose(output1, output2)

    def test_forward_different_inputs(self):
        """Test that different inputs produce different outputs"""
        disc = Discriminator()
        disc.eval()
        input_tensor1 = torch.randn(2, 3, 256, 256)
        input_tensor2 = torch.randn(2, 3, 256, 256)
        output1 = disc(input_tensor1)
        output2 = disc(input_tensor2)
        assert not torch.allclose(output1, output2)


class TestDiscriminatorGradients:
    """Test cases for gradient computation"""

    def test_gradient_flow(self):
        """Test that gradients flow through the network"""
        disc = Discriminator()
        input_tensor = torch.randn(2, 3, 256, 256, requires_grad=True)
        output = disc(input_tensor)
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None
        assert input_tensor.grad.abs().sum() > 0

    def test_parameter_gradients_exist(self):
        """Test that all parameters have gradients"""
        disc = Discriminator()
        input_tensor = torch.randn(2, 3, 256, 256)
        output = disc(input_tensor)
        loss = output.mean()
        loss.backward()
        for name, param in disc.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_no_nan_gradients(self):
        """Test that gradients are finite (no NaN or Inf)"""
        disc = Discriminator()
        input_tensor = torch.randn(2, 3, 256, 256)
        output = disc(input_tensor)
        loss = output.mean()
        loss.backward()
        for param in disc.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient detected"

    def test_train_eval_mode(self):
        """Test that train and eval modes work correctly"""
        disc = Discriminator()
        disc.train()
        assert disc.training

        disc.eval()
        assert not disc.training


class TestDiscriminatorEdgeCases:
    """Test cases for edge cases and boundary conditions"""

    def test_minimum_batch_size(self):
        """Test with minimum batch size of 1"""
        disc = Discriminator()
        input_tensor = torch.randn(1, 3, 256, 256)
        output = disc(input_tensor)
        assert output.shape == (1, 1, 1, 1)
        assert torch.isfinite(output).all()

    def test_large_batch_size(self):
        """Test with larger batch size"""
        disc = Discriminator()
        input_tensor = torch.randn(16, 3, 256, 256)
        output = disc(input_tensor)
        assert output.shape == (16, 1, 1, 1)
        assert torch.isfinite(output).all()

    def test_exact_input_size(self):
        """Test with exact expected input size 256x256"""
        disc = Discriminator()
        input_tensor = torch.randn(2, 3, 256, 256)
        output = disc(input_tensor)
        assert output.shape == (2, 1, 1, 1)

    def test_extreme_input_values(self):
        """Test with extreme input values"""
        disc = Discriminator()
        # Test with very large values
        input_tensor = torch.ones(1, 3, 256, 256) * 10
        output = disc(input_tensor)
        assert torch.isfinite(output).all()

        # Test with very small values
        input_tensor = torch.ones(1, 3, 256, 256) * 0.01
        output = disc(input_tensor)
        assert torch.isfinite(output).all()

    def test_zero_input(self):
        """Test with all-zero input"""
        disc = Discriminator()
        input_tensor = torch.zeros(1, 3, 256, 256)
        output = disc(input_tensor)
        assert output.shape == (1, 1, 1, 1)
        assert torch.isfinite(output).all()


class TestDiscriminatorModeSwitching:
    """Test cases for train/eval mode behavior"""

    def test_train_mode_batchnorm(self):
        """Test that BatchNorm layers are in training mode"""
        disc = Discriminator()
        disc.train()
        for module in disc.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert module.training

    def test_eval_mode_batchnorm(self):
        """Test that BatchNorm layers are in eval mode"""
        disc = Discriminator()
        disc.eval()
        for module in disc.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert not module.training


class TestDiscriminatorParameterCount:
    """Test cases for model parameter statistics"""

    def test_total_parameters(self):
        """Test that discriminator has expected number of parameters"""
        disc = Discriminator()
        total_params = sum(p.numel() for p in disc.parameters())
        # Discriminator should have substantial number of parameters
        assert total_params > 1000000  # At least 1M parameters for 256x256

    def test_trainable_parameters(self):
        """Test that all parameters are trainable"""
        disc = Discriminator()
        trainable_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in disc.parameters())
        assert trainable_params == total_params

    def test_no_frozen_parameters(self):
        """Verify no parameters are frozen by default"""
        disc = Discriminator()
        for param in disc.parameters():
            assert param.requires_grad


class TestDiscriminatorLossComputation:
    """Test cases for loss computation scenarios"""

    def test_bce_loss_with_real_labels(self):
        """Test BCE loss computation with real labels"""
        disc = Discriminator()
        input_tensor = torch.randn(2, 3, 256, 256)
        output = disc(input_tensor).reshape(-1)
        
        # Real labels
        real_labels = torch.ones_like(output) * 0.9
        criterion = nn.BCELoss()
        loss = criterion(output, real_labels)
        
        assert torch.isfinite(loss)
        assert loss > 0

    def test_bce_loss_with_fake_labels(self):
        """Test BCE loss computation with fake labels"""
        disc = Discriminator()
        input_tensor = torch.randn(2, 3, 256, 256)
        output = disc(input_tensor).reshape(-1)
        
        # Fake labels
        fake_labels = torch.zeros_like(output) + 0.1
        criterion = nn.BCELoss()
        loss = criterion(output, fake_labels)
        
        assert torch.isfinite(loss)
        assert loss > 0

    def test_discriminator_real_fake_loss(self):
        """Test combined real/fake loss computation"""
        disc = Discriminator()
        criterion = nn.BCELoss()
        
        # Real images
        real_input = torch.randn(2, 3, 256, 256)
        disc_real = disc(real_input).reshape(-1)
        loss_real = criterion(disc_real, torch.ones_like(disc_real) * 0.9)
        
        # Fake images
        fake_input = torch.randn(2, 3, 256, 256)
        disc_fake = disc(fake_input).reshape(-1)
        loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake) + 0.1)
        
        # Combined loss
        loss_disc = (loss_real + loss_fake) / 2
        
        assert torch.isfinite(loss_disc)
        assert loss_disc > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
