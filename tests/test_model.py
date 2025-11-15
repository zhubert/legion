"""
Tests for TinyGPT model implementation
"""

import pytest
import torch
from sim.model import TinyGPT, create_model, MultiHeadAttention, FeedForward, TransformerBlock


class TestTinyGPT:
    """Test TinyGPT model"""

    def test_model_creation(self):
        """Test that model can be created with valid parameters"""
        model = TinyGPT(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_seq_len=256
        )
        assert model is not None
        assert model.vocab_size == 1000
        assert model.d_model == 128

    def test_forward_pass(self):
        """Test forward pass produces correct output shape"""
        model = TinyGPT(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
        batch_size = 2
        seq_len = 32

        # Create random input
        idx = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        logits, loss = model(idx)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is None  # No targets provided

    def test_forward_pass_with_loss(self):
        """Test forward pass with loss computation"""
        model = TinyGPT(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, 1000, (batch_size, seq_len))

        logits, loss = model(idx, targets)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is not None
        assert loss.item() > 0  # Loss should be positive

    def test_parameter_count(self):
        """Test parameter counting"""
        model = TinyGPT(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
        param_count = model.count_parameters()
        assert param_count > 0

    def test_model_factory(self):
        """Test create_model factory function"""
        for size in ['tiny', 'small', 'medium']:
            model = create_model(size)
            assert model is not None
            assert model.count_parameters() > 0

    def test_invalid_model_size(self):
        """Test that invalid size raises error"""
        with pytest.raises(ValueError):
            create_model('invalid_size')


class TestMultiHeadAttention:
    """Test multi-head attention mechanism"""

    def test_attention_forward(self):
        """Test attention forward pass"""
        d_model = 128
        n_heads = 4
        batch_size = 2
        seq_len = 10

        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attn(x)

        assert output.shape == x.shape

    def test_attention_with_mask(self):
        """Test attention with causal mask"""
        d_model = 128
        n_heads = 4
        batch_size = 2
        seq_len = 10

        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

        output = attn(x, mask)

        assert output.shape == x.shape


class TestFeedForward:
    """Test feed-forward network"""

    def test_feedforward(self):
        """Test feed-forward forward pass"""
        d_model = 128
        d_ff = 512
        batch_size = 2
        seq_len = 10

        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ff(x)

        assert output.shape == x.shape


class TestTransformerBlock:
    """Test transformer block"""

    def test_transformer_block(self):
        """Test transformer block forward pass"""
        d_model = 128
        n_heads = 4
        batch_size = 2
        seq_len = 10

        block = TransformerBlock(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = block(x)

        assert output.shape == x.shape
