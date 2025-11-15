"""
Simple Transformer models for testing distributed training

Includes minimal GPT-style models of various sizes for PoC testing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Project and reshape to (batch, n_heads, seq_len, d_head)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout1(attn_out)

        # Feed-forward with residual
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout2(ff_out)

        return x


class TinyGPT(nn.Module):
    """
    Minimal GPT-style language model for testing

    This is a simplified transformer decoder for proof-of-concept testing.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 max_seq_len: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff=4*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (common in LLMs)
        self.head.weight = self.token_emb.weight

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        """
        Args:
            idx: Input token indices (batch_size, seq_len)
            targets: Target token indices for loss computation (batch_size, seq_len)

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        batch_size, seq_len = idx.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # Generate causal mask (prevent attending to future tokens)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).unsqueeze(0).unsqueeze(0)

        # Embeddings
        tok_emb = self.token_emb(idx)  # (batch, seq_len, d_model)
        pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.pos_emb(pos)  # (1, seq_len, d_model)

        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq_len, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(size: str = "tiny") -> TinyGPT:
    """
    Factory function to create models of different sizes

    Args:
        size: One of 'tiny', 'small', 'medium', 'large'

    Returns:
        TinyGPT model
    """
    configs = {
        'tiny': {
            'vocab_size': 1000,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 4,
            'max_seq_len': 128,
        },
        'small': {
            'vocab_size': 5000,
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'max_seq_len': 256,
        },
        'medium': {
            'vocab_size': 10000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 8,
            'max_seq_len': 512,
        },
        'large': {
            'vocab_size': 50000,
            'd_model': 768,
            'n_heads': 12,
            'n_layers': 12,
            'max_seq_len': 1024,
        }
    }

    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")

    config = configs[size]
    model = TinyGPT(**config)

    print(f"\nCreated {size} model:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  d_model: {config['d_model']}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Heads: {config['n_heads']}")
    print(f"  Vocab: {config['vocab_size']}")

    return model


if __name__ == "__main__":
    print("Testing TinyGPT models\n")

    # Test each model size
    for size in ['tiny', 'small', 'medium']:
        model = create_model(size)

        # Test forward pass
        batch_size = 2
        seq_len = 32
        vocab_size = model.vocab_size

        # Random input
        idx = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss = model(idx, targets)

        print(f"  Input shape: {idx.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        print()

    print("✓ All model tests passed!")
