"""
Baseline models for comparison.

These are simpler models WITHOUT segmentation guidance.
Used to demonstrate the improvement of our novel approach.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import warnings

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LSTMReportGenerator(nn.Module):
    """
    BASELINE 1: Simple LSTM-based report generator.
    
    Does NOT use segmentation features - only global image features.
    This demonstrates the value of segmentation-aware attention.
    """
    
    def __init__(
        self,
        image_feature_dim: int = 2048,
        hidden_dim: int = 512,
        vocab_size: int = 10000,
        num_layers: int = 2
    ):
        super().__init__()
        
        # Image encoder (simple CNN)
        self.image_encoder = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(768 * 64, image_feature_dim),
            nn.ReLU()
        )
        
        # LSTM decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Initial hidden state from image features
        self.feature_to_hidden = nn.Linear(image_feature_dim, hidden_dim * num_layers)
        self.feature_to_cell = nn.Linear(image_feature_dim, hidden_dim * num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        image_features: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        max_length: int = 100
    ):
        """
        Generate report from image features.
        
        Args:
            image_features: Global image features [B, C, D, H, W]
            target_tokens: Ground truth tokens for training [B, seq_len]
            max_length: Maximum generation length
        """
        batch_size = image_features.size(0)
        
        # Encode image
        img_features = self.image_encoder(image_features)  # [B, feature_dim]
        
        # Initialize LSTM hidden states
        h0 = self.feature_to_hidden(img_features).view(self.lstm.num_layers, batch_size, -1)
        c0 = self.feature_to_cell(img_features).view(self.lstm.num_layers, batch_size, -1)
        
        if target_tokens is not None:
            # Training mode: teacher forcing
            token_embeds = self.embedding(target_tokens)
            lstm_out, _ = self.lstm(token_embeds, (h0, c0))
            logits = self.fc_out(lstm_out)
            return {'logits': logits}
        else:
            # Inference mode: autoregressive generation
            generated = []
            input_token = torch.zeros(batch_size, 1, dtype=torch.long)  # Start token
            hidden = (h0, c0)
            
            for _ in range(max_length):
                token_embed = self.embedding(input_token)
                lstm_out, hidden = self.lstm(token_embed, hidden)
                logits = self.fc_out(lstm_out)
                next_token = torch.argmax(logits, dim=-1)
                generated.append(next_token)
                input_token = next_token
            
            return {'generated_ids': torch.cat(generated, dim=1)}


class TransformerBaseline(nn.Module):
    """
    BASELINE 2: Standard Transformer without anatomy-aware attention.
    
    Uses global image features but NO segmentation guidance.
    """
    
    def __init__(
        self,
        image_feature_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        vocab_size: int = 10000
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Image feature projection
        self.image_projector = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(image_feature_dim * 64, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Reshape to sequence
        self.to_sequence = nn.Linear(hidden_dim * 4, hidden_dim)
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        image_features: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        max_length: int = 100
    ):
        batch_size = image_features.size(0)
        
        # Project image to sequence (memory for transformer)
        img_proj = self.image_projector(image_features)
        memory = self.to_sequence(img_proj).unsqueeze(1)  # [B, 1, hidden]
        
        if target_tokens is not None:
            # Training: teacher forcing
            token_embeds = self.token_embedding(target_tokens)
            seq_len = token_embeds.size(1)
            token_embeds = token_embeds + self.pos_embedding[:, :seq_len, :]
            
            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            
            # Decode
            output = self.transformer(
                tgt=token_embeds,
                memory=memory,
                tgt_mask=tgt_mask
            )
            logits = self.fc_out(output)
            return {'logits': logits}
        else:
            # Inference: autoregressive
            generated = []
            input_tokens = torch.zeros(batch_size, 1, dtype=torch.long)
            
            for step in range(max_length):
                token_embeds = self.token_embedding(input_tokens)
                token_embeds = token_embeds + self.pos_embedding[:, :step+1, :]
                
                output = self.transformer(tgt=token_embeds, memory=memory)
                logits = self.fc_out(output[:, -1:, :])
                next_token = torch.argmax(logits, dim=-1)
                
                generated.append(next_token)
                input_tokens = torch.cat([input_tokens, next_token], dim=1)
            
            return {'generated_ids': torch.cat(generated, dim=1)}


class SimpleCNNLSTM(nn.Module):
    """
    BASELINE 3: Most basic approach - CNN feature extractor + LSTM.
    
    No pre-training, no segmentation. Pure end-to-end from scratch.
    """
    
    def __init__(self):
        super().__init__()
        
        # Very simple 3D CNN
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        
        self.fc_init = nn.Linear(64, 256)
        self.embedding = nn.Embedding(10000, 256)
        self.fc_out = nn.Linear(256, 10000)
    
    def forward(self, ct_volume, max_length=100):
        # Extract features
        features = self.cnn(ct_volume)  # [B, 64]
        
        # Initialize LSTM from features
        h0 = self.fc_init(features).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        
        # Simple greedy generation
        batch_size = ct_volume.size(0)
        input_token = torch.zeros(batch_size, 1, dtype=torch.long)
        hidden = (h0, c0)
        generated = []
        
        for _ in range(max_length):
            embed = self.embedding(input_token)
            out, hidden = self.lstm(embed, hidden)
            logits = self.fc_out(out)
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token)
            input_token = next_token
        
        return {'generated_ids': torch.cat(generated, dim=1)}


# For easier comparison
BASELINES = {
    'lstm': LSTMReportGenerator,
    'transformer': TransformerBaseline,
    'simple_cnn_lstm': SimpleCNNLSTM
}


if __name__ == "__main__":
    # Test baseline models
    print("Testing baseline models...")
    
    # Mock data
    batch_size = 2
    fake_image = torch.randn(batch_size, 768, 4, 4, 4)
    
    # LSTM baseline
    lstm_model = LSTMReportGenerator()
    lstm_out = lstm_model(fake_image)
    print(f"LSTM output shape: {lstm_out['generated_ids'].shape}")
    
    # Transformer baseline
    transformer_model = TransformerBaseline()
    transformer_out = transformer_model(fake_image)
    print(f"Transformer output shape: {transformer_out['generated_ids'].shape}")
    
    print("Baseline models working!")
