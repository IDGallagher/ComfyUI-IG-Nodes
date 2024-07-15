import logging
import math

import torch
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.models import ModelMixin
from einops import rearrange
from torch import nn

logger = logging.getLogger(__name__)

def generate_positional_encodings(length, hidden_dim):
    # Precompute positional encodings once in log space
    position = torch.arange(length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
    pe = torch.zeros(length, hidden_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class MotionPredictor(ModelMixin, ConfigMixin):
    def __init__(self, token_dim:int=768, hidden_dim:int=1024, num_heads:int=16, num_layers:int=8, total_frames:int=16, tokens_per_frame:int=16):
        super(MotionPredictor, self).__init__()
        self.total_frames = total_frames
        self.tokens_per_frame = tokens_per_frame

        # Initialize layers
        self.input_projection = nn.Linear(token_dim, hidden_dim)  # Project token to hidden dimension
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_projection = nn.Linear(hidden_dim, token_dim)  # Project back to token dimension
        # Initialize positional encodings
        self.positional_encodings = generate_positional_encodings(total_frames, hidden_dim)
        self.positional_encodings = nn.Parameter(self.positional_encodings, requires_grad=False)  # Optionally make it a parameter if you want it on the same device automatically

    def create_attention_mask(self, total_frames, num_tokens):
        # Initialize the mask with float('-inf') everywhere
        mask = torch.zeros((total_frames * num_tokens, total_frames * num_tokens), dtype=torch.bool, device=self.device)

        # Indices for the first frame tokens and the last frame tokens
        first_frame_indices = torch.arange(0, num_tokens, device=self.device)
        last_frame_indices = torch.arange((total_frames - 1) * num_tokens, total_frames * num_tokens, device=self.device)

        # Allow attention to the first and last frame tokens
        mask[first_frame_indices, :] = 0
        mask[last_frame_indices, :] = 0

        return mask

    def interpolate_tokens(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor):
        # Linear interpolation in the token space
        interpolation_steps = torch.linspace(0, 1, steps=self.total_frames, device=start_tokens.device, dtype=torch.float16)[:, None, None]
        start_tokens_expanded = start_tokens.unsqueeze(1)  # Shape becomes [batch_size, 1, tokens, token_dim]
        end_tokens_expanded = end_tokens.unsqueeze(1)      # Shape becomes [batch_size, 1, tokens, token_dim]
        interpolated_tokens = (start_tokens_expanded * (1 - interpolation_steps) + end_tokens_expanded * interpolation_steps)
        return interpolated_tokens  # Shape: [batch_size, total_frames, tokens, token_dim]

    def predict_motion(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor):
        start_tokens = start_tokens.to(self.device)
        end_tokens = end_tokens.to(self.device)

        # Get interpolated tokens
        interpolated_tokens = self.interpolate_tokens(start_tokens, end_tokens).to(self.dtype)

        # Flatten frames and tokens dimensions
        batch_size, total_frames, num_tokens, token_dim = interpolated_tokens.shape

        print(f"Interpolated tokens {interpolated_tokens.shape}")
        # Apply input projection
        projected_tokens = self.input_projection(interpolated_tokens)

        # Add positional encodings
        projected_tokens += self.positional_encodings[:total_frames * num_tokens].unsqueeze(0).unsqueeze(2)  # Add PE to each frame

        # Reshape to match the transformer expected input [seq_len, batch_size, hidden_dim]
        projected_tokens = rearrange(projected_tokens, 'b f t d -> (f t) b d')

        # Create an attention mask that only allows attending to the first and last frame
        attention_mask = self.create_attention_mask(total_frames, num_tokens)

        # Transformer predicts the motion along the new sequence dimension
        logger.debug(f"projected_tokens {projected_tokens.shape} attention_mask {attention_mask.shape}")
        motion_tokens = self.transformer(projected_tokens, projected_tokens, memory_mask=attention_mask)

        # Reshape back and apply output projection
        motion_tokens = rearrange(motion_tokens, '(f t) b d -> b f t d', t=num_tokens, f=total_frames)
        motion_tokens = self.output_projection(motion_tokens)

        return motion_tokens

    def forward(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor):
        return self.predict_motion(start_tokens, end_tokens)
