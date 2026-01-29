"""Physical Intelligence π0-style VLA framework implementation."""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .base import AbstractVLA
from ..vlm.base import AbstractVLM
from ..encoders.state.tokenizer_encoder import TokenizerStateEncoder
from ..action_heads.flow_matching.flow_model import FlowMatchingHead


class Pi0VLA(AbstractVLA):
    """
    Physical Intelligence π0-style Vision-Language-Action framework.
    
    Architecture:
        - VLM backbone processes images and instructions
        - State tokenization: Proprioception → discrete tokens → embeddings
        - Tokens are concatenated with VLM token sequence
        - Block-wise attention for efficient processing
        - Flow matching for smooth action prediction
    
    Key Features:
        - State Tokenization: VQ-VAE style tokenization of continuous state
        - Token-based Architecture: State as first-class tokens
        - Flow Matching: Continuous normalizing flows for actions
        - Unified Sequence: Visual, language, and state tokens processed together
    
    Args:
        vlm: Pre-trained VLM backbone
        action_dim: Dimension of action space
        state_dim: Dimension of proprioceptive state
        hidden_dim: Hidden dimension for processing
        num_state_tokens: Number of tokens in state codebook
        flow_steps: Number of flow matching steps
        freeze_vlm: Whether to freeze VLM backbone
        fine_tune_vlm: Whether to fine-tune VLM layers
    """
    
    def __init__(
        self,
        vlm: AbstractVLM,
        action_dim: int,
        state_dim: Optional[int] = None,
        hidden_dim: int = 512,
        num_state_tokens: int = 1024,
        flow_steps: int = 50,
        freeze_vlm: bool = False,
        fine_tune_vlm: bool = True,
    ):
        super().__init__()
        
        self.vlm = vlm
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_state_tokens = num_state_tokens
        self.flow_steps = flow_steps
        
        # Get VLM embedding dimension
        self.vlm_dim = vlm.get_embedding_dim()
        
        # Freeze VLM if requested
        if freeze_vlm:
            self.freeze_backbone()
        elif not fine_tune_vlm:
            # Partial freeze: keep most layers frozen
            for name, param in self.vlm.named_parameters():
                if 'layer' in name:
                    layer_num = int(name.split('.')[1]) if name.split('.')[1].isdigit() else 0
                    if layer_num < 20:  # Freeze early layers
                        param.requires_grad = False
        
        # State tokenizer encoder
        if state_dim is not None:
            self.state_tokenizer = TokenizerStateEncoder(
                input_dim=state_dim,
                output_dim=self.vlm_dim,  # Match VLM dimension
                num_tokens=num_state_tokens,
                hidden_dim=hidden_dim,
                commitment_cost=0.25,
                use_ema=True,
            )
        else:
            self.state_tokenizer = None
        
        # Projection layers for unified embedding space
        self.vlm_proj = nn.Sequential(
            nn.Linear(self.vlm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Block-wise attention transformer for processing combined sequence
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(4)
        ])
        
        # Flow matching action head
        self.action_head = FlowMatchingHead(
            input_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        
        # Action token query
        self.action_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of π0 VLA.
        
        Args:
            images: Input images [batch_size, C, H, W]
            text: Optional text instructions
            proprioception: Optional proprioceptive state [batch_size, state_dim]
            actions: Optional action targets for training
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, action_dim]
                - 'loss': Training loss (if actions provided)
                - 'vq_loss': Vector quantization loss (if using state)
        """
        batch_size = images.size(0)
        
        # Extract VLM features
        vlm_outputs = self.vlm(images, text=text, **kwargs)
        vlm_embeddings = vlm_outputs['embeddings']  # [batch, seq_len, vlm_dim]
        
        # Project VLM embeddings
        vlm_features = self.vlm_proj(vlm_embeddings)  # [batch, seq_len, hidden_dim]
        
        # Tokenize and embed proprioceptive state
        vq_loss = None
        if proprioception is not None and self.state_tokenizer is not None:
            state_embeddings, state_indices, vq_loss = self.state_tokenizer(
                proprioception, return_indices=True
            )
            # Project to hidden dimension
            state_embeddings = state_embeddings.unsqueeze(1)  # [batch, 1, hidden_dim]
            
            # Concatenate state tokens with VLM tokens
            # State tokens come first as conditioning
            combined_tokens = torch.cat([state_embeddings, vlm_features], dim=1)
        else:
            combined_tokens = vlm_features
        
        # Add action query token at the end
        action_queries = self.action_query.expand(batch_size, -1, -1)
        combined_tokens = torch.cat([combined_tokens, action_queries], dim=1)
        
        # Process through transformer with block-wise attention
        for layer in self.transformer:
            combined_tokens = layer(combined_tokens)
        
        # Extract action token (last token)
        action_token = combined_tokens[:, -1, :]  # [batch, hidden_dim]
        
        # Predict actions using flow matching
        if actions is not None:
            # Training mode: compute loss
            action_pred = self.action_head(action_token)
            action_loss = self.action_head.compute_loss(action_pred, actions)
            
            # Total loss includes VQ loss if using state tokenization
            total_loss = action_loss
            if vq_loss is not None:
                total_loss = total_loss + 0.1 * vq_loss  # Weight VQ loss
            
            return {
                'actions': self.action_head.sample(action_token, steps=self.flow_steps),
                'loss': total_loss,
                'action_loss': action_loss,
                'vq_loss': vq_loss if vq_loss is not None else torch.tensor(0.0),
                'action_token': action_token,
            }
        else:
            # Inference mode: sample actions
            predicted_actions = self.action_head.sample(
                action_token, steps=self.flow_steps
            )
            
            return {
                'actions': predicted_actions,
                'action_token': action_token,
            }
    
    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict actions for inference.
        
        Args:
            images: Input images
            text: Optional text instructions
            proprioception: Optional proprioceptive state
            **kwargs: Additional arguments
            
        Returns:
            Predicted actions [batch_size, action_dim]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                images=images,
                text=text,
                proprioception=proprioception,
                actions=None,
                **kwargs
            )
            return outputs['actions']
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for training.
        
        Args:
            predictions: Model predictions from forward()
            targets: Ground truth targets
            **kwargs: Additional loss computation arguments
            
        Returns:
            Dictionary containing losses
        """
        losses = {}
        
        # Action loss
        if 'action_loss' in predictions:
            losses['action_loss'] = predictions['action_loss']
        
        # VQ loss
        if 'vq_loss' in predictions:
            losses['vq_loss'] = predictions['vq_loss']
        
        # Total loss
        if 'loss' in predictions:
            losses['total_loss'] = predictions['loss']
        
        return losses
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get framework configuration.
        
        Returns:
            Dictionary containing configuration
        """
        return {
            'type': 'Pi0VLA',
            'vlm_config': self.vlm.config,
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim,
            'num_state_tokens': self.num_state_tokens,
            'flow_steps': self.flow_steps,
        }
