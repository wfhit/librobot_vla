"""Figure AI Helix-style 3-tier hierarchical VLA framework implementation."""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractVLA
from ..vlm.base import AbstractVLM


class HelixVLA(AbstractVLA):
    """
    Figure AI Helix-style 3-tier hierarchical Vision-Language-Action framework.

    Architecture:
        - High-Level: VLM for task understanding and planning
        - Mid-Level: Policy network for action sequencing
        - Low-Level: Motor control for precise execution
        - Hierarchical structure with different time scales

    Three Tiers:
        1. High-Level (VLM): Processes language instructions, visual scene
           understanding, and generates high-level plans/goals.
           Update frequency: ~1-10 Hz

        2. Mid-Level (Policy): Translates high-level goals into action sequences
           considering current state and recent history.
           Update frequency: ~10-30 Hz

        3. Low-Level (Motor Control): Executes actions with precise motor control,
           handles dynamics, and provides smooth trajectories.
           Update frequency: ~50-100 Hz

    Key Features:
        - Hierarchical Control: Multi-scale temporal reasoning
        - VLM Planning: High-level task understanding
        - Policy Bridge: Translates plans to actions
        - Motor Control: Low-level execution
        - Temporal Abstraction: Different update rates per tier

    Args:
        vlm: Pre-trained VLM for high-level reasoning
        action_dim: Dimension of action space
        state_dim: Dimension of proprioceptive state
        plan_dim: Dimension of high-level plan representation
        hidden_dim: Hidden dimension for networks
        num_policy_layers: Number of layers in policy network
        num_motor_layers: Number of layers in motor control network
        plan_horizon: Planning horizon (time steps)
        freeze_vlm: Whether to freeze VLM
    """

    def __init__(
        self,
        vlm: AbstractVLM,
        action_dim: int,
        state_dim: Optional[int] = None,
        plan_dim: int = 256,
        hidden_dim: int = 512,
        num_policy_layers: int = 4,
        num_motor_layers: int = 2,
        plan_horizon: int = 10,
        freeze_vlm: bool = True,
    ):
        super().__init__()

        self.vlm = vlm
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.plan_dim = plan_dim
        self.hidden_dim = hidden_dim
        self.plan_horizon = plan_horizon

        # Get VLM embedding dimension
        self.vlm_dim = vlm.get_embedding_dim()

        # Freeze VLM for stability
        if freeze_vlm:
            self.freeze_backbone()

        # ============================================
        # HIGH-LEVEL: VLM-based Task Understanding & Planning
        # ============================================

        # VLM output projection to plan space
        self.vlm_to_plan = nn.Sequential(
            nn.Linear(self.vlm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, plan_dim),
            nn.LayerNorm(plan_dim),
        )

        # Plan refinement network
        self.plan_refiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=plan_dim,
                nhead=4,
                dim_feedforward=plan_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=2
        )

        # ============================================
        # MID-LEVEL: Policy Network
        # ============================================

        # State encoder for mid-level
        if state_dim is not None:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.state_encoder = None

        # Policy network input dimension
        policy_input_dim = plan_dim
        if state_dim is not None:
            policy_input_dim += hidden_dim

        # Policy network: (plan, state) -> action sequence
        policy_layers = []
        dims = [policy_input_dim] + [hidden_dim] * num_policy_layers
        for i in range(len(dims) - 1):
            policy_layers.append(nn.Linear(dims[i], dims[i + 1]))
            policy_layers.append(nn.LayerNorm(dims[i + 1]))
            policy_layers.append(nn.GELU())
            policy_layers.append(nn.Dropout(0.1))

        self.policy_network = nn.Sequential(*policy_layers)

        # Action sequence head (predicts intermediate actions)
        self.action_sequence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * plan_horizon),
        )

        # ============================================
        # LOW-LEVEL: Motor Control Network
        # ============================================

        # Motor control input: (current_state, target_action, plan_context)
        motor_input_dim = action_dim + plan_dim
        if state_dim is not None:
            motor_input_dim += state_dim

        # Motor control network for precise execution
        motor_layers = []
        dims = [motor_input_dim] + [hidden_dim] * num_motor_layers + [action_dim]
        for i in range(len(dims) - 1):
            motor_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on last layer
                motor_layers.append(nn.LayerNorm(dims[i + 1]))
                motor_layers.append(nn.GELU())

        self.motor_control = nn.Sequential(*motor_layers)

        # Temporal smoothing for motor control
        self.temporal_smoother = nn.Conv1d(
            in_channels=action_dim,
            out_channels=action_dim,
            kernel_size=3,
            padding=1,
            groups=action_dim,  # Depthwise convolution
        )

    def high_level_planning(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        High-level planning using VLM.

        Args:
            images: Input images [batch, C, H, W]
            text: Text instructions
            **kwargs: Additional arguments

        Returns:
            Plan representation [batch, plan_dim]
        """
        # Process through VLM
        with torch.set_grad_enabled(not self.training or self.vlm.training):
            vlm_outputs = self.vlm(images, text=text, **kwargs)
            vlm_embeddings = vlm_outputs['embeddings']

        # Pool embeddings
        if vlm_embeddings.dim() == 3:
            vlm_embeddings = vlm_embeddings.mean(dim=1)

        # Project to plan space
        plan = self.vlm_to_plan(vlm_embeddings)

        # Refine plan (self-attention)
        plan_refined = self.plan_refiner(plan.unsqueeze(1)).squeeze(1)

        return plan_refined

    def mid_level_policy(
        self,
        plan: torch.Tensor,
        proprioception: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mid-level policy for action sequencing.

        Args:
            plan: High-level plan [batch, plan_dim]
            proprioception: Current state [batch, state_dim]

        Returns:
            Action sequence [batch, plan_horizon, action_dim]
        """
        # Encode state
        if proprioception is not None and self.state_encoder is not None:
            state_features = self.state_encoder(proprioception)
            policy_input = torch.cat([plan, state_features], dim=-1)
        else:
            policy_input = plan

        # Policy network
        policy_features = self.policy_network(policy_input)

        # Predict action sequence
        action_seq_flat = self.action_sequence_head(policy_features)
        action_seq = action_seq_flat.view(-1, self.plan_horizon, self.action_dim)

        return action_seq

    def low_level_motor_control(
        self,
        target_action: torch.Tensor,
        plan: torch.Tensor,
        proprioception: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Low-level motor control for precise execution.

        Args:
            target_action: Target action from policy [batch, action_dim]
            plan: High-level plan [batch, plan_dim]
            proprioception: Current state [batch, state_dim]

        Returns:
            Refined action [batch, action_dim]
        """
        # Build motor control input
        motor_input_parts = [target_action, plan]
        if proprioception is not None:
            motor_input_parts.append(proprioception)

        motor_input = torch.cat(motor_input_parts, dim=-1)

        # Motor control network
        refined_action = self.motor_control(motor_input)

        return refined_action

    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_sequences: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Helix VLA (all three tiers).

        Args:
            images: Input images [batch_size, C, H, W]
            text: Text instructions
            proprioception: Proprioceptive state [batch_size, state_dim]
            actions: Target actions for training [batch_size, action_dim]
            action_sequences: Target action sequences [batch_size, plan_horizon, action_dim]
            **kwargs: Additional arguments

        Returns:
            Dictionary containing outputs from all three levels
        """
        batch_size = images.size(0)

        # ============================================
        # HIGH-LEVEL: Planning
        # ============================================
        plan = self.high_level_planning(images, text, **kwargs)

        # ============================================
        # MID-LEVEL: Policy
        # ============================================
        action_sequence = self.mid_level_policy(plan, proprioception)

        # ============================================
        # LOW-LEVEL: Motor Control
        # ============================================
        # Use first action from sequence as target
        target_action = action_sequence[:, 0, :]
        refined_action = self.low_level_motor_control(target_action, plan, proprioception)

        # Apply temporal smoothing if we have sequence
        if action_sequence.size(1) > 1:
            # Transpose for Conv1d: [batch, action_dim, seq_len]
            smoothed_seq = self.temporal_smoother(
                action_sequence.transpose(1, 2)
            ).transpose(1, 2)
        else:
            smoothed_seq = action_sequence

        if actions is not None:
            # Training mode: compute losses
            losses = {}

            # Low-level action loss (immediate action)
            action_loss = F.mse_loss(refined_action, actions)
            losses['action_loss'] = action_loss

            # Mid-level sequence loss (if provided)
            if action_sequences is not None:
                seq_loss = F.mse_loss(smoothed_seq, action_sequences)
                losses['sequence_loss'] = seq_loss
                total_loss = action_loss + 0.5 * seq_loss
            else:
                total_loss = action_loss

            losses['total_loss'] = total_loss

            return {
                'actions': refined_action,
                'action_sequence': smoothed_seq,
                'plan': plan,
                'loss': total_loss,
                **losses,
            }
        else:
            # Inference mode
            return {
                'actions': refined_action,
                'action_sequence': smoothed_seq,
                'plan': plan,
            }

    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        """
        Predict actions for inference.

        Args:
            images: Input images
            text: Text instructions
            proprioception: Proprioceptive state
            return_sequence: Whether to return full action sequence
            **kwargs: Additional arguments

        Returns:
            Predicted actions [batch_size, action_dim]
            Or tuple of (actions, action_sequence) if return_sequence=True
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

            if return_sequence:
                return outputs['actions'], outputs['action_sequence']
            else:
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

        if 'action_loss' in predictions:
            losses['action_loss'] = predictions['action_loss']

        if 'sequence_loss' in predictions:
            losses['sequence_loss'] = predictions['sequence_loss']

        if 'total_loss' in predictions:
            losses['total_loss'] = predictions['total_loss']
        else:
            losses['total_loss'] = predictions.get('loss', torch.tensor(0.0))

        return losses

    def get_config(self) -> Dict[str, Any]:
        """
        Get framework configuration.

        Returns:
            Dictionary containing configuration
        """
        return {
            'type': 'HelixVLA',
            'vlm_config': self.vlm.config,
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'plan_dim': self.plan_dim,
            'hidden_dim': self.hidden_dim,
            'plan_horizon': self.plan_horizon,
        }
