"""Optimal Transport Conditional Flow Matching."""
import torch

from .flow_model import FlowMatchingHead


class OTCFMHead(FlowMatchingHead):
    """OT-CFM with optimal transport plan."""
    def compute_loss(self, predictions: dict, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        emb = predictions['embeddings']
        t = torch.rand(targets.size(0), 1, device=targets.device)
        torch.randn_like(targets)
        sigma = 0.01
        mu_t = t * targets
        xt = mu_t + sigma * torch.randn_like(targets)
        velocity = targets
        inp = torch.cat([xt, emb, t], dim=-1)
        pred_v = self.velocity_net(inp)
        return torch.nn.functional.mse_loss(pred_v, velocity)
