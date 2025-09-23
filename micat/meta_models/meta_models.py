import torch 
import typing

class NormalVariationalNet(torch.nn.Module):
    """A simple neural network that simulate the
    reparameterization trick. Its parameters are
    the mean and std-vector
    """

    def __init__(self, base_net, **kwargs) -> None:
        """
        Args:
            base_net: the base network
        """
        super(NormalVariationalNet, self).__init__()
        base_params = base_net.get_user_params()
        

    def forward(self) -> typing.Dict[str, torch.Tensor]:
        """Output the parameters of the base network in list format to pass into higher monkeypatch"""
        out = self.mean + torch.randn_like(self.mean, device=self.mean.device) * torch.exp(self.log_std)
        return out
    
def clone_state_dict(model: torch.nn.Module):
    cloned_state_dict = {key: val.clone() for key, val in model.named_parameters()}
    return cloned_state_dict


def kl_divergence_gaussians(
    p: typing.List[torch.Tensor], q: typing.List[torch.Tensor]
) -> torch.Tensor:
    """Calculate KL divergence between 2 diagonal Gaussian

    Args: p and q are lists with [mean_tensor, log_std_tensor]

    Returns: KL divergence
    """
    assert len(p) == len(q) == 2, "Expected [mean, log_std] format"

    p_mean, p_log_std = p
    q_mean, q_log_std = q

    # Add numerical stability
    p_log_std = torch.clamp(p_log_std, min=-20, max=15)
    q_log_std = torch.clamp(q_log_std, min=-20, max=15)

    # KL divergence formula for diagonal Gaussians
    log_std_diff = p_log_std - q_log_std
    mean_diff = q_mean - p_mean
    
    # Compute KL divergence efficiently in one operation
    kl_div = 0.5 * torch.sum(
        torch.exp(2 * log_std_diff) + 
        torch.square(mean_diff) * torch.exp(-2 * q_log_std) - 
        1 - 
        2 * log_std_diff
    )
    
    return kl_div

def zero_grad(params: typing.Union[torch.nn.Module, typing.Dict, typing.List]) -> None:
    if isinstance(params, torch.nn.Module):
        params.zero_grad()
    elif isinstance(params, dict):
        for p in params.values():
            if getattr(p, "grad", None) is not None:
                p.grad.zero_()
    elif isinstance(params, list):
        for p in params:
            if getattr(p, "grad", None) is not None:
                p.grad.zero_()
    else:
        raise NotImplementedError()