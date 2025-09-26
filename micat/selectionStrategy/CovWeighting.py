import torch
from torch import nn

class CoVWeightingLoss(nn.Module):
    def __init__(self, device: str):
        super().__init__() 
        self.nb_losses: int = 2
        self.device: str = device

        # Initialize tensors for online statistics
        self.t: int = 0  # Time step
        self.mean_L: torch.Tensor = torch.zeros(self.nb_losses, device=self.device)  # Mean of losses
        self.mean_l: torch.Tensor = torch.ones(self.nb_losses, device=self.device)  # Mean of loss ratios
        self.M2: torch.Tensor = torch.zeros(self.nb_losses, device=self.device)  # Sum of squares of differences from the current mean
        self.weights: torch.Tensor = torch.tensor([1.0, 1.0], device=self.device)  # Initial weights

        self.state: str = "train"  # Initialize state (assuming "train" by default)


    @torch.jit.export
    def compute_weights(self, loss_values: torch.Tensor) -> torch.Tensor:
        """
        Update the online statistics for each loss.

        Args:
            loss_values (torch.Tensor): Tensor of loss values with shape (nb_losses,).
                                        Should be on the same device as the class.

        Returns:
            torch.Tensor: Updated weights tensor.
        """
        if self.state == "eval":
            return self.weights

        # Detach and clone to prevent unwanted side effects
        L = loss_values.detach()

        # Update counts
        self.t += 1

        if self.t == 1:
            # Initialize mean_L and reset statistics for loss ratios
            self.mean_L = L.clone()
            # Compute M2 based on current weights and mean_l
            # self.M2 = (self.weights * self.mean_l).square()
            self.weights.fill_(0.5)
        else:
            # Update mean_L using Welford's algorithm
            delta = L - self.mean_L
            prev_mean_L = self.mean_L.clone()
            self.mean_L = self.mean_L + delta / self.t

            # Compute loss ratios ℓ_t = L_t / mean_{L_{t-1}}
            # Avoid division by zero by setting ratios to zero where prev_mean_L == 0
            l = torch.where(prev_mean_L != 0, L / prev_mean_L, torch.zeros_like(L))

            # Update loss_ratio_means and loss_ratio_M2 using Welford's algorithm
            d_ratio = l - self.mean_l
            self.mean_l = self.mean_l + d_ratio / self.t

            d2_ratio = l - self.mean_l
            self.M2 = self.M2 + d_ratio * d2_ratio

            # Compute standard deviation
            std = torch.sqrt(self.M2 / (self.t - 1))

            # Compute coefficient of variation c_l = σ_ℓ / mean_ℓ
            # Avoid division by zero by setting c_l to zero where mean_l == 0
            c_l = torch.where(
                self.mean_l != 0,
                std / self.mean_l,
                torch.zeros_like(self.mean_l)
            )

            # Normalize coefficients to get weights α_i
            z = torch.sum(c_l)

            if z > 0:
                self.weights = c_l / z
            else:
                # If sum is zero, assign equal weights
                self.weights.fill_(1.0 / self.nb_losses)

        return self.weights

    @torch.jit.export
    def reset(self):
        """
        Reset all statistics to their initial state.
        """
        with torch.no_grad():
            # break any accidental graph aliasing
            self.mean_L = self.mean_L.detach()
            self.mean_l = self.mean_l.detach()
            self.M2     = self.M2.detach()
    
            self.mean_L.zero_()
            self.mean_l.zero_()
            self.M2.zero_()
            self.t = 0