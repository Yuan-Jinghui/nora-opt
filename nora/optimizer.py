# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Nora optimizer with automatic AdamW fallback for 1D params."""

import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["Nora"]

class Nora(Optimizer):
    r"""Implements the Nora (Normalized Orthogonal Row Alignment) algorithm.
    
    Automatically applies Nora to 2D parameters (e.g., linear layers) and falls back 
    to AdamW for 1D or >2D parameters (e.g., biases, normalization layers, embeddings).

    Nora updates 2D parameters by explicitly projecting the momentum onto the tangent 
    space of the weights and performing a row-wise normalization.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. beta1 acts as the momentum 
            coefficient \beta for Nora. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0.0)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                if torch.is_complex(p):
                    raise RuntimeError("Nora does not support complex parameters")
                if p.grad.is_sparse:
                    raise RuntimeError("Nora does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    if p.ndim == 2:
                        # Nora state: only requires a momentum buffer
                        state["momentum_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        # AdamW state: requires exp_avg and exp_avg_sq
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                state["step"] += 1

                if p.ndim == 2:
                    # =========================================================
                    # NORA ALGORITHM (For 2D Matrix Parameters)
                    # =========================================================
                    buf = state["momentum_buffer"]
                    
                    # 1. Update momentum: m_t = \beta * m_{t-1} + (1 - \beta) * g_t
                    buf.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    
                    # 2. Row-wise tangent projection: m^{r\perp}
                    # P_{i:} = m_{i:} - (<m_{i:}, w_{i:}> / ||w_{i:}||_2^2) * w_{i:}
                    dot_prod = torch.sum(buf * p, dim=1, keepdim=True)
                    w_norm_sq = torch.sum(p * p, dim=1, keepdim=True)
                    m_proj = buf - (dot_prod / w_norm_sq.clamp(min=eps)) * p
                    
                    # 3. Row-wise normalization: d_t = m^{r\perp} / ||m^{r\perp}||_2
                    m_proj_norm = torch.linalg.vector_norm(m_proj, dim=1, keepdim=True)
                    d_t = m_proj / m_proj_norm.clamp(min=eps)
                    
                    # 4. Decoupled weight decay and parameter update
                    # w_{t+1} = w_t - \eta_t * (d_t + \lambda * w_t)
                    if weight_decay > 0.0:
                        p.mul_(1.0 - lr * weight_decay)
                        
                    p.add_(d_t, alpha=-lr)
                    
                else:
                    # =========================================================
                    # ADAMW ALGORITHM (For 1D / >2D Parameters)
                    # =========================================================
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step = state["step"]

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    step_size = lr / bias_correction1

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    # Decoupled weight decay
                    if weight_decay > 0.0:
                        p.mul_(1.0 - lr * weight_decay)

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
