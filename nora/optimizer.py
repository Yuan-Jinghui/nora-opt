import math
import torch
import torch.nn.functional as F

class NORA(torch.optim.Optimizer):
    """
    Normalized Orthogonal Row Alignment (NORA) 优化器。
    
    该优化器针对大规模矩阵训练设计：
    - 对 2D 矩阵参数（如线性层权重）应用 Nora 算法：进行动量的行空间切线投影及行归一化。
    - 对 1D 或非矩阵参数（如 Bias, LayerNorm, Embeddings）回退使用标准的 AdamW 算法。
    """

    def __init__(
        self,
        param_groups,
        lr_rmnp=0.005,       # Nora 算法使用的学习率
        lr_adam=0.001,       # AdamW 分支使用的学习率
        beta=0.95,           # Nora 的动量系数
        weight_decay=0.0,    # 权重衰减
        betas=(0.9, 0.95),   # AdamW 分支使用的 betas
        eps=1e-10,           # 数值稳定性常数
    ):
        defaults = dict(
            lr_rmnp=lr_rmnp,
            lr_adam=lr_adam,
            beta=beta,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单一的优化步骤"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group.get("beta", 0.95)
            weight_decay = group.get("weight_decay", 0.0)
            betas = group.get("betas", (0.9, 0.95))
            eps = group.get("eps", 1e-10)
            
            # 标志位决定该参数组是否使用 Nora 核心算法
            is_rmnp = group.get("is_rmnp", True)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                param_state = self.state.setdefault(p, {})

                # =========================================================
                # NORA 算法分支 (应用于 2D 参数)
                # =========================================================
                if is_rmnp and grad.dim() >= 2:
                    # 初始化动量 buffer
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format
                        )
                    buf = param_state["momentum_buffer"]

                    # 1. 计算动量 (EMA): m_t = beta * m_{t-1} + (1 - beta) * g_t
                    buf.lerp_(grad, 1 - beta)

                    # 2. 行空间切线投影 (Row-wise Tangent Projection)
                    # 提前将权重按行做 L2 归一化，既避免除零风险，又让公式计算更简洁
                    theta_hat = F.normalize(p, p=2, dim=-1, eps=eps)
                    dot_product = torch.sum(buf * theta_hat, dim=-1, keepdim=True)
                    v = buf - dot_product * theta_hat # 等价于论文的 m_t^{r\perp}

                    # 3. 行归一化 (Row-wise Normalization)
                    v_hat = F.normalize(v, p=2, dim=-1, eps=eps) # d_t

                    # 4. 学习率缩放适配 (匹配 Muon 的 RMS 缩放尺度)
                    scale = max(1.0, math.sqrt(grad.size(-2) / grad.size(-1)))
                    update_direction = v_hat * scale

                    # 5. 解耦权重衰减与参数更新
                    if weight_decay > 0:
                        p.mul_(1 - lr * weight_decay)

                    p.add_(update_direction, alpha=-lr)

                # =========================================================
                # AdamW 算法分支 (应用于 1D 偏置、LayerNorm 等)
                # =========================================================
                else:
                    if "exp_avg" not in param_state:
                        param_state["exp_avg"] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format
                        )
                        param_state["exp_avg_sq"] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format
                        )
                        param_state["step"] = 0

                    exp_avg = param_state["exp_avg"]
                    exp_avg_sq = param_state["exp_avg_sq"]
                    param_state["step"] += 1

                    # 动量与二阶矩更新
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

                    # 偏差校正
                    bias_correction1 = 1 - betas[0] ** param_state["step"]
                    bias_correction2 = 1 - betas[1] ** param_state["step"]
                    
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom

                    # 解耦权重衰减与参数更新
                    if weight_decay > 0:
                        p.mul_(1 - step_size * weight_decay)

                    p.add_(adam_update, alpha=-step_size)

        return loss


def get_nora_optimizer(
    model,
    lr_rmnp=0.005,
    lr_adam=0.001,
    weight_decay=0.1,
    beta=0.95,
):
    """
    自动化构建 Nora 优化器的辅助函数。
    """
    rmnp_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            # 区分矩阵权重(Nora)和向量权重(AdamW)
            if param.ndim >= 2 and "embed" not in name and "lm_head" not in name:
                rmnp_params.append(param)
            else:
                adam_params.append(param)

    param_groups = [
        # Nora 分组
        dict(
            params=rmnp_params,
            lr=lr_rmnp,              
            lr_rmnp=lr_rmnp,         
            lr_adam=lr_adam,         
            weight_decay=weight_decay,
            beta=beta,
            is_rmnp=True,            
        ),
        # AdamW 分组
        dict(
            params=adam_params,
            lr=lr_adam,              
            lr_rmnp=lr_rmnp,
            lr_adam=lr_adam,
            weight_decay=weight_decay,
            beta=beta,
            is_rmnp=False,           
        ),
    ]
    
    return NORA(param_groups)
