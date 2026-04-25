from __future__ import annotations

import math
from collections.abc import Iterable, MutableMapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = ["Nora", "nora", "get_nora_optimizer"]


LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
EPS = 1e-10


def _to_scalar(value: float | Tensor) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        return float(value.item())
    return float(value)


def _validate_betas(betas: tuple[float, float]) -> None:
    if len(betas) != 2:
        raise ValueError("betas must be a tuple of two values")
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"Invalid beta1 value: {beta1}")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"Invalid beta2 value: {beta2}")


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """和 Muon 类似，按矩阵形状对 Nora 更新幅度做一个可选调整。"""
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        if len(param_shape) < 2:
            return lr
        rows, cols = param_shape[-2], param_shape[-1]
        return lr * max(1.0, math.sqrt(rows / cols))
    if adjust_lr_fn == "none":
        return lr
    raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")


def _use_nora_for_param(param: Tensor, group: MutableMapping[str, Any]) -> bool:
    use_nora = group.get("use_nora", None)
    if use_nora is None:
        # 兼容旧代码里的参数组标记名。
        use_nora = group.get("is_rmnp", None)
    if use_nora is None:
        # 默认自动分流: 维度 >= nora_ndim 的参数走 Nora，其它走 Adam。
        return param.ndim >= group["nora_ndim"]
    return bool(use_nora)


def _get_fp32_param_and_grad(
    param: Tensor,
    grad: Tensor,
    state: MutableMapping[str, Any],
) -> tuple[Tensor, Tensor, bool]:
    """低精度训练时保留一份 fp32 master parameter，更新后再写回原参数。"""
    if param.dtype not in LOW_PRECISION_DTYPES:
        return param, grad, False

    master_param = state.get("fp32_param")
    if (
        master_param is None
        or master_param.shape != param.shape
        or master_param.device != param.device
    ):
        master_param = param.detach().float().clone()
        state["fp32_param"] = master_param
    elif master_param.dtype != torch.float32:
        master_param = master_param.float()
        state["fp32_param"] = master_param

    return master_param, grad.float(), True


class Nora(torch.optim.Optimizer):
    """Implements Nora with an Adam fallback for non-Nora parameters.

    Nora 适合更新矩阵类参数。默认情况下，`ndim >= 2` 的参数使用 Nora，
    bias、LayerNorm 权重等低维参数使用 Adam。也可以在参数组里显式设置
    `use_nora=True/False` 来控制分流。
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float | Tensor = 5e-3,
        adam_lr: float | Tensor = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        beta: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = EPS,
        nora_ndim: int = 2,
        adjust_lr_fn: str | None = "original",
        *,
        lr_rmnp: float | Tensor | None = None,
        lr_adam: float | Tensor | None = None,
    ) -> None:
        # 兼容旧代码里的命名: lr_rmnp/lr_adam。
        if lr_rmnp is not None:
            lr = lr_rmnp
        if lr_adam is not None:
            adam_lr = lr_adam

        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if isinstance(adam_lr, Tensor) and adam_lr.numel() != 1:
            raise ValueError("Tensor adam_lr must be 1-element")
        if _to_scalar(lr) < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if _to_scalar(adam_lr) < 0.0:
            raise ValueError(f"Adam learning rate should be >= 0 but is: {adam_lr}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay should be >= 0 but is: {weight_decay}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta should be in [0, 1) but is: {beta}")
        if eps <= 0.0:
            raise ValueError(f"eps should be > 0 but is: {eps}")
        if nora_ndim < 2:
            raise ValueError(f"nora_ndim should be >= 2 but is: {nora_ndim}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ["original", "none"]:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )
        _validate_betas(betas)

        defaults = {
            "lr": lr,
            "adam_lr": adam_lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "beta": beta,
            "nesterov": nesterov,
            "betas": betas,
            "eps": eps,
            "nora_ndim": nora_ndim,
            "adjust_lr_fn": adjust_lr_fn,
            "use_nora": None,
        }
        super().__init__(params, defaults)

    def _init_group(
        self,
        group: MutableMapping[str, Any],
        nora_params: list[Tensor],
        nora_grads: list[Tensor],
        nora_momentum_bufs: list[Tensor],
        nora_original_params: list[Tensor],
        adam_params: list[Tensor],
        adam_grads: list[Tensor],
        adam_exp_avgs: list[Tensor],
        adam_exp_avg_sqs: list[Tensor],
        adam_state_steps: list[int],
        adam_original_params: list[Tensor],
    ) -> bool:
        has_complex = False

        for param in group["params"]:
            if param.grad is None:
                continue
            if torch.is_complex(param):
                raise RuntimeError("Nora does not support complex parameters")
            if param.grad.is_sparse:
                raise RuntimeError("Nora does not support sparse gradients")

            state = self.state[param]
            grad = param.grad.detach()
            param_data, grad_data, uses_master_param = _get_fp32_param_and_grad(
                param, grad, state
            )

            if _use_nora_for_param(param, group):
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad_data, memory_format=torch.preserve_format
                    )
                elif (
                    state["momentum_buffer"].shape != grad_data.shape
                    or state["momentum_buffer"].device != grad_data.device
                ):
                    state["momentum_buffer"] = torch.zeros_like(
                        grad_data, memory_format=torch.preserve_format
                    )
                elif uses_master_param and state["momentum_buffer"].dtype != torch.float32:
                    state["momentum_buffer"] = state["momentum_buffer"].float()

                nora_params.append(param_data)
                nora_grads.append(grad_data)
                nora_momentum_bufs.append(state["momentum_buffer"])
                nora_original_params.append(param)
            else:
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        grad_data, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        grad_data, memory_format=torch.preserve_format
                    )
                    state["step"] = 0
                elif (
                    state["exp_avg"].shape != grad_data.shape
                    or state["exp_avg"].device != grad_data.device
                ):
                    state["exp_avg"] = torch.zeros_like(
                        grad_data, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        grad_data, memory_format=torch.preserve_format
                    )
                    state["step"] = 0
                elif uses_master_param and state["exp_avg"].dtype != torch.float32:
                    state["exp_avg"] = state["exp_avg"].float()
                    state["exp_avg_sq"] = state["exp_avg_sq"].float()

                step = state["step"]
                if isinstance(step, Tensor):
                    step = int(step.item())
                state["step"] = step + 1

                adam_params.append(param_data)
                adam_grads.append(grad_data)
                adam_exp_avgs.append(state["exp_avg"])
                adam_exp_avg_sqs.append(state["exp_avg_sq"])
                adam_state_steps.append(state["step"])
                adam_original_params.append(param)

        return has_complex

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            nora_params: list[Tensor] = []
            nora_grads: list[Tensor] = []
            nora_momentum_bufs: list[Tensor] = []
            nora_original_params: list[Tensor] = []
            adam_params: list[Tensor] = []
            adam_grads: list[Tensor] = []
            adam_exp_avgs: list[Tensor] = []
            adam_exp_avg_sqs: list[Tensor] = []
            adam_state_steps: list[int] = []
            adam_original_params: list[Tensor] = []

            has_complex = self._init_group(
                group,
                nora_params,
                nora_grads,
                nora_momentum_bufs,
                nora_original_params,
                adam_params,
                adam_grads,
                adam_exp_avgs,
                adam_exp_avg_sqs,
                adam_state_steps,
                adam_original_params,
            )

            if nora_params:
                nora(
                    nora_params,
                    nora_grads,
                    nora_momentum_bufs,
                    lr=group.get("lr_rmnp", group["lr"]),
                    weight_decay=group["weight_decay"],
                    momentum=group["momentum"],
                    beta=group["beta"],
                    nesterov=group["nesterov"],
                    eps=group["eps"],
                    adjust_lr_fn=group["adjust_lr_fn"],
                    has_complex=has_complex,
                )
                _copy_master_params_to_model(nora_params, nora_original_params)

            if adam_params:
                _single_tensor_adam(
                    adam_params,
                    adam_grads,
                    adam_exp_avgs,
                    adam_exp_avg_sqs,
                    adam_state_steps,
                    lr=group.get("lr_adam", group.get("adam_lr", group["lr"])),
                    weight_decay=group["weight_decay"],
                    betas=group["betas"],
                    eps=group["eps"],
                    has_complex=has_complex,
                )
                _copy_master_params_to_model(adam_params, adam_original_params)

        return loss


Nora.__doc__ = (
    r"""Implements Nora optimizer.

    Nora 对矩阵类参数执行归一化的正交行方向更新:

    1. `buf.lerp_(grad, 1 - beta)` 得到平滑梯度。`beta=0.95` 时，缓冲区保留
       95% 的历史信息，只吸收 5% 的当前梯度。
    2. 若 `nesterov=True`，`m_t = grad.lerp(buf, momentum)`，把当前梯度和平滑
       趋势融合，形成带 Nesterov 加速的更新方向；否则直接使用 `buf`。
    3. 用 `F.normalize` 对参数行向量和投影后的方向做 L2 归一化。
    4. 非 Nora 参数在同一个优化器内使用 Adam 更新。

    Args:
        params: iterable of parameters or dicts defining parameter groups.
        lr (float, Tensor, optional): Nora learning rate (default: 5e-3).
        adam_lr (float, Tensor, optional): Adam fallback learning rate (default: 1e-3).
        weight_decay (float, optional): decoupled weight decay (default: 0.0).
        momentum (float, optional): Nora Nesterov blending coefficient (default: 0.95).
        beta (float, optional): Nora smoothed-gradient EMA coefficient (default: 0.95).
        nesterov (bool, optional): enables Nesterov acceleration for Nora (default: True).
        betas (Tuple[float, float], optional): Adam coefficients (default: (0.9, 0.95)).
        eps (float, optional): term added for numerical stability (default: 1e-10).
        nora_ndim (int, optional): parameters with ndim >= nora_ndim use Nora when
            `use_nora` is not explicitly set (default: 2).
        adjust_lr_fn (str, optional): one of "original" and "none"; "original" applies
            Muon-style matrix-shape scaling to Nora updates (default: "original").

    Example:
        >>> optimizer = Nora(model.parameters(), lr=5e-3, adam_lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss = loss_fn(model(input), target)
        >>> loss.backward()
        >>> optimizer.step()

        >>> optimizer = get_nora_optimizer(model, lr=5e-3, adam_lr=1e-3)
    """
)


def _copy_master_params_to_model(
    params: list[Tensor],
    original_params: list[Tensor],
) -> None:
    for param_data, original_param in zip(params, original_params):
        if param_data is not original_param:
            original_param.copy_(param_data.to(dtype=original_param.dtype))


def _single_tensor_nora(
    params: list[Tensor],
    grads: list[Tensor],
    nora_momentum_bufs: list[Tensor],
    *,
    lr: float | Tensor,
    weight_decay: float,
    momentum: float,
    beta: float,
    nesterov: bool,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    if has_complex:
        raise RuntimeError("Nora does not support complex parameters")

    lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim < 2:
            raise ValueError("Nora parameter gradient must have ndim >= 2")

        buf = nora_momentum_bufs[i]

        # beta 控制平滑梯度的记忆力: beta 越大，越相信历史梯度。
        buf.lerp_(grad, 1 - beta)

        # nesterov=True 时，将当前梯度和平滑后的趋势再次融合，形成更有推力的方向。
        update = grad.lerp(buf, momentum) if nesterov else buf

        # PyTorch 的函数名是 F.normalize（小写）；这里是 Nora 的参数行归一化。
        theta_hat = F.normalize(param, p=2, dim=-1, eps=eps)

        # 去掉更新方向中沿参数行向量的径向分量，只保留正交切向分量。
        dot_product = torch.sum(update * theta_hat, dim=-1, keepdim=True)
        update = update - dot_product * theta_hat

        # Nora 的更新方向归一化，同样使用 F.normalize。
        update = F.normalize(update, p=2, dim=-1, eps=eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        if weight_decay != 0.0:
            param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


def _single_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[int],
    *,
    lr: float | Tensor,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    has_complex: bool,
) -> None:
    if has_complex:
        raise RuntimeError("Adam fallback in Nora does not support complex parameters")

    lr = _to_scalar(lr)
    beta1, beta2 = betas

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        if weight_decay != 0.0:
            param.mul_(1 - lr * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        denom = exp_avg_sq.sqrt().add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)


def nora(
    params: list[Tensor],
    grads: list[Tensor],
    nora_momentum_bufs: list[Tensor],
    *,
    foreach: bool | None = None,
    lr: float | Tensor,
    weight_decay: float,
    momentum: float,
    beta: float,
    nesterov: bool,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    r"""Functional API that performs Nora algorithm computation."""
    if foreach:
        raise RuntimeError("Foreach is not supported for Nora yet")

    _single_tensor_nora(
        params,
        grads,
        nora_momentum_bufs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        beta=beta,
        nesterov=nesterov,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
        has_complex=has_complex,
    )


def get_nora_optimizer(
    model: torch.nn.Module,
    lr: float | Tensor = 5e-3,
    adam_lr: float | Tensor = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    beta: float = 0.95,
    nesterov: bool = True,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = EPS,
    adjust_lr_fn: str | None = "original",
    *,
    nora_ndim: int = 2,
    exclude_from_nora: tuple[str, ...] = ("embed", "embedding", "lm_head"),
    lr_rmnp: float | Tensor | None = None,
    lr_adam: float | Tensor | None = None,
) -> Nora:
    """按名称和维度自动分组，返回一个可以整体优化模型参数的 Nora optimizer。

    默认规则:
    - `requires_grad=True`
    - `param.ndim >= nora_ndim`
    - 参数名不包含 embed / embedding / lm_head

    满足以上条件的参数走 Nora，其它参数走 Adam。
    """
    # 兼容旧代码里的命名: lr_rmnp/lr_adam。
    if lr_rmnp is not None:
        lr = lr_rmnp
    if lr_adam is not None:
        adam_lr = lr_adam

    nora_params: list[Tensor] = []
    adam_params: list[Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lowered_name = name.lower()
        should_exclude = any(key in lowered_name for key in exclude_from_nora)
        if param.ndim >= nora_ndim and not should_exclude:
            nora_params.append(param)
        else:
            adam_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if nora_params:
        param_groups.append(
            {
                "params": nora_params,
                "lr": lr,
                "adam_lr": adam_lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "beta": beta,
                "nesterov": nesterov,
                "betas": betas,
                "eps": eps,
                "nora_ndim": nora_ndim,
                "adjust_lr_fn": adjust_lr_fn,
                "use_nora": True,
            }
        )
    if adam_params:
        param_groups.append(
            {
                "params": adam_params,
                "lr": adam_lr,
                "adam_lr": adam_lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "beta": beta,
                "nesterov": nesterov,
                "betas": betas,
                "eps": eps,
                "nora_ndim": nora_ndim,
                "adjust_lr_fn": adjust_lr_fn,
                "use_nora": False,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters were found for Nora optimizer")

    return Nora(param_groups)
