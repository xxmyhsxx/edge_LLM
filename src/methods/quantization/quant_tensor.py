# 实现tensor的量化
# 全张量的量化(仅一组zero_point和scale)
# 在全张量的基础上:逐通道量化以及逐channel的量化
# 在此基础上实现分组量化------>对于同样的张量，敏感的部分应该更精细化的量化
import torch

@torch.no_grad()
def real_quantize_tensor(w, n_bits=8, zp=True):
    """
    执行 Per-Tensor 量化。
    返回: (量化后的张量, scale, zero_point)
    """
    if zp:
        # 非对称量化 (Asymmetric) -> uint8
        q_max = 2 ** n_bits - 1
        q_min = 0
        dtype = torch.uint8
        w_min = w.min()
        w_max = w.max()
        scales = (w_max - w_min).clamp_(min=1e-5) / q_max
        z_p = torch.round(-w_min / scales).clamp_(q_min, q_max)
        w_q = torch.round(w / scales + z_p).clamp_(q_min, q_max).to(dtype)
        return w_q, scales, z_p
    else:
        # 对称量化 (Symmetric) -> int8
        q_max = 2 ** (n_bits - 1) - 1
        q_min = -2 ** (n_bits - 1)
        dtype = torch.int8
        scales = w.abs().max()
        scales.clamp_(min=1e-5).div_(q_max)
        z_p = torch.tensor(0, dtype=torch.int32)  # 对称量化的 zero_point 为 0
        w_q = torch.round(w / scales).clamp_(q_min, q_max).to(dtype)
        return w_q, scales, z_p


@torch.no_grad()
def real_quantize_channel(w, n_bits=8, zp=True, dim=-1):
    """
    执行 Per-Channel 量化（适用于 weight 和 activation）。
    dim: 量化轴 (例如, weight: 0, activation: -1)
    返回: (量化后的张量, scale, zero_point)
    """
    if zp:
        # 非对称量化 (Asymmetric) -> uint8
        q_max = 2 ** n_bits - 1
        q_min = 0
        dtype = torch.uint8
        w_min = w.min(dim=dim, keepdim=True)[0]
        w_max = w.max(dim=dim, keepdim=True)[0]
        scales = (w_max - w_min).clamp_(min=1e-5) / q_max
        z_p = torch.round(-w_min / scales).clamp_(q_min, q_max)

        w_q = torch.round(w / scales + z_p).clamp_(q_min, q_max).to(dtype)
        return w_q, scales, z_p
    else:
        # 对称量化 (Symmetric) -> int8
        q_max = 2 ** (n_bits - 1) - 1
        q_min = -2 ** (n_bits - 1)
        dtype = torch.int8
        scales = w.abs().max(dim=dim, keepdim=True)[0]
        scales.clamp_(min=1e-5).div_(q_max)
        z_p = torch.zeros_like(scales, dtype=torch.int32)
        # z_p = torch.tensor(0, dtype=torch.int32)  # 对称量化的 zero_point 为 0
        w_q = torch.round(w / scales).clamp_(q_min, q_max).to(dtype)
        return w_q, scales.to(torch.bfloat16).cuda(), z_p.to(torch.bfloat16).cuda()


# ---------------------------------------------------------------------------
# 2. 核心反量化函数 (Core Dequantization Function)
# ---------------------------------------------------------------------------
@torch.no_grad()
def dequant_tensor(w_q, scale, zero_point, dtype=torch.bfloat16):
    """
    通用的反量化函数。
    """
    # 确保 scale 和 zero_point 至少是 1D，以便广播
    if scale.dim() == 0:
        scale = scale.view(1)
    if zero_point.dim() == 0:
        zero_point = zero_point.view(1)

    return (w_q.to(dtype) - zero_point) * scale


# ---------------------------------------------------------------------------
# 3. "假"量化封装 ( "Fake" Quantization Wrappers)
# ---------------------------------------------------------------------------
# 重写你原有的"假"量化函数，使其调用"真"量化和反量化，消除冗余。

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, zero_point=False, n_bits=8):
    # (out_features, in_features) -> dim=0
    w_q, scales, z_p = real_quantize_channel(w, n_bits, zero_point, dim=0)
    return dequant_tensor(w_q, scales, z_p, dtype=w.dtype)

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, zero_point=False, n_bits=8):
    w_q, scales, z_p = real_quantize_tensor(w, n_bits, zero_point)
    return dequant_tensor(w_q, scales, z_p, dtype=w.dtype)

@torch.no_grad()
def quantize_activation_per_token_absmax(t, zero_point=False, n_bits=8):
    t_q, scales, z_p = real_quantize_channel(t, n_bits, zero_point, dim=-1)
    return dequant_tensor(t_q, scales, z_p, dtype=t.dtype)

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, zero_point=False, n_bits=8):
    t_q, scales, z_p = real_quantize_tensor(t, n_bits, zero_point)
    return dequant_tensor(t_q, scales, z_p, dtype=t.dtype)


# ---------------------------------------------------------------------------
# 4. 分组量化 (Group Quantization)
# ---------------------------------------------------------------------------
@torch.no_grad()
def quantize_per_group_tensor(
        w, n_bits=8, zero_point=True, q_group_size=128, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        # 确保可以整除
        assert org_w_shape[-1] % q_group_size == 0
        # 将最后一个维度拆分为 (N_groups, group_size)
        # (..., N) -> (..., N_groups, group_size)
        w = w.reshape(-1, q_group_size)
    else:
        # q_group_size <= 0 意味着 per-channel (dim=0)
        # (Out, In) -> (Out, In)
        # 我们假设输入已经是 (N_groups, group_size) 的形式
        assert w.dim() == 2
        # w shape is now (Total_Groups, group_size)

    if zero_point:
        # 非对称
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        # 对称
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = 0  # zero_point 始终为 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # 量化
    w_q = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    # 反量化 ("假"量化)
    w_deq = (w_q - zeros) * scales
    assert torch.isnan(w_deq).sum() == 0
    # 恢复原始形状
    w_deq = w_deq.reshape(org_w_shape)
    if get_scale_zp:
        # 确保 scale/zp 形状匹配 (Total_Groups, 1)
        return w_deq, scales.view(-1, 1), zeros.view(-1, 1)
    else:
        return w_deq


# ---------------------------------------------------------------------------
# 5. 自定义非线性量化 (Custom Non-Linear Quantization)
# ---------------------------------------------------------------------------

@torch.no_grad()
def correct_quantize(t, n_bits=8, shift=0.0, scale_factor=0.4):
    # 这是一个 per-token (dim=-1) 的"假"量化
    t_shape = t.shape

    # 修复：确保 t 至少是二维的，以便 .abs().max(dim=-1) 工作
    # (B, S, H) -> (B*S, H)
    t_reshaped = t.reshape(-1, t_shape[-1])

    # 应用位移和幂缩放
    shifted_t = t_reshaped - shift
    scaled_t = torch.sign(shifted_t) * (torch.abs(shifted_t) ** scale_factor)

    # Per-token (dim=-1) 计算 scales
    scales = scaled_t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1  # 对称量化

    scales = scales.clamp(min=1e-5).div(q_max)

    # 量化：缩放、四舍五入
    w_q = torch.round(scaled_t.div(scales))

    # 反量化 (模拟)
    scaled_t_deq = w_q.mul(scales)

    # 反量化：逆幂运算并恢复位移
    dequantized_t = torch.sign(scaled_t_deq) * (torch.abs(scaled_t_deq) ** (1 / scale_factor)) + shift

    # 恢复原始形状
    return dequantized_t.view(t_shape)


@torch.no_grad()
def search_scale(w, n_bits):
    scale_factors = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_error = float('inf')
    best_scale = 1.0
    for scale in scale_factors:
        quantized = correct_quantize(w.clone(), n_bits, scale_factor=scale)
        error = torch.mean(torch.abs(w - quantized))
        if error < best_error:
            best_error = error
            best_scale = scale
    print(f"Best scale found: {best_scale} with error: {best_error}")
    return best_scale
