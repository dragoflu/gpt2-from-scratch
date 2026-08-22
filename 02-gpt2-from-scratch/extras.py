# Заготовки на будущее: свой Adam, RoPE, DPO/PPO лоссы. Пока нигде в пайплайнах не используются.
import torch
import torch.nn.functional as F
from torch import Tensor


def apply_rope_for_sing(m):
    B, S, D = m.size()
    pos = torch.arange(S).float()
    angles = torch.stack([pos / 10000 ** (2 * i / D) for i in range(D//2)]) # (step, pos)
    sin = torch.sin(angles.T) # (S, step)
    cos = torch.cos(angles.T) # (S, step)
    x1 = m[..., 0::2] # (B, S, step)
    x2 = m[..., 1::2] # (B, S, step)
    x1_new = x1 * cos - x2 * sin
    x2_new = x1 * sin + x2 * cos
    new = torch.stack((x1_new, x2_new), dim = -1)
    m = torch.flatten(new, start_dim = -2)
    return m


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen = policy_chosen_logps - ref_chosen_logps
    rejected = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen - rejected)
    loss = -F.logsigmoid(logits).mean()
    return loss


def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
             clip_ratio: float = 0.2) -> Tensor:
    old_logps = old_logps.detach()
    advantages = advantages.detach()
    r = torch.exp(new_logps - old_logps)
    unclipped = r * advantages
    clipped = torch.clamp(r, 1 - clip_ratio, 1 + clip_ratio) * advantages
    ppo_loss = -torch.min(clipped, unclipped).mean()
    return ppo_loss
