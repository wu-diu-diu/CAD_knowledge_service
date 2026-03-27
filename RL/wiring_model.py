"""
布线RL策略网络

架构：CNN全局特征提取 + 每个灯具位置的局部特征 → 对每个候选灯具打分

与灯具布局RL的区别：
  - 动作空间是"选择N个灯具之一"，而非"选择H×W个格子之一"
  - Policy head对每个候选灯具独立打分，softmax得到选择概率
  - 动作掩码：已连接的灯具被屏蔽
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

NEG_INF = -1e9


def _group_count(num_channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        g = _group_count(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class WiringEncoder(nn.Module):
    """
    轻量CNN编码器，提取房间布线状态的全局特征和中间层特征图。

    输入：[B, 6, H, W]
    输出：
      global_feat: [B, 128]  全局特征（用于value head和policy全局上下文）
      feat_map: [B, 64, H/2, W/2]  中间特征图（用于提取每个灯具的局部特征）
    """

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.stage1 = ConvBlock(in_channels, 32)   # [B, 32, H, W]
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = ConvBlock(32, 64)             # [B, 64, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = ConvBlock(64, 128)            # [B, 128, H/4, W/4]
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [B, 128, 1, 1]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        s1 = self.stage1(x)
        s2 = self.stage2(self.pool1(s1))
        s3 = self.stage3(self.pool2(s2))
        global_feat = self.global_pool(s3).flatten(1)  # [B, 128]
        return global_feat, s2  # s2: [B, 64, H/2, W/2]


class WiringPolicyNet(nn.Module):
    """
    布线PPO策略网络。

    对每个候选灯具：
      1. 从 feat_map 中双线性插值提取该灯具位置的局部特征 [B, 64]
      2. 拼接全局特征 [B, 128] → [B, 192]
      3. MLP打分 → 标量 logit

    最终输出：[B, N] 的 logits（N = 灯具数量），已应用动作掩码。
    Value head：全局特征 → 标量。

    注意：N 在不同房间中不同（2-9），所以网络对每个灯具独立打分，
    而不是固定输出维度。
    """

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.encoder = WiringEncoder(in_channels)

        # 每个灯具的打分头：局部特征(64) + 全局特征(128) → 1
        self.node_scorer = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        obs: Tensor,
        lamp_coords: Tensor,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            obs: [B, 6, H, W] 观察张量
            lamp_coords: [B, N, 2] 每个灯具在原始房间坐标系中的位置 (row, col)
                         需要映射到填充后的坐标，再归一化到 [-1, 1] 用于 grid_sample
            action_mask: [B, N] bool，True 表示该灯具可选（未连接）

        Returns:
            logits: [B, N] 已掩码的动作 logits
            value: [B, 1] 状态价值
        """
        B, N, _ = lamp_coords.shape
        H, W = obs.shape[2], obs.shape[3]

        global_feat, feat_map = self.encoder(obs)  # [B,128], [B,64,H/2,W/2]

        # 从 feat_map 中提取每个灯具位置的局部特征
        # lamp_coords: [B, N, 2] (row, col) in padded space
        # grid_sample 需要 (x, y) = (col, row)，归一化到 [-1, 1]
        fh, fw = feat_map.shape[2], feat_map.shape[3]

        # 将灯具坐标从 padded obs 空间映射到 feat_map 空间（H/2, W/2）
        # lamp_coords 已经是 padded 坐标
        lamp_r = lamp_coords[:, :, 0].float()  # [B, N]
        lamp_c = lamp_coords[:, :, 1].float()  # [B, N]

        # 归一化到 [-1, 1]（feat_map 尺寸是 H/2, W/2）
        norm_x = (lamp_c / (W / 2)) * 2.0 - 1.0  # col → x
        norm_y = (lamp_r / (H / 2)) * 2.0 - 1.0  # row → y
        grid = torch.stack([norm_x, norm_y], dim=-1)  # [B, N, 2]
        grid = grid.unsqueeze(2)  # [B, N, 1, 2]

        # grid_sample: [B, 64, N, 1] → [B, 64, N]
        node_feats = torch.nn.functional.grid_sample(
            feat_map, grid, mode="bilinear", align_corners=True
        ).squeeze(-1)  # [B, 64, N]
        node_feats = node_feats.permute(0, 2, 1)  # [B, N, 64]

        # 拼接全局特征
        global_expanded = global_feat.unsqueeze(1).expand(B, N, -1)  # [B, N, 128]
        combined = torch.cat([node_feats, global_expanded], dim=-1)  # [B, N, 192]

        # 打分
        logits = self.node_scorer(combined).squeeze(-1)  # [B, N]

        # 应用动作掩码
        logits = logits.masked_fill(~action_mask, NEG_INF)

        value = self.value_head(global_feat)  # [B, 1]
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        lamp_coords: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> dict[str, Tensor]:
        """采样或贪婪选择一个动作。"""
        logits, value = self(obs, lamp_coords, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "logits": logits,
        }

    def evaluate_actions(
        self,
        obs: Tensor,
        lamp_coords: Tensor,
        action_mask: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor]:
        """计算PPO训练所需的量。"""
        logits, value = self(obs, lamp_coords, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "logits": logits,
        }


def build_lamp_coords_tensor(
    lamp_positions: list[tuple[int, int]],
    row_offset: int,
    col_offset: int,
    device: torch.device,
) -> Tensor:
    """
    将灯具的原始房间坐标转换为填充后的坐标，返回 [1, N, 2] tensor。

    Args:
        lamp_positions: 原始房间坐标列表 [(r, c), ...]
        row_offset: 填充行偏移
        col_offset: 填充列偏移
        device: 目标设备
    """
    coords = [
        [r + row_offset, c + col_offset]
        for r, c in lamp_positions
    ]
    return torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, 2]
