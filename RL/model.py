from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


NEG_INF = -1e9


def _group_count(num_channels: int) -> int:
    """Choose a valid GroupNorm group count for the given channel width."""
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Two-layer conv block used in the shared encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample, concatenate skip features, then fuse with one conv layer."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


@dataclass
class EncoderFeatures:
    """Shared encoder outputs used by both actor and critic decoders."""

    stage1: Tensor  # [B, 32, H, W]
    stage2: Tensor  # [B, 64, H/2, W/2]
    stage3: Tensor  # [B, 128, H/4, W/4]
    stage4: Tensor  # [B, 256, H/8, W/8]


class SharedEncoder(nn.Module):
    """
    Shared U-Net encoder for the PPO actor-critic.

    Input shape:
        [B, 6, H, W] where H and W are padded_size (e.g., 32, 48, 64)
        channels:
            0: placeable_mask  — 当前仍可放灯的格子
            1: placed_lamps    — 已放灯位置
            2: switch_mask     — 开关位置
            3: lamp_progress   — 已放灯数/目标灯数，全图广播标量平面（动态）
                                 关键：让网络感知"当前第几步"，
                                 推断"剩余灯应放在哪个区域互补"
            4: room_mask       — 房间内部区域（墙外为0），辅助空间边界感知
            5: target_lamp_count — 目标灯具数量，全图广播标量平面（固定，条件输入）
                                   让模型知道"总共要放多少灯"，从而选择正确的布局策略

    Output:
        EncoderFeatures with skip tensors for the policy decoder and bottleneck
        features for both policy/value heads.
    """

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.stage1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.stage4 = ConvBlock(128, 256)

    def forward(self, x: Tensor) -> EncoderFeatures:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got shape={tuple(x.shape)}")

        s1 = self.stage1(x)
        s2 = self.stage2(self.pool1(s1))
        s3 = self.stage3(self.pool2(s2))
        s4 = self.stage4(self.pool3(s3))
        return EncoderFeatures(stage1=s1, stage2=s2, stage3=s3, stage4=s4)


class PolicyDecoder(nn.Module):
    """
    Actor decoder.

    Produces:
        - H*W spatial logits for H×W lamp-placement actions (e.g., 32×32=1024 or 48×48=2304)
        - 1 stop-action logit
        - concatenated logits of shape [B, H*W+1]

    Invalid actions are masked to a very negative value before sampling.
    """

    def __init__(self, target_lamp_count: int | None = None) -> None:
        super().__init__()
        self.target_lamp_count = target_lamp_count
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.spatial_head = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.stop_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    @staticmethod
    def build_action_mask(obs: Tensor, mask_existing_lamps: bool = True) -> Tensor:
        """
        Build a valid-action mask from the observation tensor.

        Valid placement cells satisfy:
            - placeable mask == 1
            - optionally placed lamp mask == 0

        Returns:
            Boolean mask of shape [B, H*W], where True means action is valid.
        """

        if obs.shape[1] < 2:
            raise ValueError("Observation must contain at least 2 channels: placeable (ch0), lamps (ch1).")

        placeable_mask = obs[:, 0] > 0.5
        placed_lamps = obs[:, 1] > 0.5

        valid = placeable_mask
        if mask_existing_lamps:
            valid = valid & ~placed_lamps
        return valid.view(obs.shape[0], -1)

    def build_stop_mask(self, obs: Tensor) -> Tensor:
        """
        Decide whether the stop action is currently legal.

        When `target_lamp_count` is configured, stop is only allowed once the
        observation already contains at least that many placed lamps.
        """
        if self.target_lamp_count is None:
            return torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)

        placed_lamps = obs[:, 1] > 0.5
        lamp_count = placed_lamps.view(obs.shape[0], -1).sum(dim=1)
        return lamp_count >= int(self.target_lamp_count)

    def forward(self, features: EncoderFeatures, obs: Tensor) -> Tensor:
        x = self.up3(features.stage4, features.stage3)
        x = self.up2(x, features.stage2)
        x = self.up1(x, features.stage1)

        spatial_logits = self.spatial_head(x).flatten(start_dim=1)
        stop_logit = self.stop_head(features.stage4)

        valid_mask = self.build_action_mask(obs)
        masked_spatial_logits = spatial_logits.masked_fill(~valid_mask, NEG_INF)
        stop_mask = self.build_stop_mask(obs).unsqueeze(1)
        masked_stop_logit = stop_logit.masked_fill(~stop_mask, NEG_INF)
        return torch.cat([masked_spatial_logits, masked_stop_logit], dim=1)


class ValueDecoder(nn.Module):
    """
    Critic decoder.

    Multi-scale global pooling is applied to stage2/stage3/stage4 features, then
    a small MLP regresses the scalar state value.
    """

    def __init__(self) -> None:
        super().__init__()
        pooled_dim = 64 + 128 + 256
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, features: EncoderFeatures) -> Tensor:
        p2 = torch.flatten(nn.functional.adaptive_avg_pool2d(features.stage2, 1), start_dim=1)
        p3 = torch.flatten(nn.functional.adaptive_avg_pool2d(features.stage3, 1), start_dim=1)
        p4 = torch.flatten(nn.functional.adaptive_avg_pool2d(features.stage4, 1), start_dim=1)
        fused = torch.cat([p2, p3, p4], dim=1)
        return self.head(fused)


class LightingActorCritic(nn.Module):
    """
    PPO actor-critic for lamp placement.

    Forward:
        obs -> shared encoder -> policy logits + scalar value
    """

    def __init__(self, in_channels: int = 6, target_lamp_count: int | None = None) -> None:
        super().__init__()
        self.encoder = SharedEncoder(in_channels=in_channels)
        self.policy_decoder = PolicyDecoder(target_lamp_count=target_lamp_count)
        self.value_decoder = ValueDecoder()

    def encode(self, obs: Tensor) -> EncoderFeatures:
        return self.encoder(obs)

    def policy_logits(self, obs: Tensor) -> Tensor:
        features = self.encode(obs)
        return self.policy_decoder(features, obs)

    def value(self, obs: Tensor) -> Tensor:
        features = self.encode(obs)
        return self.value_decoder(features)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encode(obs)
        logits = self.policy_decoder(features, obs)
        values = self.value_decoder(features)
        return logits, values

    def action_distribution(self, obs: Tensor) -> torch.distributions.Categorical:
        logits = self.policy_logits(obs)
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> dict[str, Tensor]:
        """
        Sample or greedily choose one action for each state in the batch.

        Returns:
            action: [B]
            log_prob: [B]
            value: [B, 1]
            logits: [B, 1025]
        """

        logits, values = self(obs)  ## obs经过共享编码器和两个解码器，得到动作的logits和状态值。logits的形状是[B, H*W+1]，其中前H*W个元素对应H×W网格的放置动作（如32×32=1024或48×48=2304），最后一个元素对应停止动作。values的形状是[B, 1]，表示每个状态的估计值。
        dist = torch.distributions.Categorical(logits=logits)  ## dist是一个Categorical分布对象，使用logits参数来定义离散动作空间的概率分布。这个分布对象可以用来采样动作或者计算动作的对数概率。
        if deterministic:  ## 如果是确定性策略，选择概率最大的动作
            action = torch.argmax(logits, dim=1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)  ##action是一个序号，表示1025中的第action个动作被选中，log_prob是对第action个动作对应的概率求log，这里的log是自然对数
        return {
            "action": action,
            "log_prob": log_prob,
            "value": values,
            "logits": logits,
        }

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """
        Compute PPO training quantities for a batch of states and chosen actions.
        """

        logits, values = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": values,
            "logits": logits,
        }


def _demo() -> None:
    """Quick shape smoke test for local debugging."""
    padded_size = 48  # Test with 48x48 instead of 32x32
    model = LightingActorCritic(target_lamp_count=4)  # in_channels=6 by default
    obs = torch.zeros(2, 6, padded_size, padded_size)  # 6 channels now
    obs[:, 0] = 1.0  # ch0: whole room placeable
    obs[:, 4] = 1.0  # ch4: whole room interior
    obs[:, 5] = 0.2  # ch5: target_lamp_count normalized (e.g., 4/20)
    logits, values = model(obs)
    print(f"Input shape: {tuple(obs.shape)}")
    print(f"Policy logits shape: {tuple(logits.shape)} (expected: [2, {padded_size*padded_size + 1}])")
    print(f"Value shape: {tuple(values.shape)}")
    assert logits.shape == (2, padded_size * padded_size + 1), f"Expected logits shape [2, {padded_size*padded_size + 1}], got {logits.shape}"
    print("✓ Model works with 6 channels and dynamic padded_size!")


if __name__ == "__main__":
    _demo()
