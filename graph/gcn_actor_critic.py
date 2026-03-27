"""GCN-based actor-critic for lamp placement PPO（batch 版）。

配合 graph/env.py 使用，obs 是节点级特征 (N_interior, F)，只含房间内部节点。
动作空间：0..N_interior-1 放灯 + N_interior stop。

邻接矩阵由环境提供的 edge_index 构建，只包含内部节点间的边。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NEG_INF = -1e9


def build_normalized_adj_from_edges(n_nodes: int, edge_index: np.ndarray, device: torch.device) -> torch.Tensor:
    """从 edge_index (2, E) 构建归一化稠密邻接矩阵（含自环）。

    返回 (N, N) float32 dense tensor。
    归一化：D^{-1/2} A D^{-1/2}
    """
    adj = torch.zeros(n_nodes, n_nodes, device=device)
    src = edge_index[0]
    dst = edge_index[1]
    adj[src, dst] = 1.0
    # 自环
    adj.fill_diagonal_(1.0)

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    return adj


class BatchGCNLayer(nn.Module):
    """支持 batch 的图卷积层。

    forward(x, adj):
        x:   (B, N, in_features)  或  (N, in_features) 单样本
        adj: (N, N)
        返回: (B, N, out_features) 或 (N, out_features)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        if x.dim() == 2:
            return torch.mm(adj, support) + self.bias
        else:
            out = torch.bmm(adj.unsqueeze(0).expand(x.shape[0], -1, -1), support)
            return out + self.bias


class GCNActorCritic(nn.Module):
    """Batch GCN-based PPO actor-critic，配合 GraphRoomEnv 使用。

    obs: (N_interior, F)，只含房间内部节点。
    动作: 0..N_interior-1 放灯，N_interior stop。
    """

    def __init__(
        self,
        in_features: int = 6,
        hidden: int = 64,
        target_lamp_count: int | None = None,
    ) -> None:
        super().__init__()
        self.target_lamp_count = target_lamp_count

        self.gc1 = BatchGCNLayer(in_features, hidden)
        self.gc2 = BatchGCNLayer(hidden, hidden * 2)
        self.gc3 = BatchGCNLayer(hidden * 2, hidden)

        self.policy_node = nn.Linear(hidden, 1)
        self.policy_stop = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )

        self._adj: torch.Tensor | None = None

    def set_adj(self, n_nodes: int, edge_index: np.ndarray, device: torch.device) -> None:
        """从环境的 edge_index 构建并缓存邻接矩阵。只需调用一次。"""
        self._adj = build_normalized_adj_from_edges(n_nodes, edge_index, device)

    def _encode_batch(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """批量 GCN 编码。x: (B, N, F) -> (B, N, hidden)"""
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        return x

    def _build_logits_and_values(
        self, batch_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """批量前向。

        batch_obs: (B, N_interior, F)
        返回:
            logits: (B, N_interior+1)
            values: (B, 1)
        """
        assert self._adj is not None, "Call set_adj() before forward pass"
        B, N, F_dim = batch_obs.shape
        adj = self._adj

        node_emb = self._encode_batch(batch_obs, adj)  # (B, N, hidden)

        # 策略：每节点一个 logit
        spatial_logits = self.policy_node(node_emb).squeeze(-1)  # (B, N)

        # stop logit：全局平均池化（只含内部节点，无墙壁污染）
        global_feat = node_emb.mean(dim=1)  # (B, hidden)
        stop_logit = self.policy_stop(global_feat)  # (B, 1)

        # 价值
        values = self.value_head(global_feat)  # (B, 1)

        # 动作掩码
        # f2: 可放灯（placeable & ~placed），f3: 已放灯
        valid = batch_obs[:, :, 2] > 0.5       # (B, N)
        placed = batch_obs[:, :, 3] > 0.5      # (B, N)
        masked_spatial = spatial_logits.masked_fill(~valid, NEG_INF)

        # stop 合法性
        if self.target_lamp_count is not None:
            lamp_counts = placed.sum(dim=1)  # (B,)
            stop_ok = lamp_counts >= self.target_lamp_count
            stop_logit = stop_logit.masked_fill(~stop_ok.unsqueeze(1), NEG_INF)

        logits = torch.cat([masked_spatial, stop_logit], dim=1)  # (B, N+1)
        return logits, values

    @torch.no_grad()
    def act(
        self, obs_tensor: torch.Tensor, deterministic: bool = False
    ) -> dict[str, torch.Tensor]:
        """obs_tensor: (1, N_interior, F)"""
        logits, value = self._build_logits_and_values(obs_tensor)
        logits = logits.squeeze(0)
        value = value.squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax() if deterministic else dist.sample()
        return {"action": action, "log_prob": dist.log_prob(action), "value": value}

    def evaluate_actions(
        self, batch_obs: torch.Tensor, batch_actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """batch_obs: (B, N_interior, F), batch_actions: (B,)"""
        logits, values = self._build_logits_and_values(batch_obs)
        dist = torch.distributions.Categorical(logits=logits)
        return {
            "log_prob": dist.log_prob(batch_actions),
            "entropy": dist.entropy(),
            "value": values,
        }
