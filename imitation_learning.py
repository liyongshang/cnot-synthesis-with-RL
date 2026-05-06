"""
模仿学习（行为克隆）最小测试：样本为 (current_parity, expert_edge)。

约定
----
- **target_parity** 恒为 N×N 单位阵 I（GF(2) 下目标为各行标准基 / 单位线性变换）。
- **current_parity**：输入网络的 parity 观测，形状 [N, N]。默认约定为 **最近一次施加 CNOT 之后**
  的矩阵（见 `make_deterministic_label_samples` 的 `observation`）；亦可设为施加前（MDP 常用）。
- **edge**：专家在该状态下选择的有向 CNOT，对应 `edge_index` 中的列下标 e ∈ {0, …, E−1}；
  执行规则：P[dst] ← P[dst] ⊕ P[src]。

节点输入说明：仅用「目标行」编码会使 T=I 时节点特征与状态无关，因此将 **每行** 拼接为
[I[i] ‖ P[i]]（长度 2N），再线性映射到 node_dim；边特征仍用 `ParityEdgeFeatEncoder(I, edge_index, P)`。

数据说明
----
- **5 比特双向环**：邻接 i—(i+1) mod 5，每条物理边对应两条有向边 (i→j) 与 (j→i)。
- **`make_deterministic_label_samples`**：默认在 **每一步施加 CNOT 之后** 记录
  `(parity_after, 该步边下标)`。若在 **施加前** 采样（`pre_action`）再按矩阵去重，不同矩阵确实可能
  只剩约 **11** 种；这与「施加后」能得到更多不同矩阵并不矛盾。

用法：运行 `train_imitation_demo()`：在环上用默认样本构造做一轮小规模 BC，检验是否能拟合。
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import CNOTPolicyNet, ParityEdgeFeatEncoder


def _parity_matrix_key(parity: torch.Tensor) -> tuple[int, ...]:
    """GF(2) parity 矩阵的稳定键（用于去重），形状 [N, N]。"""
    return tuple(int(x) for x in parity.detach().cpu().reshape(-1).tolist())


def dataset_parity_stats(samples: list[tuple[torch.Tensor, int]]) -> tuple[int, int]:
    """返回 (样本条数, 不同 parity 矩阵个数)。"""
    uniq: set[tuple[int, ...]] = set()
    for p, _ in samples:
        uniq.add(_parity_matrix_key(p))
    return len(samples), len(uniq)


def label_ambiguity_stats(samples: list[tuple[torch.Tensor, int]]) -> tuple[int, int]:
    """
    按 parity 矩阵分组看专家边是否唯一。

    返回 (不同矩阵个数, 其中「对应多于一种边标签」的矩阵个数)。
    后者 >0 时，仅用 P 作输入无法对所有样本同时分类正确。
    """
    groups: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for p, e in samples:
        groups[_parity_matrix_key(p)].add(int(e))
    n_mat = len(groups)
    n_conflict = sum(1 for es in groups.values() if len(es) > 1)
    return n_mat, n_conflict


def gf2_apply_cnot(parity: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    """单步 CNOT(src→dst)：P[dst] = P[dst] XOR P[src]。"""
    out = parity.clone()
    out[dst] = (out[src] + out[dst]) % 2
    return out


def bidirectional_ring_edge_index(
    n_qubits: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    双向环状连接：顶点 0..N-1，无向边连接 i 与 (i+1) mod N；
    每条无向边展开为两条有向边 (i→j) 与 (j→i)。

    返回 edge_index，形状 [2, E]，E = 2 * N。
    """
    dev = device or torch.device("cpu")
    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        src_list.extend([i, j])
        dst_list.extend([j, i])
    return torch.tensor([src_list, dst_list], dtype=torch.long, device=dev)


def edge_pair_to_index(edge_index: torch.Tensor, src: int, dst: int) -> int:
    """将 (src, dst) 映射到唯一列下标；不存在则抛错。"""
    src_r = edge_index[0]
    dst_r = edge_index[1]
    mask = (src_r == src) & (dst_r == dst)
    idx = mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        raise ValueError(f"edge ({src}->{dst}) not in edge_index")
    return int(idx[0, 0].item())


class StateTargetRowEncoder(nn.Module):
    """
    每行拼接 [target_row ‖ current_row]，再映射到 node_dim。
    当 target 恒为 I 时，target_row[i] 即为第 i 个标准基方向。
    """

    def __init__(self, n_qubits: int, node_dim: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.proj = nn.Linear(2 * n_qubits, node_dim)

    def forward(self, target_parity: torch.Tensor, current_parity: torch.Tensor) -> torch.Tensor:
        if target_parity.shape != (self.n_qubits, self.n_qubits):
            raise ValueError(f"target_parity 期望 [{self.n_qubits},{self.n_qubits}]")
        if current_parity.shape != target_parity.shape:
            raise ValueError("current_parity 形状须与 target_parity 一致")
        t = target_parity.to(dtype=torch.float32, device=self.proj.weight.device)
        p = current_parity.to(dtype=torch.float32, device=self.proj.weight.device)
        rows = torch.cat([t, p], dim=-1)
        return self.proj(rows)


class ImitationPolicy(nn.Module):
    """
    固定图上的 BC 策略：target=I，输入 current_parity，输出各边 logits [E, 1]。
    """

    def __init__(
        self,
        n_qubits: int,
        edge_index: torch.Tensor,
        node_dim: int = 24,
        edge_dim: int = 12,
        hidden_dim: int = 48,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.register_buffer("edge_index", edge_index.clone().long())
        self.register_buffer("target_parity", torch.eye(n_qubits, dtype=torch.float32))

        self.node_enc = StateTargetRowEncoder(n_qubits, node_dim)
        self.edge_enc = ParityEdgeFeatEncoder(n_qubits, edge_dim, raw_dim=8)
        self.policy = CNOTPolicyNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_value_head=False,
        )

    def forward(self, current_parity: torch.Tensor) -> torch.Tensor:
        node_feat = self.node_enc(self.target_parity, current_parity)
        edge_feat = self.edge_enc(self.target_parity, self.edge_index, current_parity)
        return self.policy(node_feat, self.edge_index, edge_feat)


def _apply_edge_index(parity: torch.Tensor, edge_index: torch.Tensor, edge_idx: int) -> torch.Tensor:
    """对 parity 施加 edge_index 第 edge_idx 列对应的有向 CNOT。"""
    i = int(edge_index[0, edge_idx].item())
    j = int(edge_index[1, edge_idx].item())
    return gf2_apply_cnot(parity, i, j)


def make_deterministic_label_samples(
    n_qubits: int,
    edge_index: torch.Tensor,
    walk_length: int = 1,
    *,
    observation: Literal["pre_action", "post_action"] = "post_action",
    enumerate_all_paths: bool = True,
    num_random_trajectories: int = 512,
    generator: torch.Generator | None = None,
    dedupe_mode: Literal["none", "parity_matrix", "parity_and_edge"] = "parity_and_edge",
) -> list[tuple[torch.Tensor, int]]:
    """
    在单位阵 I 上沿一条轨迹连续施加 `walk_length` 个 CNOT，每一步记录一条样本：

        (parity_observation, expert_edge_idx)

    `expert_edge_idx` 均为 **该步实际施加的** 边在 `edge_index` 中的列下标。

    **`observation`**（默认 `post_action`，推荐）：

    - **`post_action`**：观测为 **本步 CNOT 施加之后** 的 parity（即「刚执行完 label 这条边」的状态）。
      样本语义：**(应用该 CNOT 后的 P, 该 CNOT)**。此时 L=2 全枚举一般有大量不同矩阵，可与轨迹数同阶。
    - **`pre_action`**：观测为 **本步施加前** 的 parity（MDP 中常见）。L=2 时不同矩阵至多约 **11** 种
      （第一步恒为 I，第二步前仅依赖第一条边）。

    数据收集方式：

    - **enumerate_all_paths=True**：枚举 `edge_index` 上所有长度为 `walk_length` 的有序边序列
      （共 E^L 条轨迹）。适用于 E、L 较小（如环图 E=10, L=2 → 100 条轨迹）。
    - **enumerate_all_paths=False**：随机采样 `num_random_trajectories` 条轨迹（每步在 E 条
      有向边中均匀选列下标）。

    **去重 `dedupe_mode`**：

    - **`parity_and_edge`**（默认）：仅当 **(parity, edge_idx)** 完全一致时视为重复。`post_action`
      + L=2 时通常为 **110** 条（若矩阵无意外碰撞）。
    - **`parity_matrix`**：按矩阵去重，同一 P 只保留先遇到的一条标签。
    - **`none`**：不去重（枚举 L=2 共 **200** 条样本）。

    **注意**：`post_action` 下若「仅用 P 预测下一步边」，语义与这里「标签=刚施加的边」一致时表示
    **反演最后一步门**；若你的策略要学 **下一步** 门，应使用 `pre_action`。
    """
    if walk_length < 1:
        raise ValueError("walk_length must be >= 1")

    device = edge_index.device
    e_cnt = edge_index.size(1)
    eye = torch.eye(n_qubits, dtype=torch.float32, device=device)

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)

    seen_matrix: set[tuple[int, ...]] = set()
    seen_pair: set[tuple[tuple[int, ...], int]] = set()
    out: list[tuple[torch.Tensor, int]] = []

    def maybe_add(p_obs: torch.Tensor, edge_idx: int) -> None:
        pk = _parity_matrix_key(p_obs)
        if dedupe_mode == "none":
            out.append((p_obs.clone(), edge_idx))
            return
        if dedupe_mode == "parity_matrix":
            if pk in seen_matrix:
                return
            seen_matrix.add(pk)
            out.append((p_obs.clone(), edge_idx))
            return
        # parity_and_edge
        key = (pk, edge_idx)
        if key in seen_pair:
            return
        seen_pair.add(key)
        out.append((p_obs.clone(), edge_idx))

    def run_trajectory(edge_sequence: list[int]) -> None:
        p = eye.clone()
        for t in range(walk_length):
            e_t = edge_sequence[t]
            if observation == "pre_action":
                maybe_add(p, e_t)
                p = _apply_edge_index(p, edge_index, e_t)
            else:
                p = _apply_edge_index(p, edge_index, e_t)
                maybe_add(p, e_t)

    if enumerate_all_paths:
        for seq in itertools.product(range(e_cnt), repeat=walk_length):
            run_trajectory(list(seq))
    else:
        for _ in range(num_random_trajectories):
            seq = [
                int(torch.randint(0, e_cnt, (1,), generator=generator).item())
                for _ in range(walk_length)
            ]
            run_trajectory(seq)

    return out


def roll_out_demonstrations(
    n_qubits: int,
    edge_index: torch.Tensor,
    num_episodes: int,
    steps_per_episode: int,
    generator: torch.Generator | None = None,
) -> list[tuple[torch.Tensor, int]]:
    """
    从 P=I 出发，每步在 edge_index 上均匀随机选一条有向边施加 CNOT，
    记录 (施加前的 P, 该步选中的边下标)。
    """
    if generator is None:
        generator = torch.Generator(device=edge_index.device)
        generator.manual_seed(0)

    e_cnt = edge_index.size(1)
    samples: list[tuple[torch.Tensor, int]] = []

    for _ in range(num_episodes):
        p = torch.eye(n_qubits, dtype=torch.float32)
        for _ in range(steps_per_episode):
            k = int(torch.randint(0, e_cnt, (1,), generator=generator).item())
            samples.append((p.clone(), k))
            s, d = int(edge_index[0, k]), int(edge_index[1, k])
            p = gf2_apply_cnot(p, s, d)

    return samples


def behaviour_cloning_loss(logits: torch.Tensor, expert_edge_idx: torch.Tensor) -> torch.Tensor:
    """
    logits: [E, 1]，expert_edge_idx: 标量 0..E-1
    """
    log = logits.view(1, -1)
    target = expert_edge_idx.view(1).long()
    return F.cross_entropy(log, target)


@torch.no_grad()
def edge_classification_accuracy(
    model: ImitationPolicy,
    samples: list[tuple[torch.Tensor, int]],
    device: torch.device,
) -> float:
    correct = 0
    for cur, expert_idx in samples:
        cur = cur.to(device)
        logits = model(cur)
        pred = int(logits.squeeze(-1).argmax().item())
        if pred == expert_idx:
            correct += 1
    return correct / max(len(samples), 1)


def train_imitation_demo(
    device: torch.device | None = None,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    device = device or torch.device("cpu")

    n = 5
    edge_index = bidirectional_ring_edge_index(n, device=device)
    e_cnt = edge_index.size(1)
    assert e_cnt == 2 * n, "5 比特双向环应有 E = 10 条有向边"

    walk_len = 2
    # 样本语义：(施加该步 CNOT 后的 P, 该步边)。parity_matrix 去重：每个矩阵保留一条标签（演示用）
    ring_samples = make_deterministic_label_samples(
        n,
        edge_index,
        walk_length=walk_len,
        observation="post_action",
        enumerate_all_paths=True,
        dedupe_mode="parity_matrix",
    )
    n_s, n_u = dataset_parity_stats(ring_samples)
    print(f"bidirectional ring  E={e_cnt}  walk_length={walk_len}")
    print(f"  dataset  post_action + parity_matrix  samples={n_s}  unique_matrices={n_u}")
    print("edge_index [src; dst]:", edge_index.cpu().tolist())

    model = ImitationPolicy(
        n_qubits=n,
        edge_index=edge_index,
        node_dim=32,
        edge_dim=16,
        hidden_dim=64,
        num_layers=3,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    epochs = 120
    for ep in range(epochs):
        total_loss = 0.0
        for cur, expert_idx in ring_samples:
            cur = cur.to(device)
            target_lbl = torch.tensor(expert_idx, device=device, dtype=torch.long)
            logits = model(cur)
            loss = behaviour_cloning_loss(logits, target_lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        acc = edge_classification_accuracy(model, ring_samples, device)
        avg_loss = total_loss / len(ring_samples)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep + 1}/{epochs}  loss={avg_loss:.4f}  acc_on_ring_set={acc:.3f}")

    final_acc = edge_classification_accuracy(model, ring_samples, device)
    print("---")
    print(f"final acc on ring deterministic set (train=all) = {final_acc:.3f}")
    if final_acc >= 0.999:
        print("OK: converged on this dataset (near-zero CE on all ring samples).")
    else:
        print("Not fully fitted: try more epochs, larger lr, or wider hidden_dim.")


if __name__ == "__main__":
    train_imitation_demo()
