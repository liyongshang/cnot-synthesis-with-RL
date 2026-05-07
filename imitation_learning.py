"""
模仿学习（行为克隆）最小测试：样本为 (current_parity, expert_edge)。

约定
----
- **target_parity** 恒为 N×N 单位阵 I（GF(2) 下目标为各行标准基 / 单位线性变换）。
- **current_parity**：输入网络的 parity 观测，形状 [N, N]。默认约定为 **最近一次施加 CNOT 之后**
  的矩阵（见 `deterministic_samples.make_deterministic_label_samples` 的 `observation`）；亦可设为施加前（MDP 常用）。
- **edge**：专家在该状态下选择的有向 CNOT，对应 `edge_index` 中的列下标 e ∈ {0, …, E−1}；
  执行规则：P[dst] ← P[dst] ⊕ P[src]。

节点输入说明：仅用「目标行」编码会使 T=I 时节点特征与状态无关，因此将 **每行** 拼接为
[I[i] ‖ P[i]]（长度 2N），再线性映射到 node_dim；边特征仍用 `ParityEdgeFeatEncoder(I, edge_index, P)`。

数据说明
----
- 确定性样本的生成、保存与加载见 **`deterministic_samples.py`**。
- 训练默认从 **`data/imitation_ring_n5_wl2_post_pm.pt`** 读取；不存在则生成并保存。
  强制重建：``python imitation_learning.py --regenerate``。

用法：运行 `train_imitation_demo()` 或 ``python imitation_learning.py``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import CNOTPolicyNet, ParityEdgeFeatEncoder
from deterministic_samples import (
    dataset_parity_stats,
    gf2_apply_cnot,
    load_deterministic_dataset,
    make_deterministic_label_samples,
    save_deterministic_dataset,
)


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "imitation_ring_n5_wl2_post_pm.pt"


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


def bidirectional_line_edge_index(
    n_qubits: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    双向线性链：顶点 0—1—…—N−1，仅在相邻比特间有无向连接；
    每条无向边展开为两条有向边 (i→j) 与 (j→i)。

    返回 edge_index，形状 [2, E]，E = 2 * (N − 1)（N ≥ 2）。
    """
    if n_qubits < 2:
        raise ValueError("linear topology requires n_qubits >= 2")
    dev = device or torch.device("cpu")
    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(n_qubits - 1):
        j = i + 1
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
    *,
    dataset_path: Path | None = None,
    regenerate: bool = False,
) -> None:
    torch.manual_seed(seed)
    device = device or torch.device("cpu")
    dataset_path = dataset_path or DEFAULT_DATASET_PATH

    n = 5
    walk_len = 2
    obs_mode = "post_action"
    dedupe = "parity_matrix"

    if regenerate or not dataset_path.is_file():
        edge_index_cpu = bidirectional_ring_edge_index(n, device=torch.device("cpu"))
        ring_samples = make_deterministic_label_samples(
            n,
            edge_index_cpu,
            walk_length=walk_len,
            observation=obs_mode,
            enumerate_all_paths=True,
            dedupe_mode=dedupe,
        )
        save_deterministic_dataset(
            dataset_path,
            ring_samples,
            edge_index=edge_index_cpu,
            n_qubits=n,
            walk_length=walk_len,
            observation=obs_mode,
            dedupe_mode=dedupe,
            enumerate_all_paths=True,
            num_random_trajectories=None,
        )
        print(f"wrote dataset -> {dataset_path}")

    meta, ring_samples = load_deterministic_dataset(dataset_path, map_location=device)
    edge_index = meta["edge_index"].to(device=device)
    n_meta = int(meta["n_qubits"])
    assert n_meta == n, "dataset n_qubits mismatch"
    e_cnt = edge_index.size(1)
    assert e_cnt == 2 * n, "5 比特双向环应有 E = 10 条有向边"

    n_s, n_u = dataset_parity_stats(ring_samples)
    print(f"loaded dataset {dataset_path}")
    print(f"bidirectional ring  E={e_cnt}  walk_length={meta.get('walk_length')}")
    print(f"  samples={n_s}  unique_matrices={n_u}")
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
    parser = argparse.ArgumentParser(description="BC demo; loads saved deterministic samples by default.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rebuild dataset file even if it exists",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=f"path to .pt dataset (default: {DEFAULT_DATASET_PATH})",
    )
    args = parser.parse_args()
    train_imitation_demo(dataset_path=args.dataset, regenerate=args.regenerate)

