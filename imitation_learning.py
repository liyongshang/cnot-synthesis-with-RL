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

用法：运行 `train_imitation_demo()` 或 ``python imitation_learning.py``（默认 **walk_length=2** 数据，
**随机 90% 训练 / 10% 测试**，并在 **acc_train / acc_test** 达到阈值且 epoch≥``min_epochs`` 时 **早停**）。

训练结束保存权重到 ``checkpoints/imitation_policy.pt``（``--checkpoint`` 可改路径）。

交互演示：``python demo.py``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from agent import CNOTPolicyNet, ParityEdgeFeatEncoder
from collections import defaultdict

from deterministic_samples import (
    dataset_parity_stats,
    gf2_apply_cnot,
    label_ambiguity_stats,
    load_deterministic_dataset,
    make_deterministic_label_samples,
    parity_matrix_key,
    save_deterministic_dataset,
)


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "line_n5_wl2_post_pm_mindepth.pt"
DEFAULT_WL7_DATASET_PATH = Path(__file__).resolve().parent / "data" / "line_n5_wl7_post_pm.pt"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "imitation_policy.pt"
DEFAULT_WL7_CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "imitation_policy_wl7.pt"


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
        if target_parity.dim() == 2:
            if target_parity.shape != (self.n_qubits, self.n_qubits):
                raise ValueError(f"target_parity 期望 [{self.n_qubits},{self.n_qubits}]")
            if current_parity.shape != target_parity.shape:
                raise ValueError("current_parity 形状须与 target_parity 一致")
        elif target_parity.dim() == 3:
            if target_parity.shape[1:] != (self.n_qubits, self.n_qubits):
                raise ValueError(
                    f"target_parity 期望 [B,{self.n_qubits},{self.n_qubits}]"
                )
            if current_parity.shape != target_parity.shape:
                raise ValueError("current_parity 形状须与 target_parity 一致")
        else:
            raise ValueError("target_parity 须为 [N,N] 或 [B,N,N]")
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
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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

    def forward_batched(self, current_parity: torch.Tensor) -> torch.Tensor:
        """
        一批样本共享同一 target=I 与 edge_index；用 disjoint union 拼成大图一次前向。
        current_parity: [B, N, N] -> logits [B, E, 1]
        """
        if current_parity.dim() != 3:
            raise ValueError(f"forward_batched 需要 [B,N,N]，得到 {tuple(current_parity.shape)}")
        b, n, _ = current_parity.shape
        if n != self.n_qubits:
            raise ValueError(f"N={n} 与 n_qubits={self.n_qubits} 不一致")
        device = current_parity.device
        dtype = current_parity.dtype
        target = self.target_parity.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)
        node_feat = self.node_enc(target, current_parity)  # [B, N, node_dim]
        edge_feat = self.edge_enc(target, self.edge_index, current_parity)  # [B, E, edge_dim]
        e_cnt = int(self.edge_index.size(1))
        node_flat = node_feat.reshape(b * n, -1)
        ei = disjoint_batch_edge_index(self.edge_index, n, b, device=device)
        ef_flat = edge_feat.reshape(b * e_cnt, -1)
        logits_flat = self.policy(node_flat, ei, ef_flat)  # [B*E, 1]
        return logits_flat.view(b, e_cnt, 1)

    def forward(self, current_parity: torch.Tensor) -> torch.Tensor:
        if current_parity.dim() != 2:
            raise ValueError(f"单样本 forward 需要 [N,N]，得到 {tuple(current_parity.shape)}")
        return self.forward_batched(current_parity.unsqueeze(0)).squeeze(0)


def disjoint_batch_edge_index(
    edge_index: torch.Tensor,
    n_nodes: int,
    batch_size: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """同一拓扑重复 batch_size 次，节点编号加上 0, N, 2N, … 偏移，得到 [2, batch_size*E]。"""
    dev = device or edge_index.device
    ei = edge_index.to(device=dev)
    off = (
        torch.arange(batch_size, device=dev, dtype=ei.dtype).view(batch_size, 1, 1) * int(n_nodes)
    )
    stacked = ei.unsqueeze(0).expand(batch_size, -1, -1) + off
    # 不可对 [B,2,E] 直接 reshape(2, B*E)：会把同一图的 src/dst 行交错打乱。
    # 正确顺序：先沿 batch 串接每条边对应的 src，再串接 dst（与 edge_feat [B,E,*] 展平一致）。
    src_flat = stacked[:, 0, :].reshape(-1)
    dst_flat = stacked[:, 1, :].reshape(-1)
    return torch.stack([src_flat, dst_flat], dim=0)


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


def save_imitation_policy(
    path: str | Path,
    model: ImitationPolicy,
    *,
    extra_meta: dict | None = None,
) -> None:
    """保存策略权重与构图配置（含 ``edge_index``），供 ``load_imitation_policy`` 还原。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "state_dict": model.state_dict(),
        "config": {
            "n_qubits": model.n_qubits,
            "node_dim": model.node_dim,
            "edge_dim": model.edge_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "edge_index": model.edge_index.detach().cpu().long(),
        },
        "extra": extra_meta or {},
    }
    torch.save(payload, path)


def load_imitation_policy(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> tuple[ImitationPolicy, dict]:
    """
    加载 ``save_imitation_policy`` 保存的检查点。

    返回 ``(model, payload)``，``payload`` 含 ``config``、``extra`` 等元数据。
    """
    path = Path(path)
    device = map_location if isinstance(map_location, torch.device) else torch.device(
        map_location or "cpu"
    )
    blob = torch.load(path, map_location=device)
    if blob.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format_version {blob.get('format_version')}")
    cfg = blob["config"]
    edge_index = cfg["edge_index"].to(device=device)
    model = ImitationPolicy(
        n_qubits=int(cfg["n_qubits"]),
        edge_index=edge_index,
        node_dim=int(cfg["node_dim"]),
        edge_dim=int(cfg["edge_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def behaviour_cloning_loss(logits: torch.Tensor, expert_edge_idx: torch.Tensor) -> torch.Tensor:
    """
    logits: [E, 1]，expert_edge_idx: 标量 0..E-1
    """
    log = logits.view(1, -1)
    target = expert_edge_idx.view(1).long()
    return F.cross_entropy(log, target)


def behaviour_cloning_loss_batched(
    logits: torch.Tensor,
    expert_edge_idx: torch.Tensor,
) -> torch.Tensor:
    """logits: [B, E, 1]；expert_edge_idx: [B]，类别为边下标。"""
    log = logits.squeeze(-1)
    return F.cross_entropy(log, expert_edge_idx.long())


def valid_edges_frozenset_by_matrix(
    samples: list[tuple[torch.Tensor, int]],
) -> dict[tuple[int, ...], frozenset[int]]:
    """同一 parity 矩阵对应的所有合法有向边下标（合并多条样本行）。"""
    g: defaultdict[tuple[int, ...], set[int]] = defaultdict(set)
    for p, e in samples:
        g[parity_matrix_key(p)].add(int(e))
    return {k: frozenset(v) for k, v in g.items()}


def matrix_class_ids_and_valid_mask(
    samples: list[tuple[torch.Tensor, int]],
    valid_by_pk: dict[tuple[int, ...], frozenset[int]],
    num_edges: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将样本行的 parity 映射到类别 id ``0..U-1``，并构造 ``valid_mask_ec[u,e]``。
    返回 ``mids_cpu`` (CPU long [N])、``valid_mask_ec`` (device bool [U,E])。
    """
    if not samples:
        empty = torch.empty(0, dtype=torch.long)
        if device.type == "cuda":
            empty = empty.pin_memory()
        return empty, torch.zeros(0, num_edges, dtype=torch.bool, device=device)
    pk_to_id: dict[tuple[int, ...], int] = {}
    mids_list: list[int] = []
    for p, _ in samples:
        pk = parity_matrix_key(p)
        if pk not in pk_to_id:
            pk_to_id[pk] = len(pk_to_id)
        mids_list.append(pk_to_id[pk])
    u = len(pk_to_id)
    mask = torch.zeros(u, num_edges, dtype=torch.bool, device=device)
    for pk, es in valid_by_pk.items():
        rid = pk_to_id.get(pk)
        if rid is None:
            continue
        for e in es:
            ei = int(e)
            if 0 <= ei < num_edges:
                mask[rid, ei] = True
    mids_cpu = torch.tensor(mids_list, dtype=torch.long)
    if device.type == "cuda":
        mids_cpu = mids_cpu.pin_memory()
    return mids_cpu, mask


def behaviour_cloning_loss_multi_valid(
    logits: torch.Tensor,
    batch_matrix_ids: torch.Tensor,
    valid_mask_ec: torch.Tensor,
) -> torch.Tensor:
    """
    向量化：``batch_matrix_ids`` [B] 对应 ``valid_mask_ec[u,e]`` 的行。
    最小化 ``-(logsumexp(z_V) - logsumexp(z))``。
    """
    z = logits.squeeze(-1)
    m = valid_mask_ec[batch_matrix_ids]
    z_v = torch.logsumexp(z.masked_fill(~m, float("-inf")), dim=-1)
    z_all = torch.logsumexp(z, dim=-1)
    return -(z_v - z_all).mean()


def split_train_test(
    samples: list[tuple[torch.Tensor, int]],
    train_ratio: float,
    seed: int,
) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    """随机打乱后按比例划分训练集 / 测试集（用于泛化评估）。``train_ratio=1`` 时测试集为空。"""
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError("train_ratio must be in (0, 1]")
    n = len(samples)
    if n < 1:
        raise ValueError("need at least 1 sample")
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    shuffled = [samples[i] for i in perm]
    if train_ratio >= 1.0:
        return shuffled, []
    if n < 2:
        raise ValueError("need at least 2 samples when train_ratio < 1")
    # e.g. n=71, ratio=0.9 -> round(63.9)=64 train, 7 test
    n_train = max(1, min(int(round(n * train_ratio)), n - 1))
    train_samples = shuffled[:n_train]
    test_samples = shuffled[n_train:]
    return train_samples, test_samples


@torch.no_grad()
def edge_classification_accuracy(
    model: ImitationPolicy,
    samples: list[tuple[torch.Tensor, int]],
    device: torch.device,
    batch_size: int = 512,
) -> float:
    n = len(samples)
    if n == 0:
        return 0.0
    correct = 0
    for start in range(0, n, batch_size):
        chunk = samples[start : start + batch_size]
        cur = torch.stack([c for c, _ in chunk]).to(device)
        y = torch.tensor([e for _, e in chunk], device=device, dtype=torch.long)
        logits = model.forward_batched(cur)
        pred = logits.squeeze(-1).argmax(dim=-1)
        correct += int((pred == y).sum().item())
    return correct / n


@torch.no_grad()
def edge_classification_accuracy_multi(
    model: ImitationPolicy,
    parity_block: torch.Tensor,
    mids_cpu: torch.Tensor,
    valid_mask_ec: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> float:
    """``parity_block`` [N,N,N] 与 ``mids_cpu`` [N] 对齐；预测边落在对应 mask 行即判对。"""
    n = int(parity_block.size(0))
    if n == 0:
        return 0.0
    model.eval()
    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        cur = parity_block[start:end]
        mid = mids_cpu[start:end].to(device=device, non_blocking=True)
        logits = model.forward_batched(cur)
        pred = logits.squeeze(-1).argmax(dim=-1)
        m = valid_mask_ec[mid]
        ok = m[torch.arange(cur.size(0), device=device), pred]
        correct += int(ok.sum().item())
    return correct / n


def _resolve_training_device(device: torch.device | str | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def train_imitation_demo(
    device: torch.device | str | None = None,
    seed: int = 0,
    *,
    dataset_path: Path | None = None,
    regenerate: bool = False,
    checkpoint_path: Path | None = None,
    train_ratio: float = 0.9,
    split_seed: int | None = None,
    max_epochs: int = 120,
    early_stop: bool = True,
    min_epochs: int = 5,
    stop_acc_train: float = 0.999,
    stop_acc_test: float = 0.99,
    batch_size: int = 128,
    use_amp: bool = True,
    eval_batch_size: int = 512,
    eval_every_epochs: int = 5,
) -> None:
    torch.manual_seed(seed)
    resolved = _resolve_training_device(device)
    dataset_path = dataset_path or DEFAULT_DATASET_PATH

    n_ring = 5
    walk_len = 2
    obs_mode = "post_action"
    dedupe = "parity_matrix_min_depth"

    if regenerate or not dataset_path.is_file():
        edge_index_cpu = bidirectional_ring_edge_index(n_ring, device=torch.device("cpu"))
        ring_samples = make_deterministic_label_samples(
            n_ring,
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
            n_qubits=n_ring,
            walk_length=walk_len,
            observation=obs_mode,
            dedupe_mode=dedupe,
            enumerate_all_paths=True,
            num_random_trajectories=None,
        )
        print(f"wrote dataset -> {dataset_path}")

    meta, ring_samples = load_deterministic_dataset(dataset_path, map_location=resolved)
    edge_index = meta["edge_index"].to(device=resolved)
    n = int(meta["n_qubits"])
    e_cnt = int(edge_index.size(1))
    walk_meta = int(meta.get("walk_length", walk_len))

    use_amp_eff = bool(use_amp and resolved.type == "cuda")
    scaler = GradScaler(enabled=use_amp_eff)

    n_s, n_u = dataset_parity_stats(ring_samples)
    print(f"loaded dataset {dataset_path}")
    print(
        f"  device={resolved}  batch_size={batch_size}  amp={use_amp_eff}  "
        f"eval_every={eval_every_epochs}  "
        f"E={e_cnt}  n_qubits={n}  walk_length={walk_meta}"
    )
    print(f"  samples={n_s}  unique_matrices={n_u}")
    print("edge_index [src; dst]:", edge_index.cpu().tolist())

    split_s = split_seed if split_seed is not None else seed + 2025
    train_samples, test_samples = split_train_test(ring_samples, train_ratio, split_s)
    has_test = len(test_samples) > 0
    print(
        f"train/test split  ratio={train_ratio:.0%}  seed={split_s}  "
        f"|train|={len(train_samples)}  |test|={len(test_samples)}"
    )

    train_valid = valid_edges_frozenset_by_matrix(train_samples)
    test_valid = valid_edges_frozenset_by_matrix(test_samples)
    nmt, amb_tr = label_ambiguity_stats(train_samples)
    nmv, amb_te = label_ambiguity_stats(test_samples)
    print(
        f"  multi-valid-edge matrices  train={amb_tr}/{nmt}  test={amb_te}/{nmv}  "
        f"(same-matrix multi-action counts)"
    )

    train_mats_cpu = torch.stack([p for p, _ in train_samples])
    if has_test:
        test_mats_cpu = torch.stack([p for p, _ in test_samples])
    else:
        test_mats_cpu = torch.empty(0, n, n, dtype=torch.float32)
    if resolved.type == "cuda":
        train_mats_cpu = train_mats_cpu.pin_memory()
        if has_test:
            test_mats_cpu = test_mats_cpu.pin_memory()
    train_mids_cpu, train_mask_gpu = matrix_class_ids_and_valid_mask(
        train_samples, train_valid, e_cnt, resolved
    )
    test_mids_cpu, test_mask_gpu = matrix_class_ids_and_valid_mask(
        test_samples, test_valid, e_cnt, resolved
    )
    train_mats_gpu = train_mats_cpu.to(resolved, non_blocking=True)
    test_mats_gpu = test_mats_cpu.to(resolved, non_blocking=True)

    model = ImitationPolicy(
        n_qubits=n,
        edge_index=edge_index,
        node_dim=32,
        edge_dim=16,
        hidden_dim=64,
        num_layers=3,
    ).to(resolved)
    if resolved.type == "cuda":
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    stopped_early = False
    completed_epochs = 0
    shuf = torch.Generator(device="cpu")
    last_acc_tr = 0.0
    last_acc_te = 0.0
    for ep in range(max_epochs):
        shuf.manual_seed(seed + ep * 10007)
        perm = torch.randperm(len(train_samples), generator=shuf).tolist()
        total_loss = 0.0
        n_batches = 0
        n_train = len(train_samples)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            idx_t = torch.tensor(idx, dtype=torch.long, device=resolved)
            batch_cur = train_mats_gpu.index_select(0, idx_t)
            batch_mids = train_mids_cpu[idx].to(resolved, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            model.train()
            with autocast(enabled=use_amp_eff):
                logits = model.forward_batched(batch_cur)
                loss = behaviour_cloning_loss_multi_valid(logits, batch_mids, train_mask_gpu)
            if use_amp_eff:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        completed_epochs = ep + 1
        do_eval = (
            eval_every_epochs <= 1
            or completed_epochs % eval_every_epochs == 0
            or ep == 0
            or ep == max_epochs - 1
        )
        if do_eval:
            acc_tr = edge_classification_accuracy_multi(
                model,
                train_mats_gpu,
                train_mids_cpu,
                train_mask_gpu,
                resolved,
                batch_size=eval_batch_size,
            )
            if has_test:
                acc_te = edge_classification_accuracy_multi(
                    model,
                    test_mats_gpu,
                    test_mids_cpu,
                    test_mask_gpu,
                    resolved,
                    batch_size=eval_batch_size,
                )
            else:
                acc_te = float("nan")
            last_acc_tr, last_acc_te = acc_tr, acc_te
        else:
            acc_tr, acc_te = last_acc_tr, last_acc_te

        if completed_epochs % 10 == 0 or ep == 0:
            if do_eval:
                te_str = f"{acc_te:.3f}" if has_test else "n/a"
                print(
                    f"epoch {completed_epochs}/{max_epochs}  loss={avg_loss:.4f}  "
                    f"acc_train={acc_tr:.3f}  acc_test={te_str}"
                )
            else:
                print(
                    f"epoch {completed_epochs}/{max_epochs}  loss={avg_loss:.4f}  "
                    f"(skip eval, eval_every={eval_every_epochs})"
                )

        if (
            early_stop
            and do_eval
            and completed_epochs >= min_epochs
            and acc_tr >= stop_acc_train
            and (not has_test or acc_te >= stop_acc_test)
        ):
            te_str = f"{acc_te:.3f}>={stop_acc_test}" if has_test else "no test split"
            print(
                f"early stop at epoch {completed_epochs}: "
                f"acc_train={acc_tr:.3f}>={stop_acc_train}  "
                f"acc_test {te_str}"
            )
            stopped_early = True
            break

    final_acc_train = edge_classification_accuracy_multi(
        model,
        train_mats_gpu,
        train_mids_cpu,
        train_mask_gpu,
        resolved,
        batch_size=eval_batch_size,
    )
    final_acc_test = (
        edge_classification_accuracy_multi(
            model,
            test_mats_gpu,
            test_mids_cpu,
            test_mask_gpu,
            resolved,
            batch_size=eval_batch_size,
        )
        if has_test
        else float("nan")
    )
    print("---")
    te_fin = f"{final_acc_test:.3f}" if has_test else "n/a"
    print(
        f"final acc_train={final_acc_train:.3f}  acc_test={te_fin}  "
        f"(walk_length={walk_meta}, epochs={completed_epochs}"
        f"{' early_stop' if stopped_early else ''})"
    )
    if final_acc_train >= 0.999:
        print("OK: training set nearly fitted.")
    else:
        print("Training set not fully fitted; consider more epochs.")

    ckpt = checkpoint_path if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
    save_imitation_policy(
        ckpt,
        model,
        extra_meta={
            "dataset_path": str(dataset_path),
            "final_acc_train": float(final_acc_train),
            "final_acc_test": (
                float(final_acc_test) if has_test else None
            ),
            "train_ratio": float(train_ratio),
            "split_seed": int(split_s),
            "walk_length": walk_meta,
            "epochs_completed": int(completed_epochs),
            "early_stopped": bool(stopped_early),
            "max_epochs": int(max_epochs),
            "train_batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "eval_every_epochs": int(eval_every_epochs),
            "device": str(resolved),
            "use_amp": bool(use_amp_eff),
        },
    )
    print(f"saved policy checkpoint -> {ckpt}")


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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"where to save trained weights (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="训练集占比 (0,1]；为 1 时不用测试集，早停仅看 acc_train",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="seed for train/test shuffle (default: train seed + 2025)",
    )
    parser.add_argument("--max-epochs", type=int, default=120, help="upper bound on training epochs")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="disable convergence early stopping (always run --max-epochs)",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=5,
        help="minimum epochs before early stop can trigger",
    )
    parser.add_argument(
        "--stop-acc-train",
        type=float,
        default=0.999,
        help="early stop when acc_train reaches this (and test threshold)",
    )
    parser.add_argument(
        "--stop-acc-test",
        type=float,
        default=0.99,
        help="early stop when acc_test reaches this (with --stop-acc-train)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="训练设备：cpu | cuda | cuda:0 | auto（默认自动选 CUDA）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="训练 mini-batch 大小（GPU 上可调大以提升利用率）",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=512,
        help="评估准确率时的 batch 大小",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        dest="eval_every",
        help="每多少个 epoch 做一次 train/test 准确率评估（默认 5，设为 1 则每轮都评，慢）",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="关闭 CUDA 混合精度（默认在 CUDA 上开启 autocast + GradScaler）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子（数据打乱、训练洗牌）")
    args = parser.parse_args()
    train_imitation_demo(
        device=args.device,
        seed=args.seed,
        dataset_path=args.dataset,
        regenerate=args.regenerate,
        checkpoint_path=args.checkpoint,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        max_epochs=args.max_epochs,
        early_stop=not args.no_early_stop,
        min_epochs=args.min_epochs,
        stop_acc_train=args.stop_acc_train,
        stop_acc_test=args.stop_acc_test,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
        eval_batch_size=args.eval_batch_size,
        eval_every_epochs=args.eval_every,
    )

