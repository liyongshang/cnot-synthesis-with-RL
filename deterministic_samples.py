"""
确定性轨迹样本：在单位阵上沿边枚举/随机走 CNOT，生成 (parity 观测, 边下标)，并可序列化到磁盘。

训练脚本应优先 `load_deterministic_dataset`；仅在文件缺失或用 `--regenerate` 时调用
`make_deterministic_label_samples` + `save_deterministic_dataset`。
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

DedupeMode = Literal[
    "none",
    "parity_matrix",
    "parity_and_edge",
    "parity_matrix_min_depth",
]

import torch

FORMAT_VERSION = 1


def _parity_matrix_key(parity: torch.Tensor) -> tuple[int, ...]:
    """GF(2) parity 矩阵的稳定键（用于去重），形状 [N, N]。"""
    return tuple(int(x) for x in parity.detach().cpu().reshape(-1).tolist())


parity_matrix_key = _parity_matrix_key  # 对外别名，语义同 `_parity_matrix_key`


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


def apply_edge_index(parity: torch.Tensor, edge_index: torch.Tensor, edge_idx: int) -> torch.Tensor:
    """对 parity 施加 edge_index 第 edge_idx 列对应的有向 CNOT。"""
    i = int(edge_index[0, edge_idx].item())
    j = int(edge_index[1, edge_idx].item())
    return gf2_apply_cnot(parity, i, j)


def _is_identity_parity(p: torch.Tensor) -> bool:
    """GF(2) 下 parity 是否为单位阵（walk_length=0 的平凡状态）。"""
    n = int(p.shape[0])
    if p.dim() != 2 or int(p.shape[1]) != n:
        return False
    mat = ((p.detach().cpu().to(dtype=torch.float32).round().clamp(0.0, 1.0)).long() % 2).bool()
    eye = torch.eye(n, dtype=torch.bool)
    return bool(torch.all(mat == eye))


def _finalize_samples(
    raw: list[tuple[torch.Tensor, int, int]],
    dedupe_mode: DedupeMode,
) -> list[tuple[torch.Tensor, int]]:
    """
    将带前缀深度 ``depth`` 的原始三元组压成 ``(parity, edge_idx)``。

    ``depth``：到达该观测时已施加的 CNOT 步数（``post_action`` 为本步之后计数，
    ``pre_action`` 为本步之前已施加的边数）。
    """
    if dedupe_mode == "none":
        return [(p, e) for p, e, _ in raw]

    if dedupe_mode == "parity_matrix":
        seen: set[tuple[int, ...]] = set()
        out: list[tuple[torch.Tensor, int]] = []
        for p, e, _ in raw:
            pk = _parity_matrix_key(p)
            if pk in seen:
                continue
            seen.add(pk)
            out.append((p, e))
        return out

    if dedupe_mode == "parity_and_edge":
        seen_pair: set[tuple[tuple[int, ...], int]] = set()
        out = []
        for p, e, _ in raw:
            pk = _parity_matrix_key(p)
            key = (pk, int(e))
            if key in seen_pair:
                continue
            seen_pair.add(key)
            out.append((p, e))
        return out

    if dedupe_mode == "parity_matrix_min_depth":
        groups: dict[tuple[int, ...], list[tuple[torch.Tensor, int, int]]] = defaultdict(list)
        for p, e, d in raw:
            groups[_parity_matrix_key(p)].append((p, e, d))
        seen_pair: set[tuple[tuple[int, ...], int]] = set()
        out = []
        for pk, items in groups.items():
            # 单位阵 I 在平凡意义上对应「0 步」可达；连续两次同一有向 CNOT 会回到 I，
            # 其观测深度 ≥1，不应短于 0。凡此情形一律按 min_depth=0，仅保留 depth==0 的样本。
            p0 = items[0][0]
            if _is_identity_parity(p0):
                min_d = 0
            else:
                min_d = min(t[2] for t in items)
            for p, e, d in items:
                if d != min_d:
                    continue
                key = (pk, int(e))
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                out.append((p, e))
        return out

    raise ValueError(f"unknown dedupe_mode {dedupe_mode!r}")


def make_deterministic_label_samples(
    n_qubits: int,
    edge_index: torch.Tensor,
    walk_length: int = 1,
    *,
    observation: Literal["pre_action", "post_action"] = "post_action",
    enumerate_all_paths: bool = True,
    num_random_trajectories: int = 512,
    generator: torch.Generator | None = None,
    dedupe_mode: DedupeMode = "parity_and_edge",
) -> list[tuple[torch.Tensor, int]]:
    """
    在单位阵 I 上沿一条轨迹连续施加 `walk_length` 个 CNOT，每一步记录一条样本：

        (parity_observation, expert_edge_idx)

    `expert_edge_idx` 均为 **该步实际施加的** 边在 `edge_index` 中的列下标。

    **`observation`**（默认 `post_action`）：

    - **`post_action`**：记录 **本步 CNOT 施加之后** 的 parity。
    - **`pre_action`**：记录 **本步施加前** 的 parity。

    **去重 `dedupe_mode`**：

    - ``none``：不去重。
    - ``parity_matrix``：同一 parity 矩阵只保留首次出现。
    - ``parity_and_edge``：``(矩阵, 边)`` 唯一。
    - ``parity_matrix_min_depth``：对每个 parity 矩阵只保留 **最短前缀深度**
      上的样本；在这些最短样本上再按 ``(矩阵, 边)`` 去重。
      **单位阵 I** 视为 ``walk_length=0`` 可达，最短深度为 **0**；因此凡在轨迹中
      深度 ``≥1`` 才观测到的 I（例如同一有向边连续施加两次后回到 I）会被剔除，
      除非另有一条 ``depth==0`` 的样本（常见于 ``pre_action`` 第一步）。
      非 I 矩阵仍取该矩阵在数据中出现的深度最小值。
    """
    if walk_length < 1:
        raise ValueError("walk_length must be >= 1")

    device = edge_index.device
    e_cnt = edge_index.size(1)
    eye = torch.eye(n_qubits, dtype=torch.float32, device=device)

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)

    raw: list[tuple[torch.Tensor, int, int]] = []

    def maybe_add_raw(p_obs: torch.Tensor, edge_idx: int, depth: int) -> None:
        raw.append((p_obs.clone(), int(edge_idx), int(depth)))

    def run_trajectory(edge_sequence: list[int]) -> None:
        p = eye.clone()
        for t in range(walk_length):
            e_t = edge_sequence[t]
            if observation == "pre_action":
                # 施加本步之前已有 t 步
                maybe_add_raw(p, e_t, depth=t)
                p = apply_edge_index(p, edge_index, e_t)
            else:
                p = apply_edge_index(p, edge_index, e_t)
                maybe_add_raw(p, e_t, depth=t + 1)

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

    return _finalize_samples(raw, dedupe_mode)


def save_deterministic_dataset(
    path: str | Path,
    samples: list[tuple[torch.Tensor, int]],
    *,
    edge_index: torch.Tensor,
    n_qubits: int,
    walk_length: int,
    observation: str,
    dedupe_mode: str,
    enumerate_all_paths: bool,
    num_random_trajectories: int | None = None,
    topology: str | None = None,
) -> None:
    """
    将样本存为 ``.pt``：堆叠 parity 为 ``[N, n, n]``，标签为 ``[N]`` long，
    另存 meta（含 ``edge_index``）便于训练端重建图。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        raise ValueError("samples is empty")
    n = n_qubits
    mats = torch.stack([p.detach().cpu().to(dtype=torch.float32) for p, _ in samples], dim=0)
    labels = torch.tensor([e for _, e in samples], dtype=torch.long)
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "n_qubits": n_qubits,
        "walk_length": walk_length,
        "observation": observation,
        "dedupe_mode": dedupe_mode,
        "enumerate_all_paths": enumerate_all_paths,
        "num_random_trajectories": num_random_trajectories,
        "edge_index": edge_index.detach().cpu().long(),
        "num_samples": len(samples),
    }
    if topology is not None:
        meta["topology"] = topology
    torch.save({"meta": meta, "parity_tensors": mats, "edge_labels": labels}, path)


def load_deterministic_dataset(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> tuple[dict[str, Any], list[tuple[torch.Tensor, int]]]:
    """
    读取 ``save_deterministic_dataset`` 写入的文件。

    返回 ``(meta, samples)``，其中 ``meta['edge_index']`` 为 CPU tensor；
    ``samples`` 中 parity 在 ``map_location`` 设备上（默认 CPU）。
    """
    path = Path(path)
    blob = torch.load(path, map_location=map_location or "cpu")
    meta = blob["meta"]
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported format_version {meta.get('format_version')}")
    mats: torch.Tensor = blob["parity_tensors"]
    labels: torch.Tensor = blob["edge_labels"]
    device = map_location if isinstance(map_location, torch.device) else torch.device(
        map_location or "cpu"
    )
    samples: list[tuple[torch.Tensor, int]] = [
        (mats[i].to(device=device, dtype=torch.float32), int(labels[i].item()))
        for i in range(mats.size(0))
    ]
    return meta, samples
