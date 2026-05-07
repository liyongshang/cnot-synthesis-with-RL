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

import torch

FORMAT_VERSION = 1


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

    **`observation`**（默认 `post_action`）：

    - **`post_action`**：记录 **本步 CNOT 施加之后** 的 parity。
    - **`pre_action`**：记录 **本步施加前** 的 parity。

    **去重 `dedupe_mode`**：`none` | `parity_matrix` | `parity_and_edge`。
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
