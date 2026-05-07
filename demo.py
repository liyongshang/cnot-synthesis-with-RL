"""
交互演示：加载已训练的模仿学习策略，从初始 parity 出发反复施加模型预测的 CNOT，
直到得到单位阵（或达到最大步数）。

默认初始 parity：**随机从训练集数据文件**（与 BC 相同的 ``.pt``）中抽取一条样本的矩阵；
使用 ``--random`` 则改为均匀随机 0/1 矩阵（可选 ``--random-steps`` 从 I 随机走若干门）。

依赖 matplotlib 做矩阵可视化；若未安装请先执行 ``pip install matplotlib``。

用法示例::

    python demo.py --checkpoint checkpoints/imitation_policy.pt
    python demo.py --dataset data/imitation_ring_n5_wl2_post_pm.pt --seed 0
    python demo.py --random --random-steps 8 --seed 1 --save-fig parity_rollout.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from deterministic_samples import apply_edge_index, load_deterministic_dataset
from imitation_learning import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATASET_PATH, load_imitation_policy


def parity_is_identity(p: torch.Tensor) -> bool:
    n = p.size(0)
    eye = torch.eye(n, device=p.device, dtype=p.dtype)
    pr = (p % 2).round().long() % 2
    return bool((pr == eye.long()).all())


def sample_parity_from_train_dataset(
    dataset_path: Path,
    n_qubits: int,
    seed: int | None,
) -> tuple[torch.Tensor, int, int]:
    """
    从 ``save_deterministic_dataset`` / BC 训练使用的 ``.pt`` 中随机选一条样本，
    返回其 parity 矩阵及样本下标。
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"训练集文件不存在: {dataset_path}")
    meta, samples = load_deterministic_dataset(dataset_path, map_location="cpu")
    if int(meta["n_qubits"]) != n_qubits:
        raise ValueError(
            f"训练集 n_qubits={meta['n_qubits']} 与模型 {n_qubits} 不一致；请换 --dataset 或检查模型。"
        )
    if not samples:
        raise ValueError("训练集为空")
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    n_samples = len(samples)
    idx = int(torch.randint(0, n_samples, (1,), generator=g).item())
    p0, _ = samples[idx]
    return p0.detach().cpu().float(), idx, n_samples


def random_parity(n: int, seed: int | None, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return torch.randint(0, 2, (n, n), generator=g, dtype=torch.float32, device=device)


def edge_action_label(edge_index: torch.Tensor, k: int) -> str:
    s = int(edge_index[0, k].item())
    d = int(edge_index[1, k].item())
    return f"{s}->{d}"


@torch.no_grad()
def rollout_until_identity(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    p0: torch.Tensor,
    device: torch.device,
    max_steps: int,
) -> tuple[list[torch.Tensor], list[int], bool]:
    """返回 (历史 parity CPU 张量列表, 边下标列表, 是否到达单位阵)。"""
    p = p0.clone().to(device)
    ei = edge_index.to(device)
    history = [p.detach().cpu().clone()]
    actions: list[int] = []
    for _ in range(max_steps):
        if parity_is_identity(p):
            return history, actions, True
        logits = model(p)
        k = int(logits.squeeze(-1).argmax().item())
        actions.append(k)
        p = apply_edge_index(p, ei, k)
        history.append(p.detach().cpu().clone())
    return history, actions, parity_is_identity(p)


def visualize_rollout(
    history: list[torch.Tensor],
    actions: list[int],
    edge_index: torch.Tensor,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise RuntimeError("可视化需要 matplotlib：pip install matplotlib") from e

    n_steps = len(history)
    cols = min(6, n_steps)
    rows = (n_steps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes_arr = np.atleast_1d(axes).ravel()

    for t in range(n_steps):
        ax = axes_arr[t]
        mat = history[t].numpy()
        ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        if t == 0:
            ax.set_title(f"t={t}\ninit")
        else:
            lbl = edge_action_label(edge_index, actions[t - 1])
            ax.set_title(f"t={t}\nCNOT({lbl})")
        ax.set_xlabel("col")
        ax.set_ylabel("row")

    for j in range(n_steps, len(axes_arr)):
        axes_arr[j].set_visible(False)

    fig.suptitle("Parity matrix (GF2) rollout toward identity", fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"saved figure -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive parity reduction demo (BC policy).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="policy checkpoint from imitation_learning training",
    )
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f".pt 训练集路径，用于默认抽取初始 parity（默认: {DEFAULT_DATASET_PATH}）",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="不从训练集抽样；改为均匀随机 parity（可用 --random-steps）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子：训练集抽样或 --random 初始矩阵共用",
    )
    parser.add_argument("--random-steps", type=int, default=0, help="if >0, build start by random CNOTs from I")
    parser.add_argument("--save-fig", type=Path, default=None, help="save matplotlib figure to this path")
    parser.add_argument("--no-show", action="store_true", help="do not open GUI (still saves if --save-fig)")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not args.checkpoint.is_file():
        print(f"找不到检查点: {args.checkpoint}", file=sys.stderr)
        print("请先运行: python imitation_learning.py", file=sys.stderr)
        sys.exit(1)

    model, blob = load_imitation_policy(args.checkpoint, map_location=device)
    cfg = blob["config"]
    n = int(cfg["n_qubits"])
    edge_index = cfg["edge_index"]
    print(f"loaded checkpoint  n_qubits={n}  E={edge_index.size(1)}")
    print("edge_index [src row; dst row]:", edge_index.tolist())

    if args.random:
        p0 = random_parity(n, args.seed, torch.device("cpu"))
        if args.random_steps > 0:
            g = torch.Generator()
            if args.seed is not None:
                g.manual_seed(args.seed + 12345)
            ei = edge_index.long()
            for _ in range(args.random_steps):
                k = int(torch.randint(0, ei.size(1), (1,), generator=g).item())
                p0 = apply_edge_index(p0, ei, k)
        print("random start parity (not from train set):")
        print(p0.int().numpy())
    else:
        try:
            p0, sample_idx, n_train = sample_parity_from_train_dataset(args.dataset, n, args.seed)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        print(
            f"start parity from train set: sample {sample_idx}/{n_train}  "
            f"(dataset={args.dataset})"
        )
        print(p0.int().numpy())

    if p0.shape != (n, n):
        raise ValueError(f"parity 形状应为 [{n},{n}]，得到 {tuple(p0.shape)}")

    hist, acts, ok = rollout_until_identity(model, edge_index, p0, device, args.max_steps)
    print(f"steps taken: {len(acts)}  reached_identity={ok}")
    if not ok:
        print("(未在 max-steps 内到达单位阵；模型或图可能不足以化简该状态。)")

    visualize_rollout(
        hist,
        acts,
        edge_index,
        save_path=args.save_fig,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
