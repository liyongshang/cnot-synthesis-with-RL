"""
仅生成并保存确定性样本集，不进行训练。

说明：walk_length 较大时 ``enumerate_all_paths`` 对应 E^L 条轨迹，规模爆炸；
默认改为随机采样若干条轨迹（``num_random_trajectories``），再按 ``dedupe_mode`` 去重。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deterministic_samples import (
    dataset_parity_stats,
    label_ambiguity_stats,
    make_deterministic_label_samples,
    save_deterministic_dataset,
)
from imitation_learning import bidirectional_line_edge_index, bidirectional_ring_edge_index

MAX_SAFE_ENUM = 500_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic BC dataset (.pt), no training.")
    parser.add_argument("--n-qubits", type=int, default=5)
    parser.add_argument(
        "--topology",
        choices=["line", "ring"],
        default="line",
        help="hardware graph: open chain vs ring",
    )
    parser.add_argument("--walk-length", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-random-trajectories",
        type=int,
        default=40_000,
        help="when not enumerating: number of random root-to-leaf walks",
    )
    parser.add_argument(
        "--enumerate-all",
        action="store_true",
        help=f"enumerate all E^L paths (refused if E^L > {MAX_SAFE_ENUM})",
    )
    parser.add_argument(
        "--dedupe",
        choices=["none", "parity_matrix", "parity_and_edge", "parity_matrix_min_depth"],
        default="parity_matrix",
    )
    parser.add_argument(
        "--observation",
        choices=["pre_action", "post_action"],
        default="post_action",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output .pt path (default: data/{line|ring}_n{N}_wl{L}_post_pm.pt)",
    )
    args = parser.parse_args()

    n = args.n_qubits
    L = args.walk_length
    device = torch.device("cpu")
    if args.topology == "line":
        edge_index = bidirectional_line_edge_index(n, device=device)
    else:
        edge_index = bidirectional_ring_edge_index(n, device=device)
    e_cnt = edge_index.size(1)

    enumerate_all = args.enumerate_all
    if enumerate_all:
        total_paths = e_cnt**L
        if total_paths > MAX_SAFE_ENUM:
            raise SystemExit(
                f"E^L = {e_cnt}^{L} = {total_paths} exceeds limit {MAX_SAFE_ENUM}; "
                "omit --enumerate-all and use --num-random-trajectories instead."
            )

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    samples = make_deterministic_label_samples(
        n,
        edge_index,
        walk_length=L,
        observation=args.observation,
        enumerate_all_paths=enumerate_all,
        num_random_trajectories=args.num_random_trajectories,
        generator=gen,
        dedupe_mode=args.dedupe,
    )

    out_path = args.output
    if out_path is None:
        tag = "post_pm" if args.observation == "post_action" else "pre_pm"
        topo = args.topology
        out_path = Path(__file__).resolve().parent / "data" / f"{topo}_n{n}_wl{L}_{tag}.pt"

    save_deterministic_dataset(
        out_path,
        samples,
        edge_index=edge_index,
        n_qubits=n,
        walk_length=L,
        observation=args.observation,
        dedupe_mode=args.dedupe,
        enumerate_all_paths=enumerate_all,
        num_random_trajectories=None if enumerate_all else args.num_random_trajectories,
        topology=args.topology,
    )

    ns, nu = dataset_parity_stats(samples)
    n_mat_amb, n_conflict = label_ambiguity_stats(samples)
    print(f"wrote {out_path}")
    print(f"  topology={args.topology}  samples={ns}  unique_parity_matrices={nu}  E={e_cnt}  walk_length={L}")
    print(f"  dedupe_mode={args.dedupe}  matrices_with_multi_edge_labels={n_conflict}/{n_mat_amb}")
    if not enumerate_all:
        print(f"  random_trajectories={args.num_random_trajectories}  seed={args.seed}")


if __name__ == "__main__":
    main()
