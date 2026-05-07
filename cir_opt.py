"""
用训练好的 ``ImitationPolicy`` 将一条 **仅含 CX/CNOT** 的 Qiskit 线路「重写」为策略 rollout。

数据约定（与仓库其它部分一致）：
- parity 矩阵 ``P ∈ GF(2)^{N×N}``，初始为单位阵 ``I``；
- 有向 CNOT(control→target) 对应行更新 ``P[target] ← P[target] ⊕ P[control]``（见 ``gf2_apply_cnot``）；
- 策略仅在与训练相同的 **硬件 ``edge_index``** 上选边（由 checkpoint 内的 ``config`` 决定）。

流程：读取 Qiskit 线路 → 仿真得到当前 ``P`` → ``rollout_until_identity`` 直到 ``P=I``（或步数上限）
→ 将预测的边下标序列还原为新的 ``QuantumCircuit``。

依赖：``pip install qiskit``（推荐 qiskit ≥ 1.0）。

用法示例::

    python cir_opt.py --checkpoint checkpoints/imitation_policy.pt --qasm circuit.qasm
    python cir_opt.py --checkpoint ckpt.pt --cx-chain "0,1;1,2;2,1"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from demo import edge_action_label, parity_is_identity, rollout_until_identity
from deterministic_samples import gf2_apply_cnot
from imitation_learning import DEFAULT_CHECKPOINT_PATH, load_imitation_policy


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit  # noqa: F401
    except ImportError as e:
        raise RuntimeError("请先安装 qiskit：pip install qiskit") from e


def quantum_circuit_to_parity_matrix(qc: "QuantumCircuit") -> torch.Tensor:
    """
    按时间顺序施加线路中的 CX，从 ``I`` 递推 parity 矩阵（仅允许 cx/cnot；可跳过 barrier）。
    """
    n = int(qc.num_qubits)
    p = torch.eye(n, dtype=torch.float32)
    for instruction in qc.data:
        op = instruction.operation
        name = (op.name or "").lower()
        if name in ("barrier", "id", "i"):
            continue
        if name in ("measure",):
            raise ValueError("不支持含测量的线路，请使用仅 unitary / CX 的线路")
        if name not in ("cx", "cnot"):
            raise ValueError(f"仅支持 CNOT 线路，遇到非法门：{op.name!r}")
        q0, q1 = instruction.qubits
        ctrl = int(qc.find_bit(q0).index)
        tgt = int(qc.find_bit(q1).index)
        p = gf2_apply_cnot(p, ctrl, tgt)
    return p


def load_quantum_circuit_from_qasm(path: Path):
    _require_qiskit()
    from qiskit import QuantumCircuit

    text = path.read_text(encoding="utf-8")
    try:
        from qiskit import qasm2

        return qasm2.loads(text)
    except Exception:
        # 旧版 API
        return QuantumCircuit.from_qasm_str(text)


def build_circuit_from_cx_chain(cx_chain: str, n_qubits: int):
    """``cx_chain`` 形如 ``0,1;1,2``（分号分隔多门）。"""
    _require_qiskit()
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    parts = [x.strip() for x in cx_chain.split(";") if x.strip()]
    for p in parts:
        a, b = p.replace(",", " ").split()
        ctrl, tgt = int(a), int(b)
        qc.cx(ctrl, tgt)
    return qc


def parity_actions_to_quantum_circuit(
    actions: list[int],
    edge_index: torch.Tensor,
    n_qubits: int,
):
    _require_qiskit()
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    ei = edge_index.detach().cpu()
    for k in actions:
        s = int(ei[0, k].item())
        d = int(ei[1, k].item())
        qc.cx(s, d)
    return qc


def run_optimize(
    *,
    qc_input: "QuantumCircuit",
    device: str,
    max_steps: int,
    checkpoint: Path | None = None,
    model: object | None = None,
) -> tuple[object, torch.Tensor, list[int], bool, int, torch.Tensor]:
    """
    返回 ``(qc_out, p_start, actions, reached_identity, original_cx_count, edge_index_cpu)``。

    ``model`` 与 ``checkpoint`` 二选一；若都提供则以 ``model`` 为准。
    """
    if model is None:
        if checkpoint is None:
            raise ValueError("需要提供 checkpoint 或已加载的 model")
        model, _ = load_imitation_policy(checkpoint, map_location=device)
    model.train(False)
    dev = torch.device(device)
    n_model = int(model.n_qubits)
    n_circ = int(qc_input.num_qubits)
    if n_circ != n_model:
        raise ValueError(f"线路比特数 {n_circ} 与模型 n_qubits={n_model} 不一致")

    original_cx = sum(
        1 for ins in qc_input.data if (ins.operation.name or "").lower() in ("cx", "cnot")
    )

    p0 = quantum_circuit_to_parity_matrix(qc_input).to(dev)
    edge_index = model.edge_index.to(dev)

    _hist, actions, reached = rollout_until_identity(model, edge_index, p0, dev, max_steps)
    qc_out = parity_actions_to_quantum_circuit(actions, edge_index, n_model)
    return qc_out, p0.detach().cpu(), actions, reached, original_cx, model.edge_index.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qiskit CNOT 线路 → parity → 策略重写 → Qiskit")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default="cpu", help="cuda 或 cpu")
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--qasm", type=Path, default=None, help="OpenQASM 2.x 文件路径")
    parser.add_argument(
        "--cx-chain",
        type=str,
        default=None,
        help='无文件时由内构建线路，例如 "0,1;1,2;2,1"',
    )
    parser.add_argument("--num-qubits", type=int, default=None, help="与 --cx-chain 共用；默认从模型读取")
    parser.add_argument("--out-qasm", type=Path, default=None, help="输出重写后的 OpenQASM 路径")
    args = parser.parse_args()

    _require_qiskit()

    model, _ = load_imitation_policy(args.checkpoint, map_location=args.device)
    n_model = int(model.n_qubits)
    n_qubits = args.num_qubits if args.num_qubits is not None else n_model
    if n_qubits != n_model:
        raise SystemExit(f"--num-qubits={n_qubits} 与 checkpoint 中 n_qubits={n_model} 不一致")

    if args.qasm is not None:
        qc_in = load_quantum_circuit_from_qasm(Path(args.qasm))
    elif args.cx_chain:
        qc_in = build_circuit_from_cx_chain(args.cx_chain, n_qubits)
    else:
        print("请指定 --qasm 或 --cx-chain", file=sys.stderr)
        sys.exit(2)

    qc_out, p_start, actions, reached, n_orig, ei = run_optimize(
        model=model,
        device=args.device,
        max_steps=args.max_steps,
        qc_input=qc_in,
    )
    print(f"original CX gates (count): {n_orig}")
    print(f"start parity matrix shape: {tuple(p_start.shape)}  identity={parity_is_identity(p_start)}")
    print(f"rewritten CX gates (count): {len(actions)}  reached_identity={reached}")
    if actions:
        lbl = [edge_action_label(ei, k) for k in actions[:16]]
        tail = " ..." if len(actions) > 16 else ""
        print(f"first actions (edge labels): {lbl}{tail}")

    print("\n--- rewritten QuantumCircuit ---")
    print(qc_out)

    if args.out_qasm is not None:
        try:
            from qiskit import qasm2

            qasm_text = qasm2.dumps(qc_out)
        except Exception:
            qasm_text = qc_out.qasm()
        args.out_qasm.write_text(qasm_text, encoding="utf-8")
        print(f"\nwrote -> {args.out_qasm}")


if __name__ == "__main__":
    main()
