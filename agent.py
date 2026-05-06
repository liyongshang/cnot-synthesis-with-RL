"""
Connectivity-aware CNOT synthesis: GNN policy network (plain PyTorch).

Problem context (GF(2) parity matrix P, hardware graph):
- `edge_index` stores **directed** edges (src, dst). Each row is one admissible
  oriented CNOT: applying src->dst updates P[dst] <- P[dst] XOR P[src].
  If both orientations are allowed on a physical link, store **two** edges (i,j) and (j,i)
  with separate edge features rows.

Why edge features represent CNOT operations:
- A CNOT is not a property of a single qubit; it is an *interaction* on a specific
  graph edge with a *direction*. Edge-wise features (overlap / XOR statistics between
  P[i] and P[j], graph-local signals, etc.) directly describe how that gate would
  act on the current parity state if chosen.

Why XOR / overlap features matter:
- The transition is linear over GF(2): the new row is P[j] XOR P[i]. Overlap
  (e.g., dot product mod 2, Hamming overlap, agreement on pivots) tells whether
  applying the gate would cancel or create parity on certain columns—this is the
  local "progress" signal toward a target parity pattern.

Why the policy must be edge-based (not node-based):
- Actions are *directed* gates on graph edges. With a directed `edge_index`, each
  column is already one candidate direction; the policy outputs **one logit per
  directed edge**, shape [E, 1]. The reverse orientation on the same hardware link is
  a different column (and typically different edge features) if you include it.

This file defines: MLP, EdgeAwareGNNLayer, CNOTPolicyNet (+ optional value head),
and helpers for **target-parity** observations (row encoder + edge init).

Target parity as node input:
- You can stack rows as a matrix T in GF(2)^{N x N}: row i is the target parity vector
  for qubit i. The network expects continuous `node_feat` of shape [N, node_dim].
- You should **not** feed raw bool [N, N] directly unless node_dim == N and you skip GNN
  hidden layers' semantics — typically use a learned linear map `Linear(N, node_dim)`
  (or a small MLP) per row: this *is* your embedding / projection.

Edge features at inference (initialisation):
- Do **not** use random `edge_feat`: overlap / XOR statistics are action-relevant signals.
- Initialise each directed edge (src, dst) from the rows T[src], T[dst] (and during RL
  also from current parity P if available): e.g. GF(2) inner product mod 2, Hamming
  distance, means — then map to `edge_dim` with a small `Linear` (see `ParityEdgeFeatEncoder`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple 2-layer MLP: Linear -> ReLU -> Linear.

    Shapes:
        Input:  [*, in_dim]
        Output: [*, out_dim]
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gf2_dot(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Column-wise GF(2) inner product: <a,b>_2 in {0,1}, returned as float.

    Shapes:
        a, b: [..., N] with values in {0,1} (as float or bool)
        returns: [...] same batch shape
    """
    return ((a.to(torch.float32) * b.to(torch.float32)).long().sum(dim=-1) % 2).float()


class ParityRowEncoder(nn.Module):
    """
    Embed each qubit's **target** parity row (length N) into `node_dim` channels.

    Input:
        target_parity: [N, N] — row i is the target parity vector for qubit i (bool / 0-1).
    Output:
        node_feat: [N, node_dim]

    This is the recommended front-end when observations are per-row target strings.
    During training you may concatenate extra channels (e.g. distance to target from
    current P) by widening `in_dim` or concat before an extra Linear.
    """

    def __init__(self, n_qubits: int, node_dim: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.node_dim = node_dim
        self.proj = nn.Linear(n_qubits, node_dim)

    def forward(self, target_parity: torch.Tensor) -> torch.Tensor:
        if target_parity.dim() != 2 or target_parity.shape[0] != self.n_qubits:
            raise ValueError(
                f"target_parity must be [N, N] with N={self.n_qubits}, got {tuple(target_parity.shape)}"
            )
        x = target_parity.to(dtype=torch.float32, device=self.proj.weight.device)
        return self.proj(x)


class ParityEdgeFeatEncoder(nn.Module):
    """
    Build **initial** edge_feat [E, edge_dim] from parity rows along each directed edge.

    Raw features per edge (src, dst) — all interpretable for CNOT / GF(2) reasoning:

    From **target** rows T[src], T[dst] only (inference with target-only observation):
        - gf2 dot, Hamming distance, row densities (see `_raw_from_two_rows`).

    If `current_parity` is provided (typical RL step t):
        - Adds progress signals: Hamming(P[k], T[k]) at endpoints, and P[src]/P[dst] overlap.

    The stack is projected with `Linear(raw_dim, edge_dim)` so `edge_dim` is free.

    Shapes:
        target_parity, current_parity: [N, N] or None for current
        edge_index: [2, E]
        returns: edge_feat [E, edge_dim]
    """

    def __init__(self, n_qubits: int, edge_dim: int, raw_dim: int = 8) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.raw_dim = raw_dim
        self.proj = nn.Linear(raw_dim, edge_dim)

    @staticmethod
    def _ham_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Mean Hamming distance per row pair, shape [...]."""
        return (a.to(torch.float32) != b.to(torch.float32)).float().mean(dim=-1)

    def _raw_edge_vectors(
        self,
        target_parity: torch.Tensor,
        edge_index: torch.Tensor,
        current_parity: torch.Tensor | None,
    ) -> torch.Tensor:
        """Returns [E, raw_dim] float features (pad/truncate to raw_dim)."""
        src = edge_index[0].long()
        dst = edge_index[1].long()
        t = target_parity.to(torch.float32)
        Ts, Td = t[src], t[dst]  # [E, N]

        raw_list = [
            gf2_dot(Ts, Td),
            self._ham_norm(Ts, Td),
            Ts.mean(dim=-1),
            Td.mean(dim=-1),
        ]

        if current_parity is not None:
            p = current_parity.to(torch.float32)
            Ps, Pd = p[src], p[dst]
            raw_list.extend(
                [
                    self._ham_norm(Ps, Ts),
                    self._ham_norm(Pd, Td),
                    gf2_dot(Ps, Pd),
                    self._ham_norm(Ps, Pd),
                ]
            )
        else:
            raw_list.extend(
                [
                    torch.zeros(edge_index.size(1), device=t.device, dtype=torch.float32)
                    for _ in range(4)
                ]
            )

        raw = torch.stack(raw_list, dim=-1)  # [E, 8] when len(raw_list)==8
        if raw.size(-1) < self.raw_dim:
            pad = torch.zeros(
                raw.size(0),
                self.raw_dim - raw.size(-1),
                device=raw.device,
                dtype=raw.dtype,
            )
            raw = torch.cat([raw, pad], dim=-1)
        elif raw.size(-1) > self.raw_dim:
            raw = raw[..., : self.raw_dim]
        return raw

    def forward(
        self,
        target_parity: torch.Tensor,
        edge_index: torch.Tensor,
        current_parity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_parity.shape != (self.n_qubits, self.n_qubits):
            raise ValueError(
                f"target_parity must be [{self.n_qubits}, {self.n_qubits}], got {tuple(target_parity.shape)}"
            )
        if current_parity is not None and current_parity.shape != target_parity.shape:
            raise ValueError("current_parity must match target_parity shape when provided.")
        raw = self._raw_edge_vectors(target_parity, edge_index, current_parity)
        return self.proj(raw)


class EdgeAwareGNNLayer(nn.Module):
    """
    One layer of edge-aware message passing (no PyG), using `index_add_` aggregation.

    Notation: **directed** edges only:
        src = edge_index[0, e], dst = edge_index[1, e]  # edge e: src -> dst
    Edge tensor e_ij refers to that directed link; h_src, h_dst are endpoint embeddings.

    (A) Edge update — fuse endpoints and current edge embedding:
        e_ij <- MLP_edge( concat( h_src, h_dst, e_ij ) )
        Shapes: [E, 2*node_dim + edge_dim] -> [E, edge_dim]

    (B) Message passing — **unidirectional** along each directed edge (src -> dst):
        m_{src->dst} = MLP_msg( concat( h_src, e_ij ) )
        Aggregate **only** into the head node dst:
            agg[dst] += m_{src->dst}
        So information flows one way along stored edge orientation (matches typical
        directed GNN semantics). No scatter back to src.

    (C) Node update:
        h_v <- MLP_node( concat( h_v, agg_v ) )
        Shapes: h: [N, node_dim], agg: [N, node_dim] -> [N, node_dim]
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = MLP(2 * node_dim + edge_dim, hidden_dim, edge_dim)
        self.msg_mlp = MLP(node_dim + edge_dim, hidden_dim, node_dim)
        self.node_mlp = MLP(2 * node_dim, hidden_dim, node_dim)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feat: [N, node_dim]
            edge_index: [2, E] — each column is one **directed** edge (src -> dst)
            edge_feat: [E, edge_dim]

        Returns:
            node_feat_out: [N, node_dim]
            edge_feat_out: [E, edge_dim]
        """
        src = edge_index[0].long()  # [E]
        dst = edge_index[1].long()  # [E]

        h_src = node_feat[src]  # [E, node_dim]
        h_dst = node_feat[dst]  # [E, node_dim]

        # --- (A) Edge update: incorporate both endpoints into the edge embedding ---
        edge_in = torch.cat([h_src, h_dst, edge_feat], dim=-1)  # [E, 2*node_dim + edge_dim]
        edge_feat = self.edge_mlp(edge_in)  # [E, edge_dim]

        # --- (B) Unidirectional message: along src -> dst, aggregate only at dst ---
        msg = self.msg_mlp(torch.cat([h_src, edge_feat], dim=-1))  # [E, node_dim]
        agg = torch.zeros_like(node_feat)  # [N, node_dim]
        agg.index_add_(0, dst, msg)

        # --- (C) Node update ---
        node_in = torch.cat([node_feat, agg], dim=-1)  # [N, 2*node_dim]
        node_feat = self.node_mlp(node_in)  # [N, node_dim]

        return node_feat, edge_feat


class CNOTPolicyNet(nn.Module):
    """
    Edge-centric GNN policy (and optional value) for CNOT placement on a graph.

    Stack `num_layers` EdgeAwareGNNLayers (typical depth 2–4), then:

    Policy head (edge-centric, **directed** edge list):
        Each row of `edge_index` is one oriented gate src->dst. Output **one** logit
        for that directed candidate:
            logits[e] = score of applying CNOT(src -> dst) on edge e
        Final tensor: logits [E, 1]

    Value head (optional):
        Global mean pool over node embeddings -> scalar V(s).

    Shapes:
        node_feat: [N, node_dim]
        edge_index: [2, E] — directed edges
        edge_feat: [E, edge_dim]
        logits: [E, 1]
        value (if enabled): [1]
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_value_head: bool = True,
    ) -> None:
        super().__init__()
        assert 2 <= num_layers <= 4, "Recommended depth: 2–4 layers."
        self.use_value_head = use_value_head

        self.gnn_layers = nn.ModuleList(
            [EdgeAwareGNNLayer(node_dim, edge_dim, hidden_dim) for _ in range(num_layers)]
        )

        # One score per directed edge (same orientation as edge_index rows).
        self.policy_mlp = MLP(2 * node_dim + edge_dim, hidden_dim, 1)

        if use_value_head:
            self.value_mlp = MLP(node_dim, hidden_dim, 1)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [E, 1] if use_value_head is False
            (logits, value) if use_value_head is True, value shape [1]
        """
        h = node_feat
        ef = edge_feat
        for layer in self.gnn_layers:
            h, ef = layer(h, edge_index, ef)

        src = edge_index[0].long()
        dst = edge_index[1].long()
        h_i = h[src]  # [E, node_dim]
        h_j = h[dst]  # [E, node_dim]

        pol_in = torch.cat([h_i, h_j, ef], dim=-1)  # [E, 2*node_dim + edge_dim]
        logits = self.policy_mlp(pol_in)  # [E, 1] — CNOT(src -> dst) for each row

        if not self.use_value_head:
            return logits

        # Global mean pooling over qubits: permutation-invariant graph summary.
        pooled = h.mean(dim=0)  # [node_dim]
        value = self.value_mlp(pooled)  # [1]
        return logits, value


def _demo() -> None:
    """Minimal sanity check: forward pass, shapes, and gradient flow."""
    torch.manual_seed(0)
    n, e = 5, 6
    node_dim, edge_dim = 8, 4
    hidden_dim = 32
    num_layers = 3

    # Random simple graph: 6 edges on 5 nodes (example only)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 0], [1, 2, 3, 4, 4, 4]],
        dtype=torch.long,
    )
    assert edge_index.shape == (2, e)

    node_feat = torch.randn(n, node_dim)
    edge_feat = torch.randn(e, edge_dim)

    model = CNOTPolicyNet(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_value_head=True,
    )

    logits, value = model(node_feat, edge_index, edge_feat)
    assert logits.shape == (e, 1), f"expected logits [E,1], got {tuple(logits.shape)}"
    assert value.shape == (1,), f"expected value [1], got {tuple(value.shape)}"

    loss = logits.sum() + value.sum()
    loss.backward()
    print("CNOTPolicyNet demo OK.")
    print("  logits:", tuple(logits.shape), " value:", tuple(value.shape))


def _demo_parity_pipeline() -> None:
    """Target-parity rows -> node_feat / edge_feat -> policy forward."""
    torch.manual_seed(1)
    n, e = 5, 6
    node_dim, edge_dim = 8, 4
    hidden_dim = 32
    num_layers = 3

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 0], [1, 2, 3, 4, 4, 4]],
        dtype=torch.long,
    )
    target_parity = torch.randint(0, 2, (n, n), dtype=torch.float32)
    current_parity = torch.randint(0, 2, (n, n), dtype=torch.float32)

    row_enc = ParityRowEncoder(n, node_dim)
    edge_enc = ParityEdgeFeatEncoder(n, edge_dim, raw_dim=8)

    node_feat = row_enc(target_parity)
    edge_feat_tgt_only = edge_enc(target_parity, edge_index, current_parity=None)
    edge_feat_with_p = edge_enc(target_parity, edge_index, current_parity=current_parity)

    model = CNOTPolicyNet(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_value_head=True,
    )

    logits1, _ = model(node_feat, edge_index, edge_feat_tgt_only)
    logits2, _ = model(node_feat, edge_index, edge_feat_with_p)
    assert logits1.shape == (e, 1) and logits2.shape == (e, 1)
    (logits1.sum() + logits2.sum()).backward()
    print("Parity pipeline demo OK.")
    print("  node_feat:", tuple(node_feat.shape), " edge_feat:", tuple(edge_feat_with_p.shape))


if __name__ == "__main__":
    _demo()
    _demo_parity_pipeline()
