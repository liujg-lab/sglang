"""Variable-width TollTree. Path mass does not increase along an edge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TollNode:
    token: int
    pi: float
    parent: int
    depth: int
    draft_a: float


@dataclass(frozen=True)
class TollVerify:
    """Ancestor-closed prefix of the path-mass ranking.

    ``parents[i] == -1`` is the committed tail. Every other parent index is
    strictly smaller than ``i``.
    """

    tokens: tuple[int, ...]
    parents: tuple[int, ...]
    pis: tuple[float, ...]
    depths: tuple[int, ...]


class TollTree:
    """One request's draft nodes.

    ``add_child`` rejects a conditional above 1, so a child's path mass cannot
    exceed its parent. ``finish`` ranks by path mass, then shallower depth, so
    every prefix is ancestor-closed. An exited request accepts no new nodes.
    """

    def __init__(self, prefix_len: int) -> None:
        if int(prefix_len) < 0:
            raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")
        self.prefix_len = int(prefix_len)
        self.nodes: list[TollNode] = []
        self.frontier: list[int] = []
        self.exited = False

    def mark_exited(self) -> None:
        self.exited = True
        self.frontier = []

    def add_child(self, parent: int, token: int, a: float) -> int:
        if self.exited:
            raise RuntimeError("exited request cannot gain nodes")
        cond = float(a)
        if cond < 0.0 or cond > 1.0:
            raise ValueError(f"conditional a must be in [0, 1], got {a}")
        if int(parent) == -1:
            pi = cond
            depth = 1
        else:
            if int(parent) < 0 or int(parent) >= len(self.nodes):
                raise ValueError(f"parent {parent} is outside the tree")
            host = self.nodes[int(parent)]
            pi = float(host.pi) * cond
            if pi > float(host.pi) + 1e-12:
                raise ValueError("path probability increased along an edge")
            depth = int(host.depth) + 1
        self.nodes.append(
            TollNode(
                token=int(token),
                pi=pi,
                parent=int(parent),
                depth=depth,
                draft_a=cond,
            )
        )
        return len(self.nodes) - 1

    def set_frontier(self, indices) -> None:
        if self.exited:
            raise RuntimeError("exited request cannot gain nodes")
        self.frontier = [int(i) for i in indices]
        for index in self.frontier:
            if index < 0 or index >= len(self.nodes):
                raise ValueError(f"frontier index {index} is outside the tree")

    def ranked_indices(self) -> list[int]:
        """Higher path mass first. Shallower depth breaks ties, then index."""
        order = list(range(len(self.nodes)))
        order.sort(key=lambda i: (-float(self.nodes[i].pi), int(self.nodes[i].depth), i))
        return order

    def finish(self, k: int) -> TollVerify:
        """Keep the first ``k`` ranked nodes. The prefix includes its ancestors."""
        if int(k) < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        chosen = self.ranked_indices()[: min(int(k), len(self.nodes))]
        chosen_set = set(chosen)
        for index in list(chosen):
            parent = self.nodes[index].parent
            while parent != -1:
                if parent not in chosen_set:
                    raise RuntimeError(
                        f"finish({k}) dropped ancestor {parent} of node {index}"
                    )
                parent = self.nodes[parent].parent
        position = {index: i for i, index in enumerate(chosen)}
        tokens = []
        parents = []
        pis = []
        depths = []
        for index in chosen:
            node = self.nodes[index]
            tokens.append(int(node.token))
            pis.append(float(node.pi))
            depths.append(int(node.depth))
            if node.parent == -1:
                parents.append(-1)
            else:
                parent_at = position[node.parent]
                if parent_at >= position[index]:
                    raise RuntimeError("parent is not before its child in finish order")
                parents.append(parent_at)
        return TollVerify(tuple(tokens), tuple(parents), tuple(pis), tuple(depths))
