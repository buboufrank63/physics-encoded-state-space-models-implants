from collections import deque

import torch
from torch import Tensor


def element_adjacency(elements: Tensor) -> list[list[int]]:
    if elements.ndim != 2:
        raise ValueError("elements must have shape [elements, nodes]")
    node_to_elements: dict[int, list[int]] = {}
    for element_index, element in enumerate(elements.tolist()):
        for node in element:
            node_to_elements.setdefault(int(node), []).append(element_index)
    adjacency: list[set[int]] = [set() for _ in range(elements.shape[0])]
    for incident in node_to_elements.values():
        for source in incident:
            adjacency[source].update(target for target in incident if target != source)
    return [sorted(neighbors) for neighbors in adjacency]


def boundary_seed_elements(elements: Tensor, constrained_nodes: Tensor) -> list[int]:
    constrained = set(torch.nonzero(constrained_nodes, as_tuple=False).flatten().tolist())
    seeds = []
    for index, element in enumerate(elements.tolist()):
        if any(node in constrained for node in element):
            seeds.append(index)
    return seeds


def breadth_first_order(adjacency: list[list[int]], seeds: list[int]) -> Tensor:
    count = len(adjacency)
    visited = [False] * count
    queue: deque[int] = deque()
    for seed in seeds:
        if 0 <= seed < count and not visited[seed]:
            queue.append(seed)
            visited[seed] = True
    output = []
    while queue:
        current = queue.popleft()
        output.append(current)
        for neighbor in adjacency[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    for index in range(count):
        if not visited[index]:
            output.append(index)
    return torch.tensor(output, dtype=torch.long)


def load_path_order(elements: Tensor, constraints: Tensor) -> Tensor:
    adjacency = element_adjacency(elements)
    seeds = boundary_seed_elements(elements, constraints)
    if not seeds:
        seeds = [0]
    return breadth_first_order(adjacency, seeds)


def multiple_orders(elements: Tensor, constraints: Tensor, count: int) -> Tensor:
    primary = load_path_order(elements, constraints)
    orders = [primary]
    if count > 1:
        orders.append(primary.flip(0))
    for shift in range(1, max(1, count - 1)):
        orders.append(torch.roll(primary, shifts=shift, dims=0))
    return torch.stack(orders[:count], dim=0)


def invert_order(order: Tensor) -> Tensor:
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    return inverse


def reorder(sequence: Tensor, order: Tensor) -> Tensor:
    return sequence.index_select(-2, order.to(sequence.device))


def restore(sequence: Tensor, order: Tensor) -> Tensor:
    return sequence.index_select(-2, invert_order(order).to(sequence.device))
