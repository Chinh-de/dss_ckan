import collections
from typing import List, Dict, Tuple, Any
import numpy as np
import torch

def build_kg_dict(kg_np: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    kg_dict = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg_dict[int(head)].append((int(tail), int(relation)))
    return kg_dict

def generate_user_triple_set(
    liked_items: List[int],
    kg_dict: Dict[int, List[Tuple[int, int]]],
    n_layer: int = 1,
    set_size: int = 32,
    fallback_items: List[int] = None
) -> List[Tuple[List[int], List[int], List[int]]]:
    if not liked_items:
        liked_items = fallback_items or [0]

    triple_set = []
    entities = liked_items

    for l in range(n_layer):
        h, r, t = [], [], []
        for entity in entities:
            neighbors = kg_dict.get(entity, [])
            for tail, rel in neighbors:
                h.append(entity)
                r.append(rel)
                t.append(tail)

        if len(h) == 0:
            # Fallback if no neighbors found for these entities
            if triple_set:
                triple_set.append(triple_set[-1])
            else:
                dummy_h = [entities[0]] * set_size
                dummy_r = [0] * set_size
                dummy_t = [entities[0]] * set_size
                triple_set.append((dummy_h, dummy_r, dummy_t))
        else:
            replace = len(h) < set_size
            indices = np.random.choice(len(h), size=set_size, replace=replace)
            h_sampled = [h[i] for i in indices]
            r_sampled = [r[i] for i in indices]
            t_sampled = [t[i] for i in indices]
            triple_set.append((h_sampled, r_sampled, t_sampled))
            entities = t_sampled

    return triple_set

def get_triple_tensor(n_layer: int, objs: List[int], triple_set_map: Dict[int, Any], device: torch.device):
    h, r, t = [], [], []
    for i in range(n_layer):
        h_layer = [triple_set_map[obj][i][0] for obj in objs]
        r_layer = [triple_set_map[obj][i][1] for obj in objs]
        t_layer = [triple_set_map[obj][i][2] for obj in objs]
        h.append(torch.LongTensor(h_layer).to(device))
        r.append(torch.LongTensor(r_layer).to(device))
        t.append(torch.LongTensor(t_layer).to(device))
    return [h, r, t]

