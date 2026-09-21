import pytest
import numpy as np
from app.recommendation.dynamic_propagation import build_kg_dict, generate_user_triple_set

def test_kg_dict_and_dynamic_propagation():
    # Synthetic small KG: (h, r, t)
    sample_kg = np.array([
        [100, 0, 1001], # Movie 100 -> directed_by -> Director 1001
        [100, 1, 2001], # Movie 100 -> has_genre -> Sci-Fi 2001
        [101, 0, 1001], # Movie 101 -> directed_by -> Director 1001
        [102, 1, 2001], # Movie 102 -> has_genre -> Sci-Fi 2001
    ], dtype=np.int32)

    kg_dict = build_kg_dict(sample_kg)
    assert 100 in kg_dict
    assert len(kg_dict[100]) == 2

    # Dynamic generation for a brand new user who liked movie 100
    user_ts = generate_user_triple_set(liked_items=[100], kg_dict=kg_dict, n_layer=1, set_size=16)
    assert len(user_ts) == 1
    h, r, t = user_ts[0]
    assert len(h) == 16
    assert all(head == 100 for head in h)
    assert all(rel in (0, 1) for rel in r)
    assert all(tail in (1001, 2001) for tail in t)
    print("test_kg_dict_and_dynamic_propagation passed!")

def test_empty_user_fallback():
    sample_kg = np.array([[0, 0, 0]], dtype=np.int32)
    kg_dict = build_kg_dict(sample_kg)
    user_ts = generate_user_triple_set(liked_items=[], kg_dict=kg_dict, n_layer=1, set_size=8)
    assert len(user_ts) == 1
    assert len(user_ts[0][0]) == 8
    print("test_empty_user_fallback passed!")

