import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from graph.analyze import evaluate_against_ground_truth


def test_perfect_partition_scores_recall_and_precision_1():
    df = pd.DataFrame([
        {"orig": "R1", "dest": "R2", "is_fraud": 1, "ring_id": "000", "typology": "layering"},
        {"orig": "R2", "dest": "R3", "is_fraud": 1, "ring_id": "000", "typology": "layering"},
    ])
    partition = {"R1": 0, "R2": 0, "R3": 0}
    results = evaluate_against_ground_truth(None, partition, df)
    assert len(results) == 1
    assert results[0]["recall"] == 1.0
    assert results[0]["precision"] == 1.0


def test_noisy_partition_lowers_precision_not_recall():
    df = pd.DataFrame([
        {"orig": "R1", "dest": "R2", "is_fraud": 1, "ring_id": "000", "typology": "layering"},
    ])
    # community 0 correctly contains both ring members, plus 2 unrelated normal accounts
    partition = {"R1": 0, "R2": 0, "N1": 0, "N2": 0}
    results = evaluate_against_ground_truth(None, partition, df)
    assert results[0]["recall"] == 1.0
    assert results[0]["precision"] == 0.5  # 2 of 4 community members are actually in the ring


def test_no_matching_community_scores_zero():
    df = pd.DataFrame([
        {"orig": "R1", "dest": "R2", "is_fraud": 1, "ring_id": "000", "typology": "layering"},
    ])
    partition = {"R1": 0, "N1": 1}  # R2 isn't even in the partition (dropped from subgraph)
    results = evaluate_against_ground_truth(None, partition, df)
    assert results[0]["recall"] == 0.5  # only R1 recovered, R2 missing entirely
