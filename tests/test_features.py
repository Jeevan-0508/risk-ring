import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from model.train import engineer_features, REPORTING_THRESHOLD


def _toy_df():
    return pd.DataFrame([
        {"tx_id": "T0", "step": 0.0, "orig": "A", "dest": "B", "amount": 100.0,
         "tx_type": "PAYMENT", "is_fraud": 0, "ring_id": "", "typology": ""},
        {"tx_id": "T1", "step": 1.0, "orig": "A", "dest": "B", "amount": 8500.0,
         "tx_type": "TRANSFER", "is_fraud": 1, "ring_id": "000", "typology": "smurfing"},
        {"tx_id": "T2", "step": 2.0, "orig": "A", "dest": "C", "amount": 50.0,
         "tx_type": "PAYMENT", "is_fraud": 0, "ring_id": "", "typology": ""},
    ])


def test_near_threshold_flag_only_fires_for_structuring_range():
    df = engineer_features(_toy_df())
    assert df.loc[df.tx_id == "T1", "near_threshold"].iloc[0] == 1
    assert df.loc[df.tx_id == "T0", "near_threshold"].iloc[0] == 0


def test_prior_counts_are_causal_not_future_leaking():
    """The 3rd transaction from A (T2) should see 2 prior orig transactions;
    the 1st (T0) must see zero -- features must never see the future."""
    df = engineer_features(_toy_df())
    assert df.loc[df.tx_id == "T0", "orig_tx_count_prior"].iloc[0] == 0
    assert df.loc[df.tx_id == "T1", "orig_tx_count_prior"].iloc[0] == 1
    assert df.loc[df.tx_id == "T2", "orig_tx_count_prior"].iloc[0] == 2


def test_first_time_pair_flag():
    df = engineer_features(_toy_df())
    assert df.loc[df.tx_id == "T0", "is_first_time_pair"].iloc[0] == 1
    assert df.loc[df.tx_id == "T1", "is_first_time_pair"].iloc[0] == 0  # A->B seen at T0
    assert df.loc[df.tx_id == "T2", "is_first_time_pair"].iloc[0] == 1  # A->C is new
