import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from simulate.generate import (
    gen_smurfing_ring, gen_layering_ring, gen_round_trip_ring, rng_lognormal_amount,
)
import numpy as np


def _rngs():
    return np.random.default_rng(1), np.random.default_rng(1)


def test_smurfing_ring_amounts_stay_under_threshold():
    rng, py_rng = _rngs()
    rows, members = gen_smurfing_ring(rng, py_rng, "000", [f"A{i}" for i in range(50)], start_hour=0)
    smurf_rows = [r for r in rows if r.tx_type == "TRANSFER"]
    for r in smurf_rows:
        assert r.amount < 10_000, "structuring transfers must stay under the reporting threshold"
        assert r.is_fraud == 1
        assert r.ring_id == "000"
    assert len(members) >= 4  # smurfs + collector


def test_layering_chain_amount_decreases_each_hop():
    rng, py_rng = _rngs()
    rows, chain = gen_layering_ring(rng, py_rng, "001", start_hour=0)
    amounts = [r.amount for r in rows]
    assert all(a2 <= a1 for a1, a2 in zip(amounts, amounts[1:])), "each hop should take a cut"
    assert len(chain) == len(rows) + 1
    assert all(r.is_fraud == 1 and r.typology == "layering" for r in rows)


def test_round_trip_returns_to_origin():
    rng, py_rng = _rngs()
    rows, members = gen_round_trip_ring(rng, py_rng, "002", start_hour=0)
    origs = {r.orig for r in rows}
    dests = {r.dest for r in rows}
    assert origs == dests == set(members), "round-trip money should only move within the ring"


def test_lognormal_amount_respects_cap():
    rng = np.random.default_rng(0)
    amounts = [rng_lognormal_amount(rng, mean=5.7, sigma=1.3, cap=50_000.0) for _ in range(1000)]
    assert max(amounts) <= 50_000.0
    assert min(amounts) > 0
