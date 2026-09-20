"""
RISK//RING transaction simulator.

Generates a synthetic financial-transaction network with a known number of
normal accounts plus a known number of "rings" -- groups of colluding
accounts running one of three FATF-documented laundering typologies:

  - smurfing:    several accounts feed one collector, each just under the
                 reporting threshold, collector cashes out shortly after
  - layering:    a fast chain A->B->C->...  moves money through several
                 hops in minutes/hours to obscure its origin
  - round_trip:  money is sent in a loop back toward its origin through
                 intermediaries, made to look like ordinary business flow

Ground truth (is_fraud, ring_id, typology) is recorded for every
transaction so detection quality can be measured honestly instead of
asserted. Ring members also transact normally with non-ring accounts so
the fraud signal is not trivially separable.

Typology references: FATF, "Money Laundering and Terrorist Financing
Typologies" reports; EU reporting threshold of EUR 10,000 for large cash
transactions (AMLD, Regulation (EU) 2015/847 context).
"""
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass

SEED = 42
N_NORMAL_ACCOUNTS = 2000
N_RINGS = 20
SIM_DAYS = 30
REPORTING_THRESHOLD = 10_000.0
OUT_DIR = "data"


@dataclass
class Tx:
    step: float          # hours since simulation start
    orig: str
    dest: str
    amount: float
    tx_type: str
    is_fraud: int
    ring_id: str
    typology: str


def rng_lognormal_amount(rng, mean=5.7, sigma=1.3, cap=50000.0):
    return float(min(rng.lognormal(mean, sigma), cap))


def gen_normal_traffic(rng, py_rng, accounts, days):
    rows = []
    for day in range(days):
        for acc in accounts:
            n_tx = py_rng.poisson if False else None
        # vectorized-ish: poisson draw per account per day
        counts = rng.poisson(0.3, size=len(accounts))
        for acc, n in zip(accounts, counts):
            for _ in range(n):
                dest = py_rng.choice(accounts)
                if dest == acc:
                    continue
                hour = day * 24 + py_rng.uniform(0, 24)
                amount = rng_lognormal_amount(rng)
                tx_type = py_rng.choice(
                    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
                    p=[0.45, 0.30, 0.15, 0.10],
                )
                rows.append(Tx(hour, acc, dest, round(amount, 2), tx_type, 0, "", ""))
    return rows


def gen_smurfing_ring(rng, py_rng, ring_id, all_accounts, start_hour):
    n_smurfs = py_rng.integers(3, 7)
    smurfs = [f"R{ring_id}_smurf{i}" for i in range(n_smurfs)]
    collector = f"R{ring_id}_collector"
    cashout = py_rng.choice(all_accounts)
    rows = []
    window_start = start_hour
    for s in smurfs:
        amount = py_rng.uniform(8000, 9800)
        hour = window_start + py_rng.uniform(0, 48)
        rows.append(Tx(hour, s, collector, round(amount, 2), "TRANSFER", 1, ring_id, "smurfing"))
    total_in = sum(r.amount for r in rows)
    cashout_hour = window_start + 48 + py_rng.uniform(1, 12)
    rows.append(Tx(cashout_hour, collector, cashout, round(total_in * 0.9, 2), "CASH_OUT", 1, ring_id, "smurfing"))
    return rows, smurfs + [collector]


def gen_layering_ring(rng, py_rng, ring_id, start_hour):
    chain_len = py_rng.integers(4, 7)
    chain = [f"R{ring_id}_hop{i}" for i in range(chain_len)]
    rows = []
    amount = py_rng.uniform(15000, 40000)
    hour = start_hour
    for i in range(chain_len - 1):
        cut = py_rng.uniform(0.02, 0.06)
        next_amount = amount * (1 - cut)
        hour += py_rng.uniform(0.1, 3.0)
        rows.append(Tx(hour, chain[i], chain[i + 1], round(amount, 2), "TRANSFER", 1, ring_id, "layering"))
        amount = next_amount
    return rows, chain


def gen_round_trip_ring(rng, py_rng, ring_id, start_hour):
    n_members = py_rng.integers(3, 5)
    members = [f"R{ring_id}_loop{i}" for i in range(n_members)]
    n_cycles = py_rng.integers(2, 4)
    rows = []
    hour = start_hour
    for _ in range(n_cycles):
        amount = py_rng.uniform(3000, 9000)
        for i in range(n_members):
            dest = members[(i + 1) % n_members]
            hour += py_rng.uniform(0.5, 6.0)
            rows.append(Tx(hour, members[i], dest, round(amount * py_rng.uniform(0.95, 1.0), 2),
                            "TRANSFER", 1, ring_id, "round_trip"))
    return rows, members


def add_camouflage(rng, py_rng, ring_members, normal_accounts, days):
    """Ring members also run a realistic amount of ordinary transaction
    history with non-ring accounts, so their transaction-count / fan-out
    profile looks like a normal account by the time a fraud transaction
    happens -- the fraud signal has to come from timing/amount structure,
    not from 'this account has almost no history'."""
    rows = []
    for m in ring_members:
        n = py_rng.integers(6, 22)
        for _ in range(n):
            other = py_rng.choice(normal_accounts)
            hour = py_rng.uniform(0, days * 24)
            amount = rng_lognormal_amount(rng, mean=5.2, sigma=1.1, cap=8000.0)
            if py_rng.random() < 0.5:
                orig, dest = m, other
            else:
                orig, dest = other, m
            rows.append(Tx(hour, orig, dest, round(amount, 2), "PAYMENT", 0, "", ""))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    py_rng = np.random.default_rng(SEED)

    normal_accounts = [f"A{i:05d}" for i in range(N_NORMAL_ACCOUNTS)]
    all_rows = gen_normal_traffic(rng, py_rng, normal_accounts, SIM_DAYS)

    all_ring_members = []
    typologies = ["smurfing", "layering", "round_trip"]
    for r in range(N_RINGS):
        ring_id = f"{r:03d}"
        typology = typologies[r % 3]
        n_episodes = int(py_rng.integers(3, 8))
        members_this_ring = set()
        for _ in range(n_episodes):
            start_hour = py_rng.uniform(0, SIM_DAYS * 24 - 60)
            if typology == "smurfing":
                rows, members = gen_smurfing_ring(rng, py_rng, ring_id, normal_accounts, start_hour)
            elif typology == "layering":
                rows, members = gen_layering_ring(rng, py_rng, ring_id, start_hour)
            else:
                rows, members = gen_round_trip_ring(rng, py_rng, ring_id, start_hour)
            all_rows.extend(rows)
            members_this_ring.update(members)
        all_ring_members.extend(members_this_ring)
        all_rows.extend(add_camouflage(rng, py_rng, list(members_this_ring), normal_accounts, SIM_DAYS))

    n_benign_structurers = int(len(normal_accounts) * 0.03)
    benign_accs = py_rng.choice(normal_accounts, size=n_benign_structurers, replace=False)
    for acc in benign_accs:
        collector = py_rng.choice(normal_accounts)
        n_parts = int(py_rng.integers(3, 5))
        start_hour = py_rng.uniform(0, SIM_DAYS * 24 - 48)
        for _ in range(n_parts):
            amount = py_rng.uniform(7500, 9900)
            hour = start_hour + py_rng.uniform(0, 48)
            all_rows.append(Tx(hour, acc, collector, round(float(amount), 2), "TRANSFER", 0, "", ""))

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    df = df.sort_values("step").reset_index(drop=True)
    df.insert(0, "tx_id", [f"T{idx:07d}" for idx in range(len(df))])

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/transactions.csv", index=False)

    print(f"Total transactions: {len(df)}")
    print(f"Fraud transactions: {df['is_fraud'].sum()} ({100*df['is_fraud'].mean():.2f}%)")
    print(f"Rings: {df[df.is_fraud==1]['ring_id'].nunique()}")
    print(f"Typology counts:\n{df[df.is_fraud==1]['typology'].value_counts()}")


if __name__ == "__main__":
    main()
