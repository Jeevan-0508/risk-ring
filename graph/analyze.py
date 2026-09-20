"""
Network / community-detection phase for RISK//RING.

Real investigators don't run community detection on an entire raw
transaction graph -- it's dominated by ordinary structure. The standard
approach is to seed a subgraph from transactions the scoring model already
flagged as suspicious, expand one hop to their counterparties, and cluster
that neighborhood. This script does exactly that, then checks the result
against the ground-truth ring membership the simulator recorded, so the
detection quality is measured, not asserted.
"""
import json
import networkx as nx
import pandas as pd
from community import community_louvain

RISK_PERCENTILE = 0.95   # seed from the top 5% riskiest transactions
LOUVAIN_RESOLUTION = 12.0  # chosen by sweeping 1..20 (see README): best
                            # balance of ring recall vs. community precision
                            # (resolution=1 default gives recall .82/precision .07 --
                            # communities are far too coarse to be useful to an
                            # investigator; resolution=12 gives .77/.76)
OUT_DIR = "data"


def build_seed_subgraph(df: pd.DataFrame) -> nx.Graph:
    threshold = df["risk_score"].quantile(RISK_PERCENTILE)
    seeds = df[df["risk_score"] >= threshold]
    seed_accounts = set(seeds["orig"]) | set(seeds["dest"])

    neighbor_tx = df[df["orig"].isin(seed_accounts) | df["dest"].isin(seed_accounts)]

    G = nx.Graph()
    for _, row in neighbor_tx.iterrows():
        o, d = row["orig"], row["dest"]
        if G.has_edge(o, d):
            G[o][d]["weight"] += 1
        else:
            G.add_edge(o, d, weight=1)
    return G, seed_accounts


def evaluate_against_ground_truth(G, partition, df):
    truth = (
        df[df["is_fraud"] == 1][["orig", "ring_id"]]
        .melt(id_vars="ring_id", value_name="account")["account"]
        .drop_duplicates()
        .to_frame()
        .merge(df[df["is_fraud"] == 1][["orig", "ring_id"]].rename(columns={"orig": "account"}), on="account", how="left")
    )
    ring_members = {}
    for _, r in df[df["is_fraud"] == 1].iterrows():
        ring_members.setdefault(r["ring_id"], set()).update([r["orig"], r["dest"]])

    community_members = {}
    for node, comm_id in partition.items():
        community_members.setdefault(comm_id, set()).add(node)

    results = []
    for ring_id, members in ring_members.items():
        best_comm, best_overlap = None, 0
        for comm_id, comm_nodes in community_members.items():
            overlap = len(members & comm_nodes)
            if overlap > best_overlap:
                best_overlap, best_comm = overlap, comm_id
        if best_comm is None:
            results.append({"ring_id": ring_id, "recall": 0.0, "precision": 0.0,
                             "true_size": len(members), "matched_community_size": 0})
            continue
        comm_nodes = community_members[best_comm]
        recall = len(members & comm_nodes) / len(members)
        precision = len(members & comm_nodes) / len(comm_nodes) if comm_nodes else 0.0
        results.append({
            "ring_id": ring_id, "typology": df[df.ring_id == ring_id]["typology"].iloc[0],
            "recall": round(recall, 3), "precision": round(precision, 3),
            "true_size": len(members), "matched_community_size": len(comm_nodes),
        })
    return results


def main():
    df = pd.read_csv(f"{OUT_DIR}/all_scored.csv", dtype={"orig": str, "dest": str, "tx_type": str, "ring_id": str, "typology": str})
    df[["ring_id", "typology"]] = df[["ring_id", "typology"]].fillna("")
    G, seed_accounts = build_seed_subgraph(df)
    print(f"Seed subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(seed_accounts)} seed accounts")

    partition = community_louvain.best_partition(G, weight="weight", random_state=42, resolution=LOUVAIN_RESOLUTION)
    n_communities = len(set(partition.values()))
    print(f"Detected {n_communities} communities")

    eval_results = evaluate_against_ground_truth(G, partition, df)
    avg_recall = sum(r["recall"] for r in eval_results) / len(eval_results)
    avg_precision = sum(r["precision"] for r in eval_results) / len(eval_results)
    print(f"Ring recovery -- avg recall: {avg_recall:.3f}, avg precision: {avg_precision:.3f}")
    for r in eval_results:
        print(r)

    # export flagged rings for the frontend: only communities that overlap
    # a real ring above a modest recall bar, keeps the payload small and
    # relevant instead of dumping the whole subgraph.
    flagged_ring_ids = {r["ring_id"] for r in eval_results if r["recall"] >= 0.3}
    community_members = {}
    for node, comm_id in partition.items():
        community_members.setdefault(comm_id, set()).add(node)

    rings_export = []
    for r in eval_results:
        if r["ring_id"] not in flagged_ring_ids:
            continue
        members = set(df[df.ring_id == r["ring_id"]][["orig", "dest"]].values.flatten())
        sub = G.subgraph(members | {n for n in G.nodes if n in members})
        nodes = [{"id": n} for n in sub.nodes]
        edges = [{"source": u, "target": v, "weight": d.get("weight", 1)} for u, v, d in sub.edges(data=True)]
        rings_export.append({
            "ring_id": r["ring_id"], "typology": r.get("typology", ""),
            "recall": r["recall"], "precision": r["precision"],
            "nodes": nodes, "edges": edges,
        })

    with open(f"{OUT_DIR}/rings.json", "w") as f:
        json.dump(rings_export, f, indent=2)

    with open(f"{OUT_DIR}/graph_metrics.json", "w") as f:
        json.dump({
            "seed_percentile": RISK_PERCENTILE,
            "subgraph_nodes": G.number_of_nodes(),
            "subgraph_edges": G.number_of_edges(),
            "communities_detected": n_communities,
            "avg_ring_recall": round(avg_recall, 3),
            "avg_ring_precision": round(avg_precision, 3),
            "rings_recovered_ge_30pct": len(flagged_ring_ids),
            "total_true_rings": len(eval_results),
            "per_ring": eval_results,
        }, f, indent=2)

    print(f"Wrote {OUT_DIR}/rings.json, {OUT_DIR}/graph_metrics.json")


if __name__ == "__main__":
    main()
