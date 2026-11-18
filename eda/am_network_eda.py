from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

BASE = Path("data/Student folder/Student folder/train")
OUT = Path("eda")
OUT.mkdir(exist_ok=True)

network_rows = []
connectivity_rows = []

for trip_dir in sorted(BASE.iterdir()):
    if not trip_dir.is_dir():
        continue

    trip_id = trip_dir.name
    node_file = trip_dir / f"{trip_id}.nodes"
    arc_file = trip_dir / f"{trip_id}.arcs"

    if not node_file.exists() or not arc_file.exists():
        print("Skipping incomplete trip:", trip_id)
        continue

    nodes = pd.read_csv(node_file, sep=r"\s+", header=None)
    arcs = pd.read_csv(arc_file, sep=r"\s+", header=None)

    nodes.columns = ["longitude", "latitude"]
    arcs.columns = ["source", "target"]

    n = len(nodes)

    # Coordinate validation
    valid_coords = (
        nodes["longitude"].between(-180, 180)
        & nodes["latitude"].between(-90, 90)
    )

    # Endpoint validation
    valid_endpoints = (
        arcs["source"].between(0, n - 1)
        & arcs["target"].between(0, n - 1)
    )

    valid_arcs = arcs[valid_endpoints].copy()

    # Duplicate arc check
    duplicate_arcs = arcs.duplicated(["source", "target"]).sum()

    # Reverse arc check
    arc_set = set(map(tuple, valid_arcs[["source", "target"]].to_numpy()))
    reverse_count = sum(
        (v, u) in arc_set for u, v in arc_set if u != v
    ) // 2

    # Directed graph
    DG = nx.DiGraph()
    DG.add_nodes_from(range(n))
    DG.add_edges_from(valid_arcs.itertuples(index=False, name=None))

    in_degrees = dict(DG.in_degree())
    out_degrees = dict(DG.out_degree())

    # Undirected connectivity
    G = DG.to_undirected()
    components = list(nx.connected_components(G))
    component_sizes = sorted([len(c) for c in components], reverse=True)

    network_rows.append({
        "trip_id": trip_id,
        "nodes": n,
        "arcs": len(arcs),
        "invalid_coordinates": int((~valid_coords).sum()),
        "duplicate_coordinates": int(nodes.duplicated().sum()),
        "invalid_endpoints": int((~valid_endpoints).sum()),
        "duplicate_arcs": int(duplicate_arcs),
        "self_loops": int((arcs["source"] == arcs["target"]).sum()),
        "reverse_arc_pairs": int(reverse_count),
        "in_degree_mean": np.mean(list(in_degrees.values())),
        "out_degree_mean": np.mean(list(out_degrees.values())),
        "in_degree_max": max(in_degrees.values(), default=0),
        "out_degree_max": max(out_degrees.values(), default=0),
    })

    node_degrees = dict(G.degree())

connectivity_rows.append({
    "trip_id": trip_id,
    "components": len(components),
    "isolated_nodes": sum(d == 0 for d in node_degrees.values()),
    "leaf_nodes": sum(d == 1 for d in node_degrees.values()),
    "largest_component": component_sizes[0] if component_sizes else 0,
    "largest_component_percent": (
        round(component_sizes[0] / n * 100, 2) if n else 0
    ),
    "in_degree_0": sum(d == 0 for d in in_degrees.values()),
    "out_degree_0": sum(d == 0 for d in out_degrees.values()),
})

print(f"Processed trip {trip_id}")

# Save results
network_df = pd.DataFrame(network_rows)
connectivity_df = pd.DataFrame(connectivity_rows)

network_df.to_csv(OUT / "network_summary.csv", index=False)
connectivity_df.to_csv(OUT / "connectivity_summary.csv", index=False)

print("\n========== ALL-TRIP EDA COMPLETE ==========")
print("Trips processed:", len(network_df))
print("\nNetwork summary:")
print(network_df.describe().round(2).to_string())

print("\nConnectivity summary:")
print(connectivity_df.describe().round(2).to_string())

print("\nSaved:")
print("eda/network_summary.csv")
print("eda/connectivity_summary.csv")