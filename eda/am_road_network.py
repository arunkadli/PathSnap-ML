from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

BASE = Path("data/Student folder/Student folder/train")
OUT = Path("eda/figures")
OUT.mkdir(parents=True, exist_ok=True)

trip = "00000001"

node_file = BASE / trip / f"{trip}.nodes"
arc_file = BASE / trip / f"{trip}.arcs"

nodes = pd.read_csv(node_file, sep=r"\s+", header=None, names=["longitude", "latitude"])
arcs = pd.read_csv(arc_file, sep=r"\s+", header=None, names=["source", "target"])

# Basic data checks
print("\n========== ROAD NETWORK EDA ==========")
print("Total nodes:", len(nodes))
print("Total arcs:", len(arcs))
print("Duplicate nodes:", nodes.duplicated().sum())
print("Duplicate arcs:", arcs.duplicated().sum())
print("Missing node values:", nodes.isna().sum().sum())
print("Missing arc values:", arcs.isna().sum().sum())

# Endpoint validation
valid = arcs["source"].between(0, len(nodes)-1) & arcs["target"].between(0, len(nodes)-1)
print("Invalid arc endpoints:", (~valid).sum())

# Degree distribution
G = nx.Graph()
G.add_nodes_from(range(len(nodes)))
G.add_edges_from(arcs[["source", "target"]].itertuples(index=False, name=None))

degree = dict(G.degree())
degree_df = pd.DataFrame({"node": list(degree.keys()), "degree": list(degree.values())})

print("\nDegree Statistics:")
print(degree_df["degree"].describe())
print("Isolated nodes:", (degree_df["degree"] == 0).sum())
print("Leaf nodes (degree 1):", (degree_df["degree"] == 1).sum())

# Connectivity
components = list(nx.connected_components(G))
component_sizes = sorted([len(c) for c in components], reverse=True)

print("\nConnected Components:", len(components))
print("Largest component size:", component_sizes[0])
print("Nodes in largest component (%):", round(component_sizes[0]/len(nodes)*100, 2))
print("Top 10 component sizes:", component_sizes[:10])

# Summary output
summary = pd.DataFrame([{
    "trip_id": trip,
    "nodes": len(nodes),
    "arcs": len(arcs),
    "invalid_endpoints": int((~valid).sum()),
    "isolated_nodes": int((degree_df.degree == 0).sum()),
    "leaf_nodes": int((degree_df.degree == 1).sum()),
    "components": len(components),
    "largest_component": component_sizes[0],
    "largest_component_percent": round(component_sizes[0]/len(nodes)*100, 2),
    "average_degree": degree_df.degree.mean(),
    "max_degree": degree_df.degree.max()
}])

summary.to_csv("eda/amrithaa_network_summary.csv", index=False)
degree_df.to_csv("eda/amrithaa_node_degrees.csv", index=False)

# 1. Node geographic distribution
plt.figure(figsize=(8, 6))
plt.scatter(nodes.longitude, nodes.latitude, s=1, alpha=0.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Road Network Node Geographic Distribution")
plt.tight_layout()
plt.savefig(OUT / "node_geographic_distribution.png", dpi=200)
plt.close()

# 2. Node degree distribution
plt.figure(figsize=(8, 5))
degree_counts = degree_df.degree.value_counts().sort_index()
plt.bar(degree_counts.index, degree_counts.values)
plt.xlabel("Node Degree")
plt.ylabel("Number of Nodes")
plt.title("Road Network Node Degree Distribution")
plt.tight_layout()
plt.savefig(OUT / "node_degree_distribution.png", dpi=200)
plt.close()

# 3. Connected component sizes
plt.figure(figsize=(8, 5))
plt.hist(component_sizes, bins=50)
plt.xlabel("Component Size (Nodes)")
plt.ylabel("Number of Components")
plt.title("Connected Component Size Distribution")
plt.tight_layout()
plt.savefig(OUT / "connected_components.png", dpi=200)
plt.close()

print("\nPlots saved to:", OUT)
print("Summary saved to eda/amrithaa_network_summary.csv")
print("EDA COMPLETE")
print(summary.to_string(index=False))