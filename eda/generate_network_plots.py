"""
generate_network_plots.py
==========================
Amrithaa – Road-network plots deliverable.

Reads network_summary.csv, connectivity_summary.csv, and arc_length_summary.csv
(already in eda/) and produces all required PNG plots into eda/figures/.

Run from the PathSnap-ML/ repo root:
    python eda/generate_network_plots.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150

EDA_DIR = Path("eda")
FIG_DIR = EDA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load summary tables ───────────────────────────────────────────────────────
net = pd.read_csv(EDA_DIR / "network_summary.csv")
conn = pd.read_csv(EDA_DIR / "connectivity_summary.csv")
arc_len = pd.read_csv(EDA_DIR / "arc_length_summary.csv")

print(f"Loaded {len(net)} trips from network_summary.csv")

# ── Merge for convenience ─────────────────────────────────────────────────────
merged = net.merge(conn, on="trip_id", how="left").merge(arc_len, on="trip_id", how="left")
merged["arc_node_ratio"] = merged["arcs"] / merged["nodes"]
merged["isolated_pct"] = merged["isolated_nodes"] / merged["nodes"] * 100
merged["largest_comp_pct"] = merged["largest_component"] / merged["nodes"] * 100

# =============================================================================
# Plot 1 – Node and arc count distributions
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(merged["nodes"], bins=30, edgecolor="k", linewidth=0.4, color="steelblue")
axes[0].set_xlabel("Nodes per trip")
axes[0].set_ylabel("Count of trips")
axes[0].set_title("Node count distribution")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

axes[1].hist(merged["arcs"], bins=30, edgecolor="k", linewidth=0.4, color="coral")
axes[1].set_xlabel("Arcs per trip")
axes[1].set_ylabel("Count of trips")
axes[1].set_title("Arc count distribution")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(FIG_DIR / "plot1_node_arc_counts.png", dpi=150)
plt.close()
print("Saved plot1_node_arc_counts.png")

# =============================================================================
# Plot 2 – Arc-to-node ratio
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(merged["arc_node_ratio"], bins=25, edgecolor="k", linewidth=0.4, color="mediumpurple")
ax.axvline(2.0, color="red", linestyle="--", linewidth=1.2, label="Expected ≈ 2.0")
ax.set_xlabel("Arc-to-node ratio")
ax.set_ylabel("Count of trips")
ax.set_title("Arc-to-node ratio across all trips")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "plot2_arc_node_ratio.png", dpi=150)
plt.close()
print("Saved plot2_arc_node_ratio.png")

# =============================================================================
# Plot 3 – In-degree and out-degree mean distributions
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(merged["in_degree_mean"], bins=25, edgecolor="k", linewidth=0.4, color="salmon")
axes[0].set_xlabel("Mean in-degree per trip")
axes[0].set_ylabel("Count of trips")
axes[0].set_title("Mean in-degree distribution")

axes[1].hist(merged["out_degree_mean"], bins=25, edgecolor="k", linewidth=0.4, color="cornflowerblue")
axes[1].set_xlabel("Mean out-degree per trip")
axes[1].set_ylabel("Count of trips")
axes[1].set_title("Mean out-degree distribution")

plt.tight_layout()
plt.savefig(FIG_DIR / "plot3_degree_distributions.png", dpi=150)
plt.close()
print("Saved plot3_degree_distributions.png")

# =============================================================================
# Plot 4 – Max in-degree and out-degree across trips
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
x = range(len(merged))
ax.bar(x, merged["in_degree_max"], label="Max in-degree", alpha=0.7, color="salmon")
ax.bar(x, merged["out_degree_max"], label="Max out-degree", alpha=0.7, color="steelblue")
ax.set_xlabel("Trip index")
ax.set_ylabel("Max degree")
ax.set_title("Max in/out-degree per trip")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "plot4_max_degree_per_trip.png", dpi=150)
plt.close()
print("Saved plot4_max_degree_per_trip.png")

# =============================================================================
# Plot 5 – Duplicate nodes and arcs
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

dup_nodes = merged["duplicate_coordinates"].sort_values(ascending=False)
dup_arcs  = merged["duplicate_arcs"].sort_values(ascending=False)

axes[0].bar(range(len(dup_nodes)), dup_nodes.values, color="orange", edgecolor="k", linewidth=0.3)
axes[0].set_xlabel("Trip (sorted)")
axes[0].set_ylabel("Duplicate node count")
axes[0].set_title("Duplicate nodes per trip")

axes[1].bar(range(len(dup_arcs)), dup_arcs.values, color="tomato", edgecolor="k", linewidth=0.3)
axes[1].set_xlabel("Trip (sorted)")
axes[1].set_ylabel("Duplicate arc count")
axes[1].set_title("Duplicate arcs per trip")

plt.tight_layout()
plt.savefig(FIG_DIR / "plot5_duplicates.png", dpi=150)
plt.close()
print("Saved plot5_duplicates.png")

# =============================================================================
# Plot 6 – Connected components and isolated nodes
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(merged["components"], bins=30, edgecolor="k", linewidth=0.4, color="teal")
axes[0].set_xlabel("Number of connected components")
axes[0].set_ylabel("Count of trips")
axes[0].set_title("Connected components per trip")

axes[1].hist(merged["isolated_pct"], bins=25, edgecolor="k", linewidth=0.4, color="darkorange")
axes[1].set_xlabel("Isolated nodes (%)")
axes[1].set_ylabel("Count of trips")
axes[1].set_title("Isolated node percentage per trip")

plt.tight_layout()
plt.savefig(FIG_DIR / "plot6_components_isolated.png", dpi=150)
plt.close()
print("Saved plot6_components_isolated.png")

# =============================================================================
# Plot 7 – Arcs vs Nodes scatter (with regression)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(merged["nodes"], merged["arcs"], alpha=0.7, s=25, color="steelblue", edgecolors="k", linewidths=0.3)
m, b = np.polyfit(merged["nodes"], merged["arcs"], 1)
x_line = np.linspace(merged["nodes"].min(), merged["nodes"].max(), 200)
ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5, label=f"y = {m:.2f}x + {b:.0f}")
ax.set_xlabel("Number of nodes")
ax.set_ylabel("Number of arcs")
ax.set_title("Arcs vs Nodes – all trips")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "plot7_arcs_vs_nodes.png", dpi=150)
plt.close()
print("Saved plot7_arcs_vs_nodes.png")

# =============================================================================
# Plot 8 – Arc length statistics across trips
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(arc_len["mean_length_m"], bins=25, edgecolor="k", linewidth=0.4, color="seagreen")
axes[0].set_xlabel("Mean arc length (m)")
axes[0].set_ylabel("Count of trips")
axes[0].set_title("Mean arc length distribution")

axes[1].scatter(range(len(arc_len)), arc_len["total_length_km"].sort_values(ascending=False).values,
                s=15, color="steelblue")
axes[1].set_xlabel("Trip (sorted by total network length)")
axes[1].set_ylabel("Total arc length (km)")
axes[1].set_title("Total network length per trip")
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

plt.tight_layout()
plt.savefig(FIG_DIR / "plot8_arc_length_stats.png", dpi=150)
plt.close()
print("Saved plot8_arc_length_stats.png")

# =============================================================================
# Plot 9 – Reverse arc pairs per trip (heatmap-style bar)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4))
sorted_rev = merged.sort_values("reverse_arc_pairs", ascending=False)
ax.bar(range(len(sorted_rev)), sorted_rev["reverse_arc_pairs"].values,
       color="slateblue", edgecolor="k", linewidth=0.2)
ax.set_xlabel("Trip (sorted by reverse arc count)")
ax.set_ylabel("Reverse arc pairs")
ax.set_title("Bidirectional arc pairs per trip")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(FIG_DIR / "plot9_reverse_arcs.png", dpi=150)
plt.close()
print("Saved plot9_reverse_arcs.png")

# =============================================================================
# Plot 10 – Self-loops and invalid endpoint summary
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sl = merged["self_loops"].sort_values(ascending=False)
axes[0].bar(range(len(sl)), sl.values, color="crimson", edgecolor="k", linewidth=0.3)
axes[0].set_xlabel("Trip (sorted)")
axes[0].set_ylabel("Self-loop count")
axes[0].set_title("Self-loops per trip")

ep = merged["invalid_endpoints"].sort_values(ascending=False)
axes[1].bar(range(len(ep)), ep.values, color="darkcyan", edgecolor="k", linewidth=0.3)
axes[1].set_xlabel("Trip (sorted)")
axes[1].set_ylabel("Invalid endpoint count")
axes[1].set_title("Invalid arc endpoints per trip")

plt.tight_layout()
plt.savefig(FIG_DIR / "plot10_selfloops_invalid_ep.png", dpi=150)
plt.close()
print("Saved plot10_selfloops_invalid_ep.png")

# =============================================================================
# Plot 11 – Cleaning summary: nodes and arcs removed
# =============================================================================
clean_path = EDA_DIR / "gps_cleaning_summary.csv"
if clean_path.exists():
    clean = pd.read_csv(clean_path)
    clean["nodes_removed"] = clean["n_nodes_raw"] - clean["n_nodes_clean"]
    clean["arcs_removed"]  = clean["n_arcs_raw"]  - clean["n_arcs_clean"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(len(clean)),
                clean["nodes_removed"].sort_values(ascending=False).values,
                color="orange", edgecolor="k", linewidth=0.3)
    axes[0].set_xlabel("Trip (sorted)")
    axes[0].set_ylabel("Nodes removed")
    axes[0].set_title("Nodes removed in cleaning per trip")

    axes[1].bar(range(len(clean)),
                clean["arcs_removed"].sort_values(ascending=False).values,
                color="tomato", edgecolor="k", linewidth=0.3)
    axes[1].set_xlabel("Trip (sorted)")
    axes[1].set_ylabel("Arcs removed")
    axes[1].set_title("Arcs removed in cleaning per trip")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot11_cleaning_impact.png", dpi=150)
    plt.close()
    print("Saved plot11_cleaning_impact.png")
else:
    print("[SKIP] gps_cleaning_summary.csv not found – skipping plot 11")

print(f"\nAll plots saved to: {FIG_DIR.resolve()}")
