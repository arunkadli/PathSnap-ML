"""
feature_engineering.py
=======================
Amrithaa – Network feature engineering for map-matching ML models.

Produces two kinds of features:

A. Network features  (per arc, computed once from cleaned nodes+arcs)
   – arc length in metres
   – source/target node in-degree, out-degree, total degree
   – is_bidirectional flag
   – turn angle at source / target (mean turn cost)
   – betweenness-based connectivity proxy (degree ratio)

B. Candidate-match features  (per GPS-obs × candidate-arc row)
   – perpendicular distance from GPS point to arc (metres)
   – along-track distance from GPS point to nearest arc endpoint
   – heading difference (degrees)
   – speed × arc-length compatibility score
   – time-gap flag
   – previous-arc transition indicator (arc adjacency)
   – normalised candidate rank

Both feature sets are merged into the final training/prediction table.

Usage
-----
    from src.feature_engineering import build_network_features, build_candidate_features

    # Once per trip
    net_feats = build_network_features(nodes_df, arcs_df, crs)

    # Per GPS-track
    candidate_feats = build_candidate_features(candidates_df, net_feats)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    print("[WARN] networkx not found – graph-based features will be skipped.")

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

try:
    from src.road_network_preprocessing import get_crs_for_trip
    from src.candidate_generation import _project_coords, _bearing_deg, _heading_diff
except ImportError:
    from road_network_preprocessing import get_crs_for_trip
    from candidate_generation import _project_coords, _bearing_deg, _heading_diff


# ─────────────────────────────────────────────────────────────────────────────
# A. Network features
# ─────────────────────────────────────────────────────────────────────────────

def build_network_features(
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    crs: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute arc-level network features.

    Parameters
    ----------
    nodes : DataFrame[node_id, lon, lat]
    arcs  : DataFrame[arc_id, orig_arc_id, source, target, length_m]
    crs   : EPSG string; derived from nodes if not provided

    Returns
    -------
    DataFrame indexed by arc_id with network feature columns:
        orig_arc_id, arc_id, source, target,
        length_m,
        in_degree_src, out_degree_src, total_degree_src,
        in_degree_tgt, out_degree_tgt, total_degree_tgt,
        is_bidirectional,
        turn_angle_src_deg, turn_angle_tgt_deg,
        connectivity_ratio_src, connectivity_ratio_tgt,
        arc_bearing_deg
    """
    if crs is None:
        crs = get_crs_for_trip(nodes["lon"].values, nodes["lat"].values)

    # ── Project ──────────────────────────────────────────────────────────────
    nidx = nodes.set_index("node_id")

    src_lon = nidx.loc[arcs["source"].values, "lon"].values
    src_lat = nidx.loc[arcs["source"].values, "lat"].values
    tgt_lon = nidx.loc[arcs["target"].values, "lon"].values
    tgt_lat = nidx.loc[arcs["target"].values, "lat"].values

    sx, sy = _project_coords(src_lon, src_lat, crs)
    tx, ty = _project_coords(tgt_lon, tgt_lat, crs)

    arc_bearing = _bearing_deg(sx, sy, tx, ty)

    # ── Directed graph ────────────────────────────────────────────────────────
    if _HAS_NX:
        DG = nx.DiGraph()
        DG.add_nodes_from(nodes["node_id"].values)
        DG.add_edges_from(arcs[["source", "target"]].values)

        in_deg  = dict(DG.in_degree())
        out_deg = dict(DG.out_degree())
    else:
        # Manual degree computation without networkx
        in_deg  = arcs.groupby("target")["arc_id"].count().to_dict()
        out_deg = arcs.groupby("source")["arc_id"].count().to_dict()
        for nid in nodes["node_id"]:
            in_deg.setdefault(nid, 0)
            out_deg.setdefault(nid, 0)

    src_in  = np.array([in_deg.get(s, 0)  for s in arcs["source"]])
    src_out = np.array([out_deg.get(s, 0) for s in arcs["source"]])
    tgt_in  = np.array([in_deg.get(t, 0)  for t in arcs["target"]])
    tgt_out = np.array([out_deg.get(t, 0) for t in arcs["target"]])

    # ── Bidirectionality ─────────────────────────────────────────────────────
    arc_set = set(zip(arcs["source"], arcs["target"]))
    is_bidir = np.array([
        1 if (int(t), int(s)) in arc_set else 0
        for s, t in zip(arcs["source"], arcs["target"])
    ])

    # ── Turn angle at endpoints ───────────────────────────────────────────────
    # For each arc's source node: mean bearing difference with incoming arcs.
    # For each arc's target node: mean bearing difference with outgoing arcs.
    # This is a proxy for how "straight" the road is at that node.
    src_turn, tgt_turn = _compute_turn_angles(arcs, arc_bearing)

    # ── Connectivity ratio (out / (in+out) at source; in / (in+out) at target)
    conn_src = np.where(src_in + src_out > 0,
                        src_out / (src_in + src_out), 0.5)
    conn_tgt = np.where(tgt_in + tgt_out > 0,
                        tgt_in  / (tgt_in + tgt_out), 0.5)

    # ── Assemble ─────────────────────────────────────────────────────────────
    feat = pd.DataFrame({
        "arc_id":                arcs["arc_id"].values,
        "orig_arc_id":           arcs["orig_arc_id"].values,
        "source":                arcs["source"].values,
        "target":                arcs["target"].values,
        "length_m":              arcs["length_m"].values,
        "in_degree_src":         src_in,
        "out_degree_src":        src_out,
        "total_degree_src":      src_in + src_out,
        "in_degree_tgt":         tgt_in,
        "out_degree_tgt":        tgt_out,
        "total_degree_tgt":      tgt_in + tgt_out,
        "is_bidirectional":      is_bidir,
        "turn_angle_src_deg":    src_turn,
        "turn_angle_tgt_deg":    tgt_turn,
        "connectivity_ratio_src": conn_src,
        "connectivity_ratio_tgt": conn_tgt,
        "arc_bearing_deg":        arc_bearing,
    })

    return feat


def _compute_turn_angles(
    arcs: pd.DataFrame,
    arc_bearing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each arc compute the mean heading difference between this arc and all
    other arcs at its source node (incoming arcs) and target node (outgoing).

    Returns (turn_at_source, turn_at_target) arrays aligned with arcs.
    """
    bearing_by_arc = dict(zip(arcs["arc_id"].values, arc_bearing))

    # outgoing arcs grouped by source node
    src_groups: dict[int, list[float]] = {}
    tgt_groups: dict[int, list[float]] = {}
    for row, b in zip(arcs.itertuples(index=False), arc_bearing):
        src_groups.setdefault(int(row.source), []).append(b)
        tgt_groups.setdefault(int(row.target), []).append(b)

    src_turn = np.zeros(len(arcs))
    tgt_turn = np.zeros(len(arcs))

    for i, (row, b) in enumerate(zip(arcs.itertuples(index=False), arc_bearing)):
        # turn at source: compare with all incoming arcs (arcs whose target == source)
        inc = tgt_groups.get(int(row.source), [b])
        diffs = [min(abs(b - x) % 360, 360 - abs(b - x) % 360) for x in inc]
        src_turn[i] = float(np.mean(diffs))

        # turn at target: compare with all outgoing arcs (arcs whose source == target)
        out = src_groups.get(int(row.target), [b])
        diffs = [min(abs(b - x) % 360, 360 - abs(b - x) % 360) for x in out]
        tgt_turn[i] = float(np.mean(diffs))

    return src_turn, tgt_turn


# ─────────────────────────────────────────────────────────────────────────────
# B. Candidate-match features
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_features(
    candidates: pd.DataFrame,
    net_feats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge candidate rows with network features and add GPS-match features.

    Parameters
    ----------
    candidates : output of candidate_generation.generate_candidates()
    net_feats  : output of build_network_features()

    Returns
    -------
    Feature DataFrame ready for ML model training/prediction.
    Columns include all network features plus:
        norm_perp_dist      – perp_dist_m normalised to [0, 1] within trip
        norm_heading_diff   – heading_diff_deg / 180
        speed_arc_compat    – gps_speed_mps / (length_m / 10 + 1e-6)  clipped [0,1]
        is_prev_arc_adj     – 1 if this arc is adjacent to previous obs's arc
        norm_rank           – candidate_rank / max_rank_in_obs
        log_length_m        – log1p(length_m)
        log_perp_dist_m     – log1p(perp_dist_m)
    """
    # Drop geometry columns not needed for ML (keep orig_arc_id for label join)
    merge_cols = [
        "arc_id", "orig_arc_id", "length_m",
        "in_degree_src", "out_degree_src", "total_degree_src",
        "in_degree_tgt", "out_degree_tgt", "total_degree_tgt",
        "is_bidirectional",
        "turn_angle_src_deg", "turn_angle_tgt_deg",
        "connectivity_ratio_src", "connectivity_ratio_tgt",
        "arc_bearing_deg",
    ]
    # Only keep columns that exist in net_feats
    merge_cols = [c for c in merge_cols if c in net_feats.columns]

    merged = candidates.merge(
        net_feats[merge_cols],
        on=["arc_id", "orig_arc_id"] if "orig_arc_id" in candidates.columns else ["arc_id"],
        how="left",
        suffixes=("", "_net"),
    )

    # ── Derived features ──────────────────────────────────────────────────────

    # Normalised perpendicular distance within each trip
    trip_max_dist = (
        merged.groupby("trip_id")["perp_dist_m"]
        .transform("max")
        .replace(0, 1.0)
    )
    merged["norm_perp_dist"] = (merged["perp_dist_m"] / trip_max_dist).clip(0, 1)

    # Heading difference normalised to [0, 1]
    merged["norm_heading_diff"] = (merged["heading_diff_deg"] / 180.0).clip(0, 1)

    # Speed–arc compatibility: low if GPS speed is too fast relative to short arcs
    merged["speed_arc_compat"] = (
        merged["gps_speed_mps"] / (merged["length_m"] / 10.0 + 1e-6)
    ).clip(0, 1)

    # Log transforms for skewed features
    merged["log_length_m"]    = np.log1p(merged["length_m"])
    merged["log_perp_dist_m"] = np.log1p(merged["perp_dist_m"])

    # Normalised candidate rank within each GPS observation
    max_rank = merged.groupby(["trip_id", "gps_idx"])["candidate_rank"].transform("max")
    merged["norm_rank"] = merged["candidate_rank"] / max_rank.replace(0, 1)

    # Previous-arc adjacency: 1 if this arc's source == prev obs's chosen arc's target
    # (approximated using same-trip, consecutive gps_idx, rank-1 arc)
    merged["is_prev_arc_adj"] = _flag_prev_arc_adjacency(merged)

    return merged


def _flag_prev_arc_adjacency(df: pd.DataFrame) -> pd.Series:
    """
    For each row, check if the arc's source node equals the target node of the
    best-rank (rank=1) arc at the previous GPS observation in the same trip.

    Returns a binary Series (0/1).
    """
    # Get the best candidate's target node per (trip_id, gps_idx)
    best = (
        df[df["candidate_rank"] == 1]
        .set_index(["trip_id", "gps_idx"])["target"]
        .rename("prev_target")
    )

    result = pd.Series(0, index=df.index)

    for trip_id, grp in df.groupby("trip_id"):
        obs_indices = sorted(grp["gps_idx"].unique())
        for obs_idx in obs_indices:
            if obs_idx == 0:
                continue
            prev_key = (trip_id, obs_idx - 1)
            if prev_key not in best.index:
                continue
            prev_tgt = best.loc[prev_key]
            mask = (grp["gps_idx"] == obs_idx) & (grp["source"] == prev_tgt)
            result.loc[grp[mask].index] = 1

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Node degree export helper (used by candidate_generation degree_csv)
# ─────────────────────────────────────────────────────────────────────────────

def export_node_degrees(
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """
    Compute and save per-node degree table.

    Columns: node_id, orig_node_id, lon, lat, in_degree, out_degree, total_degree
    """
    in_deg  = arcs.groupby("target")["arc_id"].count().rename("in_degree")
    out_deg = arcs.groupby("source")["arc_id"].count().rename("out_degree")

    deg = nodes.set_index("node_id").join(in_deg).join(out_deg)
    deg["in_degree"]    = deg["in_degree"].fillna(0).astype(int)
    deg["out_degree"]   = deg["out_degree"].fillna(0).astype(int)
    deg["total_degree"] = deg["in_degree"] + deg["out_degree"]
    deg = deg.reset_index()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    deg.to_csv(out_path, index=False)
    return deg


# ─────────────────────────────────────────────────────────────────────────────
# Batch feature pipeline (all trips)
# ─────────────────────────────────────────────────────────────────────────────

def build_all_features(
    network_root: Path,
    candidates_root: Path,
    out_dir: Path,
) -> pd.DataFrame:
    """
    For every trip that has network + candidate files, build full feature table.

    Parameters
    ----------
    network_root    : results/network_cleaned/
    candidates_root : results/candidates/
    out_dir         : results/features/

    Returns
    -------
    Combined feature DataFrame for all trips.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []

    trip_ids = sorted(d.name for d in network_root.iterdir() if d.is_dir())

    for tid in trip_ids:
        nodes_csv = network_root / tid / f"{tid}_nodes_clean.csv"
        arcs_csv  = network_root / tid / f"{tid}_arcs_clean.csv"
        cands_csv = candidates_root / f"{tid}_candidates.csv"

        if not (nodes_csv.exists() and arcs_csv.exists() and cands_csv.exists()):
            print(f"[SKIP] {tid}: missing file(s)")
            continue

        try:
            nodes  = pd.read_csv(nodes_csv)
            arcs   = pd.read_csv(arcs_csv)
            cands  = pd.read_csv(cands_csv)
            crs    = get_crs_for_trip(nodes["lon"].values, nodes["lat"].values)

            net_feats  = build_network_features(nodes, arcs, crs)
            feat_table = build_candidate_features(cands, net_feats)

            # Also export node degree file
            export_node_degrees(
                nodes, arcs,
                out_dir / tid / f"{tid}_node_degrees.csv"
            )

        except Exception as exc:
            print(f"[ERROR] {tid}: {exc}")
            continue

        per_trip_path = out_dir / f"{tid}_features.csv"
        feat_table.to_csv(per_trip_path, index=False)
        all_frames.append(feat_table)
        print(f"[OK] {tid}  features: {len(feat_table)} rows × {len(feat_table.columns)} cols")

    if not all_frames:
        print("[WARN] No feature files generated.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(out_dir / "all_features.csv", index=False)
    print(f"\n[DONE] Combined feature table: {len(combined)} rows")
    print(f"  → {out_dir / 'all_features.csv'}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    combined = build_all_features(
        network_root    = Path("results/network_cleaned"),
        candidates_root = Path("results/candidates"),
        out_dir         = Path("results/features"),
    )
    if not combined.empty:
        print("\nFeature columns:", list(combined.columns))
        print(combined.head(5).to_string(index=False))
