"""
network_analysis.py
===================
Builds on exploration.py — single heterogeneous collaboration network.

Nodes: actors, directors, and producers (each tagged with a "role" attribute)
Edges: two people share an edge if they worked on the same movie
Weight: number of movies they collaborated on

This gives a dense, interpretable graph where communities correspond to
real working clusters (e.g. a director + their regular producer + their
repertory actors). Comparing centrality across roles reveals who the
true connectors are — and whether directors, producers, or actors serve
as the structural bridges of Hollywood.

Outputs:
  - Centrality tables (degree, betweenness, eigenvector, closeness)
  - Power-law CCDF log-log plot
  - GEXF file for Gephi (with centrality + role as node attributes)
  - Community detection handled in Gephi (Louvain modularity)

Requirements (beyond exploration.py's deps):
    pip install networkx powerlaw matplotlib
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import powerlaw
except ImportError:
    sys.exit("pip install powerlaw")

from exploration import (
    load_data,
    filter_cast_by_importance,
    analyze_action_actor_genres,
    add_role_flags,
    USE_IMPORTANCE_FILTER,
    BILLING_CUTOFF,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_EDGE_WEIGHT = 2      # prune edges with fewer shared movies than this
OUTPUT_DIR      = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD HETEROGENEOUS COLLABORATION NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def build_collaboration_network(movies, cast, credit, actors, crew,
                                 action_actor_ids=None,
                                 min_edge_weight=MIN_EDGE_WEIGHT):
    """
    Single graph: actors + directors + producers as nodes.
    Edge (A, B, weight=w) means A and B worked on w movies together.

    Edge types that emerge naturally:
      actor  <-> director   (casting relationship)
      actor  <-> producer   (employment relationship)
      actor  <-> actor      (co-starring relationship)
      director <-> producer (production partnership)
      director <-> director (rare — co-directing)
      producer <-> producer (co-production)
    """
    credit_flagged = add_role_flags(credit)

    # ── Gather per-movie participant sets ─────────────────────────────────
    # For each movie, collect all people who worked on it and their roles

    # Actors from cast table
    cast_edges = cast[["Source", "Target"]].copy()
    cast_edges.columns = ["MovieId", "PersonId"]
    if action_actor_ids is not None:
        cast_edges = cast_edges[cast_edges["PersonId"].isin(action_actor_ids)]

    # Directors from credit table
    dir_edges = credit_flagged[credit_flagged["IsDirector"]][["Source", "Target"]].copy()
    dir_edges.columns = ["MovieId", "PersonId"]

    # Producers from credit table
    prod_edges = credit_flagged[credit_flagged["IsProducer"]][["Source", "Target"]].copy()
    prod_edges.columns = ["MovieId", "PersonId"]

    # Restrict to action movies
    action_movie_ids = set(
        movies[movies["Genre"].astype(str).str.contains("Action", case=False, na=False)]["Id"]
    )
    cast_edges = cast_edges[cast_edges["MovieId"].isin(action_movie_ids)]
    dir_edges  = dir_edges[dir_edges["MovieId"].isin(action_movie_ids)]
    prod_edges = prod_edges[prod_edges["MovieId"].isin(action_movie_ids)]

    # Build role lookup: PersonId -> primary role
    # (if someone is both director and producer, they get "director" priority)
    role_map = {}
    for pid in cast_edges["PersonId"].unique():
        role_map[pid] = "actor"
    for pid in prod_edges["PersonId"].unique():
        role_map[pid] = "producer"       # overwrites actor if dual-role
    for pid in dir_edges["PersonId"].unique():
        role_map[pid] = "director"       # director takes priority

    # Build name lookup
    actor_labels = actors.set_index("Id")["Label"].to_dict()
    crew_labels  = crew.set_index("Id")["Label"].to_dict()
    name_map = {**actor_labels, **crew_labels}

    # ── Per-movie participant lists (vectorized) ────────────────────────
    # Stack all (MovieId, PersonId) edges from cast + directors + producers
    # Source = MovieId, Target = PersonId in all three tables
    all_edges = pd.concat([cast_edges, dir_edges, prod_edges], ignore_index=True)
    movie_people = all_edges.groupby("MovieId")["PersonId"].apply(set).to_dict()

    # ── Count pairwise collaborations ────────────────────────────────────
    pair_weights = defaultdict(int)
    for mid, people in movie_people.items():
        people_list = sorted(people)
        for i in range(len(people_list)):
            for j in range(i + 1, len(people_list)):
                pair_weights[(people_list[i], people_list[j])] += 1

    # ── Build graph ──────────────────────────────────────────────────────
    G = nx.Graph()

    for (p1, p2), w in pair_weights.items():
        if w >= min_edge_weight:
            G.add_edge(p1, p2, weight=w)

    # Attach node attributes
    for n in G.nodes():
        G.nodes[n]["label"] = name_map.get(n, str(n))
        G.nodes[n]["role"]  = role_map.get(n, "unknown")

    # Remove isolates (shouldn't exist after edge filter, but just in case)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    # ── Summary ──────────────────────────────────────────────────────────
    role_counts = defaultdict(int)
    for n in G.nodes():
        role_counts[G.nodes[n]["role"]] += 1

    print(f"\nCollaboration Network:")
    print(f"  Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
    for role in ["actor", "director", "producer"]:
        print(f"    {role}s: {role_counts[role]}")

    return G


# ══════════════════════════════════════════════════════════════════════════════
# 2. CENTRALITY
# ══════════════════════════════════════════════════════════════════════════════

def compute_centralities(G, name="Network"):
    print(f"\n[{name}] Computing centralities on {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges ...")

    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clo = nx.closeness_centrality(G)

    if nx.is_connected(G):
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    else:
        gcc_nodes = max(nx.connected_components(G), key=len)
        gcc = G.subgraph(gcc_nodes).copy()
        eig_gcc = nx.eigenvector_centrality_numpy(gcc, weight="weight")
        eig = {n: eig_gcc.get(n, 0.0) for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]["degree_centrality"]      = deg[n]
        G.nodes[n]["betweenness_centrality"]  = bet[n]
        G.nodes[n]["closeness_centrality"]    = clo[n]
        G.nodes[n]["eigenvector_centrality"]  = eig.get(n, 0.0)
        G.nodes[n]["weighted_degree"]         = G.degree(n, weight="weight")

    rows = []
    for n in G.nodes():
        rows.append({
            "Id": n,
            "Label": G.nodes[n].get("label", ""),
            "Role": G.nodes[n].get("role", ""),
            "Degree": G.degree(n),
            "WeightedDeg": G.nodes[n]["weighted_degree"],
            "DegreeCent": round(deg[n], 6),
            "BetweenCent": round(bet[n], 6),
            "ClosenessCent": round(clo[n], 6),
            "EigenvecCent": round(eig.get(n, 0.0), 6),
        })
    df = pd.DataFrame(rows).sort_values("EigenvecCent", ascending=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. POWER-LAW ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_power_law(G, name="Network"):
    degrees = np.array([d for _, d in G.degree()])
    degrees = degrees[degrees > 0]

    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    alpha = fit.power_law.alpha
    xmin  = fit.power_law.xmin
    print(f"\n[{name}] Power-law fit: alpha = {alpha:.3f}, x_min = {xmin}")

    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
    R_ex, p_ex = fit.distribution_compare("power_law", "exponential")
    print(f"  vs lognormal:   R = {R_ln:.3f}, p = {p_ln:.4f}")
    print(f"  vs exponential: R = {R_ex:.3f}, p = {p_ex:.4f}")

    # CCDF log-log plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sorted_deg = np.sort(degrees)[::-1]
    ccdf = np.arange(1, len(sorted_deg) + 1) / len(sorted_deg)
    ax.scatter(sorted_deg, ccdf, s=12, alpha=0.5, label="Empirical CCDF", zorder=3)
    fit.power_law.plot_ccdf(ax=ax, color="r", linestyle="--",
                            label=f"Power-law fit (α={alpha:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("P(K ≥ k)")
    ax.set_title(f"{name} — Degree CCDF (log-log)")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)

    path = os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_ccdf.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return {"alpha": alpha, "xmin": xmin,
            "R_lognormal": R_ln, "p_lognormal": p_ln,
            "R_exponential": R_ex, "p_exponential": p_ex}


# ══════════════════════════════════════════════════════════════════════════════
# 5. STRUCTURAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_structural_summary(G, name="Network"):
    print(f"\n{'=' * 60}")
    print(f"STRUCTURAL SUMMARY: {name}")
    print(f"{'=' * 60}")

    n, m = G.number_of_nodes(), G.number_of_edges()
    density = nx.density(G)
    components = nx.number_connected_components(G)
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc_size = len(gcc_nodes)
    gcc = G.subgraph(gcc_nodes).copy()
    avg_clustering = nx.average_clustering(G, weight="weight")

    try:
        avg_path = nx.average_shortest_path_length(gcc)
    except Exception:
        avg_path = float("nan")

    print(f"  Nodes: {n}  |  Edges: {m}  |  Density: {density:.5f}")
    print(f"  Components: {components}  |  GCC: {gcc_size} ({gcc_size/n*100:.1f}%)")
    print(f"  Avg clustering coeff: {avg_clustering:.4f}")
    print(f"  Avg shortest path (GCC): {avg_path:.2f}")

    # Role breakdown
    role_counts = defaultdict(int)
    for nd in G.nodes():
        role_counts[G.nodes[nd]["role"]] += 1
    for role in ["actor", "director", "producer"]:
        print(f"    {role}s: {role_counts[role]}")

    # Degree stats
    degrees = [d for _, d in G.degree()]
    print(f"  Degree: mean={np.mean(degrees):.1f}, median={np.median(degrees):.0f}, "
          f"max={max(degrees)}")

    # Assortativity by role
    try:
        assort = nx.attribute_assortativity_coefficient(G, "role")
        print(f"  Role assortativity: {assort:.4f}")
        print(f"    (negative = roles tend to connect across types, i.e. heterophilic)")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_gexf(G, name="network"):
    path = os.path.join(OUTPUT_DIR, f"{name}.gexf")
    nx.write_gexf(G, path)
    print(f"GEXF -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data via exploration.py ...")
    movies, cast, credit, actors, crew = load_data()

    if USE_IMPORTANCE_FILTER:
        cast = filter_cast_by_importance(cast, movies, actors)
        print(f"Billing filter (ordering <= {BILLING_CUTOFF}). Cast rows: {len(cast)}")

    action = analyze_action_actor_genres(movies, cast, actors)
    action_ids = action["action_actor_ids"]
    print(f"Action actors: {len(action_ids)}")

    # ── Build the single heterogeneous network ───────────────────────────
    G = build_collaboration_network(
        movies, cast, credit, actors, crew,
        action_actor_ids=action_ids,
        min_edge_weight=MIN_EDGE_WEIGHT,
    )

    # ── Structural summary ───────────────────────────────────────────────
    print_structural_summary(G, "Action Collaboration")

    # ── Centrality ───────────────────────────────────────────────────────
    cent = compute_centralities(G, "Action Collaboration")

    print("\nTop 20 by Eigenvector Centrality:")
    print(cent[["Label", "Role", "Degree", "WeightedDeg",
                "EigenvecCent", "BetweenCent"]].head(20).to_string(index=False))

    # Top by role
    for role in ["actor", "director", "producer"]:
        role_df = cent[cent["Role"] == role]
        print(f"\nTop 10 {role}s by Eigenvector Centrality:")
        print(role_df[["Label", "Degree", "WeightedDeg",
                        "EigenvecCent", "BetweenCent"]].head(10).to_string(index=False))

    cent.to_csv(os.path.join(OUTPUT_DIR, "centrality_all.csv"), index=False)

    # ── Power law ────────────────────────────────────────────────────────
    pl = analyze_power_law(G, "Action Collaboration")

    # ── Export ────────────────────────────────────────────────────────────
    export_gexf(G, "action_collaboration")

    print(f"\nDone. Files in ./{OUTPUT_DIR}/")
    print("Gephi workflow:")
    print("  1. Layout: ForceAtlas 2 (scaling ~2.0, gravity ~1.0, LinLog)")
    print("  2. Statistics -> Modularity (run Louvain for community detection)")
    print("  3. Node size  -> eigenvector_centrality or weighted_degree")
    print("  4. Node color -> Modularity Class (community) OR role")
    print("  5. Labels: show only top ~30 nodes by eigenvector centrality")
    print("  6. Edge thickness -> weight")


if __name__ == "__main__":
    main()