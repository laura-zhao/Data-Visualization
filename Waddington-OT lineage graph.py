"""
Cell ancestor graph with optimal transport (Waddington-OT)

Reproduces the MATLAB `plot(digraph(...), 'layout', 'layered')` lineage
figure from the Waddington-OT papers, using only the transport maps
already saved to disk by wot.OTModel.compute_all_transport_maps() - no OT
computation happens in this script.

Two things the earlier version of this script got wrong, corrected here:

  1. The MATLAB graph is NOT built with one node per (stage, cluster) pair.
     It strips the stage prefix off each cluster label before building the
     digraph, so a cluster observed at several time points collapses onto
     a single shared node, and edges from every consecutive-day transition
     all land on that same small set of nodes. That's what produces the
     wide, arcing "cage" look: nodes have many in/out edges from different
     time steps, and MATLAB's layered layout assigns each node's rank by
     longest path through the whole graph (not a fixed 3-row grid), so
     edges that skip several ranks get routed as long curved arcs.
  2. It is a plain 2D plot with constant-width edges (`p.LineWidth =
     ecdfQuantile(weight)*10`), not a 3D scene and not tapered Sankey
     ribbons - both of those were wrong guesses in an earlier revision.

This script matches that: nodes are raw cluster labels shared across time
points, layer/rank is computed by longest path from source nodes (a
"dot"-style topological layering) refined with barycenter sweeps, edges
are constant-width curved arcs, edge color encodes which consecutive-day
transition the edge came from (matching MATLAB's `zTT` coloring), edge
width is the percentile rank of the transition weight, and node size is
PageRank centrality - all mirroring the MATLAB formulas directly.
"""

import wot
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Resource paths
H5AD_PATH = '/home/ZKX/cdh6/hjx.h5ad'
TMAP_DIR = '/home/ZKX/tmaps/bladder'

TIME_MAP = {'Ta': 0.0, 'T1': 1.0, '>T2': 2.0}
STEPS = [('Ta', 'T1', 0.0, 1.0), ('T1', '>T2', 1.0, 2.0)]
CLUSTER_KEY = 'seurat_clusters'
MIN_EDGE_WEIGHT = 0.01


def load_expression_data():
    adata = wot.io.read_dataset(H5AD_PATH)
    adata.obs['day'] = adata.obs['orig.ident'].map(TIME_MAP).astype(float)
    return adata


def cluster_transition_matrix(tmap_model, adata, day0, day1, cluster_key=CLUSTER_KEY):
    """Aggregate the cell-level coupling between day0 and day1 into a
    cluster x cluster matrix, row-normalized so each source cluster's
    outgoing mass sums to 1."""
    coupling = tmap_model.get_coupling(day0, day1)

    src_clusters = adata.obs.loc[coupling.obs.index, cluster_key]
    tgt_clusters = adata.obs.loc[coupling.var.index, cluster_key]

    df = pd.DataFrame(coupling.X, index=src_clusters.values, columns=tgt_clusters.values)
    agg = df.groupby(level=0).sum().T.groupby(level=0).sum().T
    agg = agg.div(agg.sum(axis=1), axis=0).fillna(0)
    return agg


def build_ancestor_graph(steps, adata, tmap_model, min_weight=MIN_EDGE_WEIGHT):
    """Build MultiDiGraph over raw cluster labels (shared across time points)
    by stacking every consecutive-day transition matrix."""
    G = nx.MultiDiGraph()

    for step_idx, (_, _, day0, day1) in enumerate(steps):
        matrix = cluster_transition_matrix(tmap_model, adata, day0, day1)
        for src in matrix.index:
            for tgt in matrix.columns:
                w = matrix.loc[src, tgt]
                if w > min_weight:
                    G.add_edge(str(src), str(tgt), weight=w, step=step_idx)

    return G


def acyclic_skeleton(G):
    """Detect and remove back edges to break cycles for ranking.
    Because nodes are shared cluster labels, the merged graph can contain
    cycles (e.g. cluster A -> B in one step and B -> A in the next).
    The returned DAG is only used for ranking; original G is still drawn."""
    simple = nx.DiGraph()
    simple.add_nodes_from(G.nodes)
    simple.add_edges_from({(u, v) for u, v in G.edges()})

    acyclic = nx.DiGraph()
    acyclic.add_nodes_from(simple.nodes)
    visited, in_stack = set(), set()

    def dfs(n):
        visited.add(n)
        in_stack.add(n)
        for succ in simple.successors(n):
            if succ not in visited:
                acyclic.add_edge(n, succ)
                dfs(succ)
            elif succ not in in_stack:
                acyclic.add_edge(n, succ)
        in_stack.discard(n)

    for n in simple.nodes:
        if n not in visited:
            dfs(n)

    return acyclic


def longest_path_layers(G):
    """Topological rank by longest path - dot-style layer assignment."""
    acyclic = acyclic_skeleton(G)
    layer = {}
    for n in nx.topological_sort(acyclic):
        preds = list(acyclic.predecessors(n))
        layer[n] = max((layer[p] for p in preds), default=-1) + 1
    return layer


def layered_layout(G, iterations=6):
    """Sugiyama-style layout with barycenter sweeps to minimize crossings."""
    layer = longest_path_layers(G)
    max_layer = max(layer.values())
    order = {d: sorted([n for n in G.nodes if layer[n] == d], key=str)
             for d in range(max_layer + 1)}

    def positions():
        pos = {}
        for d, nodes in order.items():
            for i, n in enumerate(nodes):
                pos[n] = (float(i) - (len(nodes) - 1) / 2.0, -float(d))
        return pos

    pos = positions()
    for _ in range(iterations):
        for d in range(1, max_layer + 1):
            def bary_fwd(n):
                xs = [pos[p][0] for p in G.predecessors(n) if p in pos]
                return np.mean(xs) if xs else pos[n][0]
            order[d] = sorted(order[d], key=bary_fwd)
            pos = positions()

        for d in range(max_layer - 1, -1, -1):
            def bary_bwd(n):
                xs = [pos[s][0] for s in G.successors(n) if s in pos]
                return np.mean(xs) if xs else pos[n][0]
            order[d] = sorted(order[d], key=bary_bwd)
            pos = positions()

    return pos


def plot_ancestor_graph(G, pos, step_labels, max_linewidth=10):
    pagerank = nx.pagerank(G, weight='weight')
    weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    ranks = pd.Series(weights).rank(pct=True).values

    step_cmap = cm.get_cmap('Set1', len(step_labels))

    fig, ax = plt.subplots(figsize=(16, 16), dpi=100)

    for (u, v, d), rank in zip(G.edges(data=True), ranks):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        layer_gap = abs(y2 - y1)
        rad = 0.15 + 0.06 * layer_gap
        if hash((u, v, d['step'])) % 2:
            rad = -rad

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>', mutation_scale=14,
            linewidth=rank * max_linewidth + 0.5,
            color=step_cmap(d['step']),
            alpha=0.7, zorder=1,
        )
        ax.add_patch(arrow)

    for node, (x, y) in pos.items():
        size = max(20, round(pagerank[node] * 150) + 4)
        ax.scatter(x, y, s=size, color='black', edgecolors='white',
                   linewidth=2, zorder=3)
        ax.text(x, y + 0.2, node, fontsize=18, fontweight='bold',
                ha='center', va='bottom', zorder=4)

    handles = [plt.Line2D([0], [0], color=step_cmap(i), lw=4, label=label)
               for i, label in enumerate(step_labels)]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=14)

    ax.set_title("Waddington-OT: cell ancestor graph", fontsize=20, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.margins(0.15)
    fig.tight_layout()
    return fig


# Main execution
adata = load_expression_data()
tmap_model = wot.tmap.TransportMapModel.from_directory(TMAP_DIR)

G = build_ancestor_graph(STEPS, adata, tmap_model)
pos = layered_layout(G)
step_labels = [f'{a} -> {b}' for a, b, _, _ in STEPS]

plot_ancestor_graph(G, pos, step_labels)
plt.show()
