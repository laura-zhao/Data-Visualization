"""
Cell lineage graph with optimal transport (Waddington-OT)

Reproduces the "layered digraph" style lineage plots from the Waddington-OT
papers (e.g. the lung KP tumor evolution figure), using only the transport
maps already saved to disk by wot.OTModel.compute_all_transport_maps() -
no OT computation happens in this script.

The MATLAB original builds a digraph of cluster -> cluster edges and calls
plot(G, 'layout', 'layered'), which is a topological (graphviz "dot" style)
layout that minimizes edge crossings. This script reproduces the same idea
with networkx:
  - nodes are laid out in horizontal layers, one per time point
  - edge width is scaled by the rank (percentile) of its weight, not the
    raw weight, so a few huge transitions don't drown out everything else
  - edge color is inherited from the earliest (Ta) ancestor, so a lineage
    keeps one color as it fans out over time
  - node size is proportional to PageRank centrality on the weighted graph,
    matching the MATLAB centrality(G, 'pagerank') sizing
"""

import os
import wot
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Resource paths
H5AD_PATH = '/home/ZKX/cdh6/hjx.h5ad'
TMAP_DIR = 'tmaps/bladder'

TIME_MAP = {'Ta': 0.0, 'T1': 1.0, '>T2': 2.0}
STAGE_ORDER = ['Ta', 'T1', '>T2']
CLUSTER_KEY = 'seurat_clusters'
MIN_EDGE_WEIGHT = 0.01  # drop transitions below this row-normalized probability


def load_expression_data():
    adata = wot.io.read_dataset(H5AD_PATH)
    adata.obs['day'] = adata.obs['orig.ident'].map(TIME_MAP).astype(float)
    return adata


def cluster_transition_matrix(tmap_model, adata, day0, day1, cluster_key=CLUSTER_KEY):
    """Aggregate the cell-level coupling between day0 and day1 into a
    cluster x cluster matrix, row-normalized so each source cluster's
    outgoing mass sums to 1 (i.e. P(target cluster | source cluster))."""
    coupling = tmap_model.get_coupling(day0, day1)

    src_clusters = adata.obs.loc[coupling.obs.index, cluster_key]
    tgt_clusters = adata.obs.loc[coupling.var.index, cluster_key]

    df = pd.DataFrame(coupling.X, index=src_clusters.values, columns=tgt_clusters.values)
    agg = df.groupby(level=0).sum().T.groupby(level=0).sum().T
    agg = agg.div(agg.sum(axis=1), axis=0).fillna(0)
    return agg


def build_lineage_graph(matrices, min_weight=MIN_EDGE_WEIGHT):
    """matrices: list of (stage_from, stage_to, matrix) in chronological
    order. Returns a networkx.DiGraph with node attr 'stage' and edge
    attr 'weight'."""
    G = nx.DiGraph()

    for stage, _, matrix in [(m[0], m[1], m[2]) for m in matrices]:
        for cluster in matrix.index:
            G.add_node(f'{stage}_{cluster}', stage=stage, cluster=str(cluster))
    last_stage, last_matrix = matrices[-1][1], matrices[-1][2]
    for cluster in last_matrix.columns:
        G.add_node(f'{last_stage}_{cluster}', stage=last_stage, cluster=str(cluster))

    for stage_from, stage_to, matrix in matrices:
        for src in matrix.index:
            for tgt in matrix.columns:
                w = matrix.loc[src, tgt]
                if w > min_weight:
                    G.add_edge(f'{stage_from}_{src}', f'{stage_to}_{tgt}', weight=w)

    return G


def lineage_colors(G, root_stage):
    """Color every node/edge by its earliest (root_stage) ancestor, so a
    lineage keeps one color as it fans out across later time points."""
    roots = [n for n, d in G.nodes(data=True) if d['stage'] == root_stage]
    cmap = cm.get_cmap('tab10', max(len(roots), 1))
    root_color = {r: cmap(i) for i, r in enumerate(roots)}

    color = dict(root_color)
    for n in nx.topological_sort(G):
        if n in color:
            continue
        preds = list(G.predecessors(n))
        if not preds:
            color[n] = (0.6, 0.6, 0.6, 1.0)
            continue
        # inherit the color of the strongest incoming edge's source
        best_pred = max(preds, key=lambda p: G[p][n]['weight'])
        color[n] = color.get(best_pred, (0.6, 0.6, 0.6, 1.0))
    return color


def layered_layout(G, stage_order):
    """Horizontal-layer layout: y = stage index (Ta on top), x = position
    within the layer chosen with a barycenter heuristic (average x of
    neighbors in the adjacent, already-placed layer) to reduce edge
    crossings - a lightweight stand-in for graphviz's 'dot' layout."""
    layers = {s: [n for n, d in G.nodes(data=True) if d['stage'] == s] for s in stage_order}

    pos = {}
    for i, node in enumerate(sorted(layers[stage_order[0]])):
        pos[node] = (float(i), 0.0)

    for depth in range(1, len(stage_order)):
        stage = stage_order[depth]
        nodes = layers[stage]

        def barycenter(n):
            preds = list(G.predecessors(n))
            xs = [pos[p][0] for p in preds if p in pos]
            return np.mean(xs) if xs else 0.0

        ordered = sorted(nodes, key=barycenter)
        for i, node in enumerate(ordered):
            pos[node] = (float(i), -float(depth))

    return pos


def plot_lineage_graph(G, pos, node_color, out_path='pictures/wot_lineage_graph.png'):
    pagerank = nx.pagerank(G, weight='weight')
    weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    ranks = pd.Series(weights).rank(pct=True).values  # ecdfQuantile-style scaling

    fig, ax = plt.subplots(figsize=(12, 12))

    for (u, v, d), rank in zip(G.edges(data=True), ranks):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle='arc3,rad=0.15',
            arrowstyle='-|>', mutation_scale=12,
            linewidth=rank * 8 + 0.3,
            color=node_color.get(u, (0.6, 0.6, 0.6, 1.0)),
            alpha=0.6, zorder=1,
        )
        ax.add_patch(arrow)

    for node, (x, y) in pos.items():
        size = pagerank[node] * 4000 + 60
        ax.scatter(x, y, s=size, color='black', edgecolors='white',
                   linewidth=1.2, zorder=3)
        ax.text(x, y + 0.12, G.nodes[node]['cluster'], fontsize=13,
                fontweight='bold', ha='center', zorder=4)

    for stage in set(nx.get_node_attributes(G, 'stage').values()):
        ys = [pos[n][1] for n, d in G.nodes(data=True) if d['stage'] == stage]
        if ys:
            ax.text(min(x for x, _ in pos.values()) - 1.5, ys[0], stage,
                    fontsize=14, fontweight='bold', ha='right', va='center')

    ax.set_title("Waddington-OT: cell lineage graph", fontsize=18)
    ax.set_axis_off()
    ax.margins(0.15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return fig


if __name__ == '__main__':
    adata = load_expression_data()
    tmap_model = wot.tmap.TransportMapModel.from_directory(TMAP_DIR)

    matrix_0_1 = cluster_transition_matrix(tmap_model, adata, 0.0, 1.0)
    matrix_1_2 = cluster_transition_matrix(tmap_model, adata, 1.0, 2.0)

    G = build_lineage_graph([
        ('Ta', 'T1', matrix_0_1),
        ('T1', '>T2', matrix_1_2),
    ])

    pos = layered_layout(G, STAGE_ORDER)
    node_color = lineage_colors(G, root_stage='Ta')

    plot_lineage_graph(G, pos, node_color)
    plt.show()
