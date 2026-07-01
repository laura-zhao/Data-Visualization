"""
Cell lineage graph with optimal transport (Waddington-OT)

Reproduces the "layered digraph" style lineage plots from the Waddington-OT
papers (e.g. the lung KP tumor evolution figure), using only the transport
maps already saved to disk by wot.OTModel.compute_all_transport_maps() -
no OT computation happens in this script.

The MATLAB original builds a digraph of cluster -> cluster edges and calls
plot(G, 'layout', 'layered'), which is a topological (graphviz "dot" style)
layout that minimizes edge crossings, then draws each edge as a flowing
ribbon rather than a plain line. This script reproduces both pieces:
  - nodes are laid out in horizontal layers, one per time point, with
    iterative barycenter sweeps (forward + backward, Sugiyama-style) to
    reduce edge crossings
  - each edge is drawn as a filled 3D Sankey-style ribbon: a smooth S-curve
    that tapers to a point at both nodes and bulges to its full width in
    the middle, while also swaying out into the depth axis and back - that
    out-and-back sway (combined with a tilted camera) is what produces the
    woven, cage-like look instead of a flat diagram
  - edge color is inherited from the earliest (Ta) ancestor, so a lineage
    keeps one color as it fans out over time
  - node size is proportional to PageRank centrality on the weighted graph,
    log-scaled so a few dominant clusters don't dwarf everything else
"""

import wot
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

    for stage, _, matrix in matrices:
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


def layered_layout(G, stage_order, iterations=4):
    """Sugiyama-style layered layout: y = stage index (Ta on top), x =
    position within the layer. Node order within each layer is refined
    over several forward (top-down) and backward (bottom-up) barycenter
    sweeps to reduce edge crossings - a lightweight stand-in for
    graphviz's 'dot' layout."""
    layers = {s: [n for n, d in G.nodes(data=True) if d['stage'] == s] for s in stage_order}
    order = {s: sorted(layers[s], key=str) for s in stage_order}

    def positions_from_order():
        pos = {}
        for depth, stage in enumerate(stage_order):
            for i, node in enumerate(order[stage]):
                pos[node] = (float(i), -float(depth))
        return pos

    pos = positions_from_order()

    for _ in range(iterations):
        for depth in range(1, len(stage_order)):
            stage = stage_order[depth]

            def bary_fwd(n):
                xs = [pos[p][0] for p in G.predecessors(n) if p in pos]
                return np.mean(xs) if xs else pos[n][0]

            order[stage] = sorted(order[stage], key=bary_fwd)
            pos = positions_from_order()

        for depth in range(len(stage_order) - 2, -1, -1):
            stage = stage_order[depth]

            def bary_bwd(n):
                xs = [pos[s][0] for s in G.successors(n) if s in pos]
                return np.mean(xs) if xs else pos[n][0]

            order[stage] = sorted(order[stage], key=bary_bwd)
            pos = positions_from_order()

    return pos


def sankey_ribbon_3d(x1, z1, x2, z2, width, bow, n=40):
    """Vertex strip for one edge in 3D: x eases from x1 to x2 (smoothstep),
    z (time) descends linearly, and y (depth) bows out to `bow` at the
    midpoint and back to 0 at both nodes - this out-and-back sway in the
    depth axis, combined with a tilted camera, is what gives the plot its
    woven, cage-like look instead of a flat diagram. The ribbon's in-plane
    width also bulges to `width` at the midpoint and tapers to a point at
    both endpoints, same as a Sankey flow."""
    t = np.linspace(0, 1, n)
    ease = 3 * t ** 2 - 2 * t ** 3
    x = x1 + (x2 - x1) * ease
    z = z1 + (z2 - z1) * t
    y = bow * np.sin(np.pi * t)
    w = width * np.sin(np.pi * t)

    upper = np.stack([x + w / 2, y, z], axis=1)
    lower = np.stack([x - w / 2, y, z], axis=1)
    return upper, lower


def plot_lineage_graph_3d(G, pos, node_color, max_ribbon_width=0.7,
                           elev=18, azim=-70):
    pagerank = nx.pagerank(G, weight='weight')
    weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    ranks = pd.Series(weights).rank(pct=True).values

    xs_all = [p[0] for p in pos.values()]
    x_span = max(xs_all) - min(xs_all) or 1.0

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111, projection='3d')

    for (u, v, d), rank in zip(G.edges(data=True), ranks):
        x1, z1 = pos[u]
        x2, z2 = pos[v]
        # longer transitions sway further into the depth axis; the sign
        # alternates by source position so lineages weave in front of and
        # behind one another instead of all bowing the same way
        side = 1.0 if (hash(u) % 2 == 0) else -1.0
        bow = side * (0.35 + 0.9 * abs(x2 - x1) / x_span)
        width = rank * max_ribbon_width + 0.02

        upper, lower = sankey_ribbon_3d(x1, z1, x2, z2, width, bow)
        color = node_color.get(u, (0.6, 0.6, 0.6, 1.0))
        quads = [[upper[i], upper[i + 1], lower[i + 1], lower[i]] for i in range(len(upper) - 1)]
        ax.add_collection3d(Poly3DCollection(quads, facecolor=color, edgecolor='none', alpha=0.55))

    for node, (x, z) in pos.items():
        size = np.log1p(pagerank[node] * 1000) * 300 + 60
        ax.scatter(x, 0, z, s=size, color='black', edgecolors='white',
                   linewidth=1.2, depthshade=False, zorder=5)
        ax.text(x, 0, z + 0.15, G.nodes[node]['cluster'], fontsize=12,
                fontweight='bold', ha='center', zorder=6)

    for stage in set(nx.get_node_attributes(G, 'stage').values()):
        zs = [pos[n][1] for n, d in G.nodes(data=True) if d['stage'] == stage]
        if zs:
            ax.text(min(xs_all) - 1.5, 0, zs[0], stage, fontsize=14,
                    fontweight='bold', ha='right', va='center')

    ax.set_title("Waddington-OT: cell lineage graph", fontsize=18)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((x_span, x_span * 0.6, x_span * 0.7))
    fig.tight_layout()
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

    plot_lineage_graph_3d(G, pos, node_color)
    plt.show()
