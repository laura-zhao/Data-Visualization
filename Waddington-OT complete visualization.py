"""
Waddington-OT: Complete Cell Lineage Visualization

Displays both:
1. 2D layered graph (MATLAB-style) - left panel
2. 3D cage-like trajectory landscape - right panel

Both views use the same underlying transport map data,
just different geometric layouts for complementary insights.
"""

import wot
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# CONFIGURATION
# ==========================================
H5AD_PATH = '/home/ZKX/cdh6/hjx.h5ad'
TMAP_DIR = '/home/ZKX/tmaps/bladder'

TIME_MAP = {'Ta': 0.0, 'T1': 1.0, '>T2': 2.0}
STEPS = [('Ta', 'T1', 0.0, 1.0), ('T1', '>T2', 1.0, 2.0)]
CLUSTER_KEY = 'seurat_clusters'
MIN_EDGE_WEIGHT = 0.01
THRESHOLD_3D = 0.005

# ==========================================
# DATA LOADING
# ==========================================
def load_expression_data():
    """Load H5AD file and add time mapping"""
    adata = wot.io.read_dataset(H5AD_PATH)
    adata.obs['day'] = adata.obs['orig.ident'].map(TIME_MAP).astype(float)
    return adata


def cluster_transition_matrix(tmap_model, adata, day0, day1, cluster_key=CLUSTER_KEY):
    """Aggregate cell-level coupling into cluster x cluster matrix"""
    coupling = tmap_model.get_coupling(day0, day1)
    src_clusters = adata.obs.loc[coupling.obs.index, cluster_key]
    tgt_clusters = adata.obs.loc[coupling.var.index, cluster_key]

    df = pd.DataFrame(
        coupling.X,
        index=src_clusters.values,
        columns=tgt_clusters.values
    )
    agg = df.groupby(level=0).sum().T.groupby(level=0).sum().T
    agg = agg.div(agg.sum(axis=1), axis=0).fillna(0)
    return agg


def build_ancestor_graph(steps, adata, tmap_model, min_weight=MIN_EDGE_WEIGHT):
    """Build MultiDiGraph from transport maps"""
    G = nx.MultiDiGraph()

    for step_idx, (_, _, day0, day1) in enumerate(steps):
        matrix = cluster_transition_matrix(tmap_model, adata, day0, day1)
        for src in matrix.index:
            for tgt in matrix.columns:
                w = matrix.loc[src, tgt]
                if w > min_weight:
                    G.add_edge(str(src), str(tgt), weight=w, step=step_idx)

    return G


# ==========================================
# 2D LAYOUT FUNCTIONS
# ==========================================
def acyclic_skeleton(G):
    """Remove back edges for DAG ranking"""
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
    """Topological rank by longest path"""
    acyclic = acyclic_skeleton(G)
    layer = {}
    for n in nx.topological_sort(acyclic):
        preds = list(acyclic.predecessors(n))
        layer[n] = max((layer[p] for p in preds), default=-1) + 1
    return layer


def layered_layout(G, iterations=6):
    """Sugiyama-style layout with barycenter sweeps"""
    layer = longest_path_layers(G)
    max_layer = max(layer.values())
    order = {
        d: sorted([n for n in G.nodes if layer[n] == d], key=str)
        for d in range(max_layer + 1)
    }

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


# ==========================================
# 3D LAYOUT FUNCTIONS
# ==========================================
def bezier_3d(p0, p2, num_points=50):
    """Generate 3D quadratic Bezier curve with waterfall effect"""
    p0 = np.array(p0)
    p2 = np.array(p2)
    p1 = np.array([p0[0], p0[1], p2[2]])

    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 3))
    for i in range(num_points):
        curve[i] = (1-t[i])**2 * p0 + 2*(1-t[i])*t[i] * p1 + t[i]**2 * p2
    return curve


def circular_layout_3d(adata):
    """Arrange nodes in circles at each time point"""
    days = [0.0, 1.0, 2.0]
    labels = ['Ta', 'T1', '>T2']
    z_positions = {0.0: 2.0, 1.0: 1.0, 2.0: 0.0}
    pos_3d = {}
    node_sizes = {}

    circle_radius = 1.5

    for day, label in zip(days, labels):
        clusters = sorted(
            adata.obs[adata.obs['day'] == day]['seurat_clusters'].unique().astype(int)
        )
        n_clusters = len(clusters)
        z = z_positions[day]

        for i, cluster_id in enumerate(clusters):
            angle = 2 * np.pi * i / n_clusters
            x = circle_radius * np.cos(angle)
            y = circle_radius * np.sin(angle)

            node_name = f"{label}_{cluster_id}"
            pos_3d[node_name] = (x, y, z)

            cell_count = len(adata.obs[
                (adata.obs['day'] == day) & (adata.obs['seurat_clusters'] == cluster_id)
            ])
            node_sizes[node_name] = max(50, min(200, cell_count / 2))

    return pos_3d, node_sizes


# ==========================================
# PLOT 2D (Left panel)
# ==========================================
def plot_2d_lineage(ax, G, pos, step_labels, max_linewidth=10):
    """Plot 2D MATLAB-style layered graph"""
    pagerank = nx.pagerank(G, weight='weight')
    weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    ranks = pd.Series(weights).rank(pct=True).values

    step_cmap = cm.get_cmap('Set1', len(step_labels))

    # Draw edges
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

    # Draw nodes
    for node, (x, y) in pos.items():
        size = max(20, round(pagerank[node] * 150) + 4)
        ax.scatter(x, y, s=size, color='black', edgecolors='white',
                   linewidth=2, zorder=3)
        ax.text(x, y + 0.2, node, fontsize=16, fontweight='bold',
                ha='center', va='bottom', zorder=4)

    # Legend
    handles = [plt.Line2D([0], [0], color=step_cmap(i), lw=4, label=label)
               for i, label in enumerate(step_labels)]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=12)

    ax.set_title("2D Layered Lineage Graph", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.margins(0.15)


# ==========================================
# PLOT 3D (Right panel)
# ==========================================
def plot_3d_landscape(ax, G, adata, pos_3d, node_sizes, matrix_0_1, matrix_1_2):
    """Plot 3D cage-like trajectory landscape"""

    # Color setup
    ta_clusters = sorted(
        adata.obs[adata.obs['day'] == 0.0]['seurat_clusters'].unique().astype(int)
    )
    cmap = cm.get_cmap('Set1', len(ta_clusters))
    color_dict = {c: cmap(i) for i, c in enumerate(ta_clusters)}
    fallback_color = '#c0c0c0'

    t1_ancestor_map = {}
    for c1 in matrix_1_2.index:
        if c1 in matrix_0_1.columns:
            major_ancestor = matrix_0_1[c1].idxmax()
            t1_ancestor_map[int(c1)] = color_dict.get(int(major_ancestor), fallback_color)
        else:
            t1_ancestor_map[int(c1)] = fallback_color

    # Draw Ta -> T1 edges
    for c1 in matrix_0_1.index:
        for c2 in matrix_0_1.columns:
            w = matrix_0_1.loc[c1, c2]
            if w > THRESHOLD_3D:
                p0 = pos_3d.get(f"Ta_{c1}")
                p2 = pos_3d.get(f"T1_{c2}")

                if p0 and p2:
                    curve = bezier_3d(p0, p2)
                    color = color_dict.get(int(c1), fallback_color)
                    ax.plot(
                        curve[:, 0], curve[:, 1], curve[:, 2],
                        color=color,
                        linewidth=w * 25,
                        alpha=0.55,
                        zorder=1
                    )

    # Draw T1 -> >T2 edges
    for c1 in matrix_1_2.index:
        for c2 in matrix_1_2.columns:
            w = matrix_1_2.loc[c1, c2]
            if w > THRESHOLD_3D:
                p0 = pos_3d.get(f"T1_{c1}")
                p2 = pos_3d.get(f">T2_{c2}")

                if p0 and p2:
                    curve = bezier_3d(p0, p2)
                    color = t1_ancestor_map.get(int(c1), fallback_color)
                    ax.plot(
                        curve[:, 0], curve[:, 1], curve[:, 2],
                        color=color,
                        linewidth=w * 25,
                        alpha=0.55,
                        zorder=1
                    )

    # Plot nodes
    for node_name, (x, y, z) in pos_3d.items():
        size = node_sizes.get(node_name, 100)
        ax.scatter(
            x, y, z,
            color='black',
            s=size,
            edgecolors='white',
            linewidth=1.5,
            zorder=5
        )

        cluster_num = node_name.split('_')[1]
        ax.text(
            x, y, z + 0.15,
            cluster_num,
            color='black',
            fontsize=11,
            fontweight='bold',
            ha='center',
            zorder=6
        )

    ax.view_init(elev=15, azim=-60)
    ax.set_axis_off()
    ax.set_title("3D Cage-like Trajectory Landscape", fontsize=14, fontweight='bold')


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Loading data...")
    adata = load_expression_data()
    tmap_model = wot.tmap.TransportMapModel.from_directory(TMAP_DIR)

    print("Computing transition matrices...")
    matrix_0_1 = cluster_transition_matrix(tmap_model, adata, 0.0, 1.0)
    matrix_1_2 = cluster_transition_matrix(tmap_model, adata, 1.0, 2.0)

    print("Building graph...")
    G = build_ancestor_graph(STEPS, adata, tmap_model)

    print("Computing 2D layout...")
    pos_2d = layered_layout(G)

    print("Computing 3D layout...")
    pos_3d, node_sizes = circular_layout_3d(adata)

    # Create figure with subplots
    print("Rendering visualization...")
    fig = plt.figure(figsize=(20, 9))

    # 2D plot (left)
    ax_2d = fig.add_subplot(121)
    step_labels = [f'{a} -> {b}' for a, b, _, _ in STEPS]
    plot_2d_lineage(ax_2d, G, pos_2d, step_labels)

    # 3D plot (right)
    ax_3d = fig.add_subplot(122, projection='3d')
    plot_3d_landscape(ax_3d, G, adata, pos_3d, node_sizes, matrix_0_1, matrix_1_2)

    # Main title
    fig.suptitle("Waddington-OT: Cell Ancestor Graphs", fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("Done!")


if __name__ == '__main__':
    main()
