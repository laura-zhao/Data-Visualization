"""
Waddington-OT: 3D Evolutionary Trajectory Landscape

Creates a 3D visualization where:
- X,Y: nodes at each time point arranged in a circle
- Z: developmental time axis (Ta at top, >T2 at bottom)
- Edges: Bezier curves colored by lineage
- Result: cage-like structure as seen in literature
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# ==========================================
# Bezier curve for smooth 3D edges
# ==========================================
def bezier_3d(p0, p2, num_points=50):
    """Generate a 3D quadratic Bezier curve from p0 to p2"""
    p0 = np.array(p0)
    p2 = np.array(p2)
    # Control point: intermediate Z creates waterfall effect
    p1 = np.array([p0[0], p0[1], p2[2]])

    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 3))
    for i in range(num_points):
        curve[i] = (1-t[i])**2 * p0 + 2*(1-t[i])*t[i] * p1 + t[i]**2 * p2
    return curve

# ==========================================
# Arrange nodes in circles by time point
# ==========================================
days = [0.0, 1.0, 2.0]
labels = ['Ta', 'T1', '>T2']
z_positions = {0.0: 2.0, 1.0: 1.0, 2.0: 0.0}  # Top to bottom
pos_3d = {}
node_sizes = {}  # For node size encoding

# Circle radius (adjustable)
circle_radius = 1.5

for day, label in zip(days, labels):
    # Get all clusters at this time point
    clusters = sorted(
        adata.obs[adata.obs['day'] == day]['seurat_clusters'].unique().astype(int)
    )
    n_clusters = len(clusters)
    z = z_positions[day]

    # Arrange clusters in a circle
    for i, cluster_id in enumerate(clusters):
        # Uniform angular distribution
        angle = 2 * np.pi * i / n_clusters
        x = circle_radius * np.cos(angle)
        y = circle_radius * np.sin(angle)

        node_name = f"{label}_{cluster_id}"
        pos_3d[node_name] = (x, y, z)

        # Node size: count of cells in cluster
        cell_count = len(adata.obs[
            (adata.obs['day'] == day) & (adata.obs['seurat_clusters'] == cluster_id)
        ])
        node_sizes[node_name] = max(50, min(200, cell_count / 2))

# ==========================================
# Color mapping by lineage
# ==========================================
ta_clusters = sorted(
    adata.obs[adata.obs['day'] == 0.0]['seurat_clusters'].unique().astype(int)
)
cmap = cm.get_cmap('Set1', len(ta_clusters))
color_dict = {c: cmap(i) for i, c in enumerate(ta_clusters)}
fallback_color = '#c0c0c0'

# Map T1 clusters to their major Ta ancestor for color inheritance
t1_ancestor_map = {}
for c1 in matrix_1_2.index:
    if c1 in matrix_0_1.columns:
        major_ancestor = matrix_0_1[c1].idxmax()
        t1_ancestor_map[int(c1)] = color_dict.get(int(major_ancestor), fallback_color)
    else:
        t1_ancestor_map[int(c1)] = fallback_color

# ==========================================
# 3D Visualization
# ==========================================
fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(111, projection='3d')

threshold = 0.005

# Draw Ta -> T1 edges
for c1 in matrix_0_1.index:
    for c2 in matrix_0_1.columns:
        w = matrix_0_1.loc[c1, c2]
        if w > threshold:
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
        if w > threshold:
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

    # Add cluster label
    cluster_num = node_name.split('_')[1]
    ax.text(
        x, y, z + 0.12,
        cluster_num,
        color='black',
        fontsize=12,
        fontweight='bold',
        ha='center',
        zorder=6
    )

# ==========================================
# View and styling
# ==========================================
ax.view_init(elev=15, azim=-60)
ax.set_axis_off()

# Time axis indicator
ax.text2D(
    0.02, 0.98,
    "Time / Developmental Axis",
    transform=ax.transAxes,
    fontsize=13,
    fontweight='bold'
)
ax.annotate(
    '',
    xy=(0.045, 0.88),
    xytext=(0.045, 0.96),
    xycoords='axes fraction',
    arrowprops=dict(facecolor='black', width=2, headwidth=8)
)

plt.title("Waddington-OT: 3D Evolutionary Trajectory Landscape", fontsize=18, pad=20)
plt.tight_layout()
plt.show()
