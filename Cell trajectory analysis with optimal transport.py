"""
Cell trajectory analysis with optimal transport (Waddington-OT)

Once wot.OTModel.compute_all_transport_maps() has written the coupling
matrices to disk (e.g. tmaps/bladder_<t0>_<t1>.h5ad), there is no need to
recompute anything after restarting the kernel. This script shows how to
reload the saved transport maps and the original expression data, rebuild
the cell sets / populations used for trajectory and fate analysis, and
visualize the results on the existing UMAP/t-SNE embedding.
"""

import wot
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Resource paths
H5AD_PATH = '/home/ZKX/cdh6/hjx.h5ad'
TMAP_DIR = 'tmaps/bladder'

TIME_MAP = {'Ta': 0.0, 'T1': 1.0, '>T2': 2.0}
TARGET_TIME = 2.0


def load_expression_data():
    """Re-load the original AnnData and recompute the derived columns
    (day, x, y) that were added before the OT model was fit. This is
    cheap, so it is fine to redo on every kernel restart."""
    adata = wot.io.read_dataset(H5AD_PATH)
    adata.obs['day'] = adata.obs['orig.ident'].map(TIME_MAP).astype(float)

    if 'X_umap' in adata.obsm:
        adata.obs['x'] = adata.obsm['X_umap'][:, 0]
        adata.obs['y'] = adata.obsm['X_umap'][:, 1]
    elif 'X_tsne' in adata.obsm:
        adata.obs['x'] = adata.obsm['X_tsne'][:, 0]
        adata.obs['y'] = adata.obsm['X_tsne'][:, 1]

    return adata


def load_transport_maps():
    """Reload the transport maps saved by compute_all_transport_maps().
    No OT computation happens here, this just reads the saved h5ad files."""
    return wot.tmap.TransportMapModel.from_directory(TMAP_DIR)


def build_target_cell_sets(adata, target_time):
    """Recreate the >T2 cluster cell sets that were passed to
    population_from_cell_sets() before the kernel was restarted."""
    cell_sets = {}
    clusters = adata.obs[adata.obs['day'] == target_time]['seurat_clusters'].unique()
    for cluster in clusters:
        mask = (adata.obs['day'] == target_time) & (adata.obs['seurat_clusters'] == cluster)
        cell_sets[f'>T2_cluster_{cluster}'] = adata.obs[mask].index.tolist()
    return cell_sets


def plot_trajectory_probabilities(adata, trajectory_ds):
    """One UMAP panel per target cluster, colored by the probability that
    each earlier cell is an ancestor of that cluster."""
    names = list(trajectory_ds.var.index)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        probs = pd.Series(trajectory_ds[:, name].X.flatten(), index=trajectory_ds.obs.index)
        xy = adata.obs.loc[probs.index, ['x', 'y']]
        sc = ax.scatter(xy['x'], xy['y'], c=probs.values, s=4, cmap='plasma')
        ax.set_title(name)
        ax.axis('off')
        fig.colorbar(sc, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig('pictures/trajectory_probabilities.png', dpi=150)
    return fig


def plot_dominant_fate(adata, fate_ds, target_time):
    """Color each early-timepoint cell by its most likely >T2 cluster fate."""
    fate_df = pd.DataFrame(fate_ds.X, index=fate_ds.obs.index, columns=fate_ds.var.index)
    fate_df['dominant_fate'] = fate_df.idxmax(axis=1)
    fate_df[['x', 'y']] = adata.obs.loc[fate_df.index, ['x', 'y']]

    fig, ax = plt.subplots(figsize=(10, 10))

    late_mask = adata.obs['day'] == target_time
    ax.scatter(adata.obs.loc[late_mask, 'x'], adata.obs.loc[late_mask, 'y'],
               c='lightgray', s=4, alpha=0.3, label='>T2 (target)')

    codes = pd.Categorical(fate_df['dominant_fate']).codes
    sc = ax.scatter(fate_df['x'], fate_df['y'], c=codes, s=4, cmap='tab10')

    ax.axis('off')
    ax.set_title('Predicted dominant >T2 fate for Ta / T1 cells')
    fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig('pictures/dominant_fate.png', dpi=150)
    return fig


if __name__ == '__main__':
    adata = load_expression_data()
    tmap_model = load_transport_maps()

    cell_sets = build_target_cell_sets(adata, TARGET_TIME)
    populations = tmap_model.population_from_cell_sets(cell_sets, at_time=TARGET_TIME)

    trajectory_ds = tmap_model.trajectories(populations)
    fate_ds = tmap_model.fates(populations)

    plot_trajectory_probabilities(adata, trajectory_ds)
    plot_dominant_fate(adata, fate_ds, TARGET_TIME)

    plt.show()
