
# coding: utf-8

# In[7]:


"""
Template for week 11 project in Data Visualization

Compute k-means clustering
Plot clusters and centers for both clustering methods
Compute and compare distortions for both clustering methods
"""
import math
import random
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import project10_provided as provided
import project10_solution as project10


# In[8]:


# Resource paths
DATA_PATH = "data/"

# Set global flags
SAVE_PLOTS = False


# In[9]:


######################################################################
# Part 1 - Compute and plot k-means clusterings in various styles

def init_cluster_center(cluster_list, num_clusters):
    """
    init cluster centers
    """
    copy_clusters = cluster_list
    copy_clusters.sort(key=lambda x: x.total_population(), reverse=True)
    initial_center = [[copy_clusters[i].horiz_center(), 
                       cluster_list[i].vert_center()] for i in range(num_clusters)]
    return initial_center


def distance(cluster, center):
    """
    get distence from cluster to center
    """
    vert_dist = cluster.vert_center() - center[1]
    horiz_dist = cluster.horiz_center() - center[0]
    return math.sqrt(vert_dist ** 2 + horiz_dist ** 2)

def kmeans_clustering(cluster_list, num_clusters, num_iterations):
    """
    Input: List of clusters, number of clusters, number of iterations
    
    Output: List of clusters whose length is num_clusters
    
    NOTE: The function may mutate cluster_list
    
    """

    initial_center = init_cluster_center(cluster_list, num_clusters)
    for dummy in range(num_iterations):
        clusters = [provided.Cluster(set(), 0, 0, 0, 0) for dummy2 in range(num_clusters)]
        for cluster in cluster_list:
            dis = float('inf')
            cur = 0
            for idx in range(num_clusters):
                if distance(cluster, initial_center[idx]) < dis:
                    dis = distance(cluster, initial_center[idx])
                    cur = idx
            clusters[cur] = clusters[cur].merge_clusters(cluster)
        for idx in range(num_clusters):
            initial_center[idx] = [clusters[idx].horiz_center(), clusters[idx].vert_center()]
    return clusters


# In[12]:


MAP = DATA_PATH + "USA_Counties_1000x634.png"
COLORS = ['Aqua', 'Yellow', 'Blue', 'Fuchsia', 'Black', 'Green', 
          'Lime', 'Maroon', 'Navy', 'Olive', 'Orange', 'Purple', 'Red', 'Brown', 'Teal']
FIPS, X_CENTER, Y_CENTER, POP, RISK = ("FIPS", "x center", 
                                       "y center", "population", "cancer risk")
COL_TYPES = {FIPS : "str", X_CENTER : "float", Y_CENTER : "float", POP : "int", RISK : "float"}
PIXELS_PER_INCH = 80
SIZE_CONSTANT = math.pi / (200.0 ** 2)
def marker_size(population):
    """
    Input: integer population
    
    Output: Area of circle in pixels proportional to population for use in plt.scatter
    """
    return  SIZE_CONSTANT * population
def plot_clusters_centers(cluster_list, risk_frame, 
                          title="Clusters of counties (with center) based on cancer risk", 
                          save_plot=False):
    """
    Input: List cluster_list, dataframe risk_frame (indexed by FIPS)
    
    Output: matplotlib figure of USA map with cluster members 
    connected to corresponding center
    """
    

    fig = provided.plot_image(MAP, title)
    axs = fig.axes[0]

    # draw the counties colored by cluster on the map
    for cluster_idx in range(len(cluster_list)):
        cluster = cluster_list[cluster_idx]
        cluster_color = COLORS[cluster_idx % len(COLORS)]
        x_centers = []
        y_centers = []
        sizes = []
        for fips_code in cluster.fips_codes():
            x_centers.append(risk_frame.loc[fips_code, X_CENTER])
            y_centers.append(risk_frame.loc[fips_code, Y_CENTER])

            axs.plot([cluster.horiz_center(), risk_frame.loc[fips_code, X_CENTER]],
                     [cluster.vert_center(), risk_frame.loc[fips_code, Y_CENTER]], 
                     cluster_color)

            sizes.append(marker_size(risk_frame.loc[fips_code, POP]))          
        axs.scatter(x=x_centers, y=y_centers, s=sizes, c=cluster_color)
        radium = pow(marker_size(cluster.total_population())/math.pi, 0.5)
        cir = Circle((cluster.horiz_center(), cluster.vert_center()), radius=radium, 
                     zorder=10, facecolor='none', edgecolor='black')
        axs.add_artist(cir)

    if save_plot:
        fig.savefig(DATA_PATH + title + ".png")
    return fig


# In[14]:


##################################################################################################
# Part 2 - Plot running times for closest pair methods and distortion for clustering methods

# Provided code
def gen_random_cluster(num_clusters):
    """
    Input: integer num_clusters
    
    Output: List of empty clusters of length num_clusters
    with centers randomly distributed in [-1, 1]^2
    """
    
    cluster_list = []
    for dummy_index in range(num_clusters):
        new_cluster = provided.Cluster(set([]), 2 * random.random() - 1, 
                                       2 * random.random() - 1, 0, 0)
        cluster_list.append(new_cluster)
    return cluster_list


# In[15]:


def plot_closest_pair_times(max_clusters, title="Running time of slow vs. fast closest pairs"):
    """
    Input: Integer max_clusters
    
    Output: matplotlib figure consisting of plot comparing
    of running times slow_closest_pair() and fast_closest_pair()
    for specified number of clusters created using gen_random_cluster().
    """  
    fig, dummy = plt.subplots()
     
    individual = []
    together = []
    for num_cluster in range(max_clusters):
        cluster_list = gen_random_cluster(num_cluster)

        begin_time = time.time()
        project10.slow_closest_pair(cluster_list)
        end_time = time.time()
        individual.append((num_cluster, end_time - begin_time))
    
        begin_time = time.time()
        project10.fast_closest_pair(cluster_list)
        end_time = time.time()
        together.append((num_cluster, end_time - begin_time))
        
    ind = np.array(individual)
    tog = np.array(together)
    
    plt.plot(ind[1:, 0], ind[1:, 1], label="slow_closest_pair")
    plt.plot(tog[1:, 0], tog[1:, 1], label="fast_closest_pair")
    plt.title("Running times for slow closest vs fast closest")
    #plt.title("Running times for plotting together")
    plt.xlabel("Number of clusters")
    plt.ylabel("Time in seconds")
    plt.legend()
    plt.show()
    return fig


# In[ ]:


def hierarchical_distortion(cluster_list, risk_frame, min_clusters, max_clusters):
    """
    Input: List cluster_list of clusters, dataframe risk_frame, 
    integers min_clusters, max_clusters

    Output: List whose entries are [num_clusters, distortion] where distortion 
    is the distortion associated with the hierachical clustering of size num_clusters
    """
    copy = [cluster_list[i].copy() for i in range(len(cluster_list))]
    dist = [0 for dummy in range(len(copy) + 1)]
    err = 0
    for dummy_idx in range(min_clusters, len(copy)):
        copy.sort(key=lambda cluster: cluster.horiz_center())
        (dummy_dist, idx1, idx2) = project10.fast_closest_pair(copy)
        cluster1 = copy[idx1]
        cluster2 = copy[idx2]
        prev = cluster1.cluster_error(risk_frame) + cluster2.cluster_error(risk_frame)
        if len(cluster1.fips_codes()) >= len(cluster2.fips_codes()):    
            cluster1.merge_clusters(cluster2)
            err += cluster1.cluster_error(risk_frame) - prev
            copy.pop(idx2)
        else:
            cluster2.merge_clusters(cluster1)
            err += cluster2.cluster_error(risk_frame) - prev
            copy.pop(idx1)
        dist[len(copy)] = err
    ans = [[idx, dist[idx]] for idx in range(min_clusters, max_clusters + 1)]    
    return ans


# In[ ]:


DISTORTION_SCALE = 10 ** 11

def plot_distortion(risk_frame, min_clusters, max_clusters, 
                    title="Distortion for hierarchical vs k-means clusterings"):
    """
    Input: dataframe risk_frame, integers min_clusters, max_clusters, optional string title
    
    Output: matplotlib figure generated by computing and plotting 
    distortions for hierarchical and k-means clustering algorithms
    """
    
    
    fig, dummy = plt.subplots()
    dist = []
    cluster_list = provided.dataframe_to_singleton_clusters(risk_frame)
    for num in range(min_clusters, max_clusters + 1):
        clusters = kmeans_clustering(cluster_list=cluster_list, 
                                     num_clusters=num, num_iterations=5)
        error = 0
        for cluster in clusters:
            error += cluster.cluster_error(risk_frame)
        dist.append([num, error])
    
    dist_hir = hierarchical_distortion(cluster_list, risk_frame, min_clusters, max_clusters)

    ind = np.array(dist)
    tog = np.array(dist_hir)
    
    plt.plot(ind[1:, 0], ind[1:, 1], label="k-means")
    plt.plot(tog[1:, 0], tog[1:, 1], label="hierarchical")
    plt.title("k-means vs hierarchical")
    plt.xlabel("Number of clusters")
    plt.ylabel("distortion")
    plt.legend()
    plt.show()
    return fig
    

