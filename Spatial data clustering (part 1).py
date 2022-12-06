
# coding: utf-8

# In[1]:


"""
Template for week 10 project in Data Visualization

Compute closest pairs of points using both slow and fast method
Compare running times of both methods on random sets of points
Compute hierarchical clustering using closest pair code
"""

import time
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import project10_provided as provided


# In[2]:


# Resources paths
DATA_PATH = "data/"

# Set global flags
SAVE_PLOTS = False


# In[3]:


######################################################
# Provided code for computing distance between two clusters


def pair_distance(cluster_list, idx1, idx2):
    """
    Input: List cluster_list of clusters, integers idx1, idx2
    
    Output: Tuple (dist, idx1, idx2) where dist is distance between
    cluster_list[idx1] and cluster_list[idx2].
    
    Notes: Returned tuple always has idx1 < idx2. 
    """
    return (cluster_list[idx1].distance(cluster_list[idx2]), min(idx1, idx2), max(idx1, idx2))


# In[4]:


###################################################################
# Part 1 - Compute closest pairs using brute-force and divide-and-conquer

def slow_closest_pair(cluster_list):
    """
    Input: List cluster_list of clusters
    
    Output: Tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.
    
    Notes: Returned tuple always has idx1 < idx2. Implements O(n^2) time algorithm.
    """
    idx_min_1 = -1
    idx_min_2 = -1
    dist_min = float('inf')
    for idx1 in range(len(cluster_list)):
        for idx2 in range(idx1+1, len(cluster_list)):
            dist = pair_distance(cluster_list, idx1, idx2)
            if  dist[0] < dist_min:
                idx_min_1 = idx1
                idx_min_2 = idx2
                dist_min = dist[0]
        
    return (dist_min, idx_min_1, idx_min_2)


# In[5]:


def fast_binarysearch(sorted_cluster_list, left, right):
    """
    search both side to find the closest pair
    """

    if len(sorted_cluster_list[left:right]) <= 3: 
        dist_min = float('inf')
        idx_min_1 = -1
        idx_min_2 = -1
        for idx1 in range(left, right):
            for idx2 in range(idx1+1, right):
                dist = pair_distance(sorted_cluster_list, idx1, idx2)
                #print(idx1, idx2, dist)
                if  dist[0] < dist_min:
                    idx_min_1 = idx1
                    idx_min_2 = idx2
                    dist_min = dist[0]
                    #print(dist_min,"a")
        #print(dist_min,"b")
        return (dist_min, idx_min_1, idx_min_2)

    
    # Find the middle point 
    mid = (left+right) // 2
    midpoint = sorted_cluster_list[mid]
                            
    distl = fast_binarysearch(sorted_cluster_list, left, mid)
    distr = fast_binarysearch(sorted_cluster_list, mid, right)
    #print(distl,"a")
    #print(distr,"b")

    if distl[0] < distr[0]:
        d_sameside = distl
    else:
        d_sameside = distr
        

    d_diffside = closest_pair_strip(sorted_cluster_list, midpoint.horiz_center(), d_sameside[0])
    #print(d_diffside ,d_sameside)
    if d_diffside == (float('inf'), -1, -1):
        return d_sameside
    if d_sameside[0] > d_diffside[0]:
        return d_diffside
    else:
        return d_sameside
    
def fast_closest_pair(sorted_cluster_list):
    """
    Input: List sorted_cluster_list of clusters SORTED SUCH THAT THE HORIZONTAL POSIIONS
    OF THEIR CENTERS ARE IN ASCENDING ORDER
    
    Output: Tuple of the form (dist, idx1, idx2) where the centers of the clusters
    sorted_cluster_list[idx1] and sorted_cluster_list[idx2] have minimum distance dist.
    
    Note: Returned tuple always has idx1 < idx2. Implements O(n log(n)^2) algorithm
    """
    length = len(sorted_cluster_list)
    #print(fast_binarysearch(sorted_cluster_list, 0, length))
    return fast_binarysearch(sorted_cluster_list, 0, length)
    #return None


# In[6]:


def closest_pair_strip(cluster_list, horiz_center, half_width):
    """
    Input: List cluster_list of clusters,
    float horiz_center is the horizontal position of the strip's vertical center line
    float half_width is the half the width of the strip (i.e; the maximum horizontal distance
    that a cluster can lie from the center line)

    Output: Tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] lie in the strip and have minimum distance dist.
    
    NOTE: Returned tuple always has idx1 < idx2. Implements O(n log(n)) algorithm.
    """
    stripp = []
    stripp_dict = {}

    length = len(cluster_list)
    #print(cluster_list)
    for idx in range(length): 
        if abs(cluster_list[idx].horiz_center() - horiz_center) < half_width: 
            stripp.append(cluster_list[idx])
            stripp_dict[cluster_list[idx]] = idx
    stripp.sort(key=lambda cluster: cluster.vert_center())
    length2 = len(stripp)
    #print(stripP_dict)
    if length2 <= 1:
        return (float('inf'), -1, -1)
    dist_min = float('inf')
    idx_min_1 = -1
    idx_min_2 = -1
    for idx1 in range(length2-1):
        #print(idx1, min(idx+4, length2))
        for idx2 in range(idx1+1, min(idx1+4, length2)):
            #print(idx2)
            dist = pair_distance(stripp, idx1, idx2)
            #print(idx1, idx2, dist)
            if  dist[0] < dist_min:
                idx_min_1 = stripp_dict[stripp[idx1]]
                idx_min_2 = stripp_dict[stripp[idx2]]
                dist_min = dist[0]
    #print((dist_min[0], min(idx_min_1, idx_min_2),max(idx_min_1, idx_min_2)))
    return (dist_min, min(idx_min_1, idx_min_2), max(idx_min_1, idx_min_2))


# In[41]:


######################################################################
# Part 2 - Compute and plot hierarchical clusterings

def hierarchical_clustering(cluster_list, num_clusters):
    """
    Input: List cluster_list of clusters, interger num_clusters
    
    Output: List of clusters whose length is num_clusters
    
    NOTE: Function should mutate cluster_list to improve efficiency
    """
    clusters = len(cluster_list)
    cluster_list.sort(key=lambda cluster: cluster.horiz_center())
    while clusters > num_clusters:
        cluster_list.sort(key=lambda cluster: cluster.horiz_center())
        merge = fast_closest_pair(cluster_list)
        cluster_list[merge[2]].merge_clusters(cluster_list[merge[1]])
        cluster_list.pop(merge[1])
        clusters -= 1
    return cluster_list

