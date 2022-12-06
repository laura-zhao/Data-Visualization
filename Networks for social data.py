
# coding: utf-8

# In[18]:


"""
Template for week 7 project in Data Visualization

Compute simple graph layouts using optimization and plot community structure
"""

import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize as opt
import community


# In[19]:


# Resource paths
PLOTS_PATH = "plots/"
DATA_PATH = "data/"



# In[20]:


###############################################################################
# Provided code 

def make_graph(nodes, edges, name=None):
    """
    Input: list nodes, list edges consisting of tuples of integer node indices
    optional string name
    
    Output: networkx graph
    """
    
    grph = nx.Graph()
    grph.add_nodes_from(nodes)
    networkx_edges = [[nodes[edge[0]], nodes[edge[1]]] for edge in edges]
    grph.add_edges_from(networkx_edges)
    if name:
        grph.name = name
    
    return grph


# In[21]:


NODE_RANGE = [-1, 1]

def random_node_pos():
    """
    Output: Tuple of random floats in NODE_RANGE
    """

    node_pos = (random.uniform(NODE_RANGE[0], NODE_RANGE[1]),
                random.uniform(NODE_RANGE[0], NODE_RANGE[1]))
    return node_pos


# In[22]:


def random_layout(grph, seed=None):
    """
    Input: graph grph, float seed
    Output: dictionary indexed by nodes whose values are 2D node positions
    """
    if seed:
        random.seed(seed)
        
    layout = {}
    for node in grph.nodes():
        layout[node] = random_node_pos()
    
    return layout


# In[23]:


def plot_graph(grph, layout, title, with_labels=True, node_colors='y', axs=None):
    """
    Input: graph grph, dictionary layout of 2D node positions, string title
    optional with_labels, node_colors as defined in draw_networkx
    optional axes axs
    
    Output: matplotlib figure with specified axes updated to
    include graph drawn using draw_networkx with node outlines being black
    """
    
    base_plot = axs is None
    if base_plot:
        fig, axs = plt.subplots()
    else:
        fig = axs.figure
    
    axs.set_title(title)
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_aspect("equal")
    fig.tight_layout()
    
    nx.draw_networkx(grph, pos=layout, with_labels=with_labels, 
                     node_color=node_colors, cmap="terrain", ax=axs)
    axs.collections[0].set_edgecolor('k') 
    
    return fig


# In[25]:


###############################################################################
# Part 1 - Compute energy-based layout for graphs 

def get_node_indices(grph):
    """
    Input: graph grph
    
    Output: Dictionary whose keys are nodes in grph and whose values 
    are corresponding positions of nodes in grph.nodes()
    """
    nodes_dict = {}
    nodes_lst = list(grph.nodes)
    for idx in range(len(nodes_lst)):
        nodes_dict[nodes_lst[idx]] = idx 
    return nodes_dict


# In[26]:


def distance_error(flat_node_pos, path_lengths):
    """
    Input: 1D numpy array flat_node_pos of the form [x0 y0 x1 y1 ...],
    nested dictionary path_lengths of path lengths keyed by node indices
    
    Output: Sum of squares of differences between path lengths
    and geometric distances between pairs of nodes (based on values in flat_node_pos) 
    
    Note: path_lengths will be computed via all_pairs_shortest_path_length()
    """
    nodes_num = len(path_lengths.keys())
    error = 0
    for idx1 in range(nodes_num):
        for idx2 in range(idx1+1, nodes_num):
            x_diff = flat_node_pos[2*idx1]-flat_node_pos[2*idx2]
            y_diff = flat_node_pos[2*idx1 + 1]-flat_node_pos[2*idx2 + 1]
            geo = x_diff**2 + y_diff**2
            geo_dis = math.sqrt(geo)
            path_length = path_lengths[idx1][idx2]
            error += (path_length - geo_dis)**2
    return error*2   


# In[27]:


def distance_layout(grph, seed=None):
    """
    Input: graph grph, optional integer seed
    
    Output: Dictionary of node positions keyed by nodes 
    and computed using shortest path distances
    
    Note: The initial guess for opt.minimize() should be generated via n calls 
    to random_node_pos() where n is the number of nodes in grph.
    """
    if seed:
        random.seed(seed)
    init = []
    for dummy in range(len(grph.nodes)):
        ran = list(random_node_pos()) 
        init.append(ran[0])
        init.append(ran[1])
        
    all_shortest = dict(nx.all_pairs_shortest_path(grph))
    node_dict = get_node_indices(grph)
    graph_copy = {}
    for key, value in all_shortest.items():
        nested_dict = {}
        for key2, value2 in value.items():
            nested_dict[node_dict[key2]] = len(value2) - 1
        graph_copy[node_dict[key]] = nested_dict
    
    def make_quad_fun(minimum):
        def quad_fun(vals):
            return distance_error(vals, minimum)
        return quad_fun
    
    quad_fun = make_quad_fun(graph_copy)
    computed = opt.minimize(quad_fun, init).x
    pos_dict = {}
    dum = 0
    for node in grph.nodes:
        pos_dict[node] = (computed[2*dum], computed[2*dum + 1])
        dum += 1
    return pos_dict
          


# In[28]:


#########################################################################
# Student code comparing layout methods (peer-graded)

def plot_spring_vs_distance(grph, with_labels=True, node_colors='y', seed=None):
    """
    Input: graph grph, optional bool with_labels, optional string node_colors, optional int seed
    
    Output: matplotlib figure consisting of side-by-side comparision of 
    grph using spring and distance layouts
    """
    if seed:
        random.seed(seed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    ax1.set_title("spring layout")
    ax2.set_title("distance layout")
    nx.draw(grph, with_labels=True, node_color="y", ax=ax1)
    lay = distance_layout(grph)
    nx.draw(grph, pos=lay, with_labels=True, node_color="y")
 
    return fig


# In[30]:


##########################################################
# Part 2 - Compute and plot communities

def get_communities(grph, seed=None):
    """
    Input: graph grph, optional int seed
    
    Output: List of integers indicating the community for each
    corresponding node in grph
    """
    if seed:
        random.seed(seed)
    part = community.best_partition(grph)
    part_list = []
    for node in grph.nodes:
        part_list.append(part[node])
    return part_list


# In[31]:


def plot_caveman_communities(num_cliques, clique_size, prob, seed=None):
    """
    Input: integers num_cliques, clique_sizes, floats prob, seed
    
    Output: matplotlib figure containing plot of communities in
    relaxed caveman graphs using spring layout
    """
    if seed:
        random.seed(seed)
    grph = nx.relaxed_caveman_graph(num_cliques, clique_size, prob, seed)
    layout = nx.spring_layout(grph, seed=seed)
    partition = get_communities(grph, seed)
    fig, axs = plt.subplots()
    title = ("spring Layout Relaxed Caveman Graph with " + "n=" + 
             str(num_cliques) + " s="+str(clique_size) +" p="+str(prob))
    plot_graph(grph, layout, title, with_labels=True, node_colors=partition, axs=axs)
    return fig


# In[33]:


def plot_facebook_communities(seed=None):
    """
    Input: optional int seed
    
    Output: matplotlib figure consisting of communities for Facebook ego network from
    https://blog.dominodatalab.com/social-network-analysis-with-networkx/
    """
    if seed:
        random.seed(seed)
    grph = nx.read_edgelist(DATA_PATH+"facebook_combined.txt", create_using=nx.Graph(), nodetype=int)
    partition = get_communities(grph, seed)
    layout = nx.spring_layout(grph, seed=seed)
    fig, axs = plt.subplots()
    nx.draw_networkx(grph, layout, node_size=35, node_color=partition, 
                     ax=axs, with_labels=False, cmap="terrain")
    axs.set_title("Facebook Community")
    return fig

