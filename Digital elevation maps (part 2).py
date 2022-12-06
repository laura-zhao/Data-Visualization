
# coding: utf-8

# In[1]:


"""
Template for week 13 project in Data Visualization

Plot various triangulated surfaces of digital elavation model data for Grand Canyon
https://pubs.usgs.gov/ds/121/grand/grand.html
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff


# In[2]:


# Resource paths
PLOTS_PATH = "plots/"
DATA_PATH = "data/"
GC_DEM = DATA_PATH + "gc_dem.tiff"

DEMS = {}

# Sub-region of interest defined as numpy slice objects
REGION = np.s_[1500:1800, 1300:1600]

# Custom colorscale/colormap for elevations
ELEV = ("rgb(5,10,172)",
        "rgb(34,46,193)",
        "rgb(63,83,215)",
        "rgb(92,119,236)",
        "rgb(134,155,228)",
        "rgb(190,190,190)",
        "rgb(220,170,132)", 
        "rgb(230,145,90)",
        "rgb(213,100,69)",
        "rgb(195,55,49)",
        "rgb(178,10,28)")


# In[3]:


###############################################################################
# Provided code from Project 12

def load_dem(dem_file):
    """
    Input: String dem_file
    
    Output: Numpy array of integer heights
    
    NOTE: The loaded height are in decimeters. Divide by 10 to
    return integer heights in meters
    """
    gc_image = plt.imread(dem_file)
    dem_array = np.array(gc_image) // 10

    return dem_array


# In[4]:


def compute_features(dem_array):
    """
    Input: Numpy array dem_array
    
    Output: Numpy array with boundary rows and columns trimmed
    """    
    float_dem = dem_array.astype(float)
    
    # Use numpy-friendly operations for speed
    horiz_error = float_dem[1 : -1, : -2] - 2 * float_dem[1 : -1, 1 : -1] + float_dem[1 : -1, 2 :]
    vert_error = float_dem[: -2, 1 : -1] - 2 * float_dem[1 : -1, 1 : -1] + float_dem[2 :, 1 : -1]
    
    feature_array = np.abs(horiz_error) + np.abs(vert_error)
    return feature_array


# In[5]:


def create_dems():
    """ Create some example dems for testing/plotting """
    
    # Small examples
    DEMS["2x2"] = np.array([[1, 2], [3, 4]])
    DEMS["3x5"] = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
    
    # Medium examples
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    ridge1 = np.minimum(x_grid - y_grid, -x_grid + y_grid)
    ridge2 = np.minimum(x_grid + y_grid, -x_grid - y_grid)
    DEMS["ridge"] = np.maximum(ridge1, ridge2)
    
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-2, 2, 41))
    DEMS["error"] = x_grid ** 2 - y_grid ** 2
    
    # Flip up/down due to difference in coordinate systems between image and 3D plots
    gc_dem = load_dem(GC_DEM)
    DEMS["region"] = np.flipud(gc_dem[REGION])

create_dems()


# In[6]:


####################################################################
# Part 1 - Generate topology of quad mesh for given grid shape

def make_quads(grid_shape):
    """
    Input: Tuple grid_shape consisting size of y and x dimensions of grid respectively
    
    Output: Tuple consisting of 2D array of 2D vertex positions
    and list of quads represented as tuples of vertex indices
    
    NOTE: The ordering of the returned vertex indices must be consistent 
    with the order returned by the ravel() method
    """
    rows = np.linspace(0, grid_shape[1] - 1, grid_shape[1])
    cols = np.linspace(0, grid_shape[0] - 1, grid_shape[0])
    rows, cols = np.meshgrid(rows, cols)
    rows = rows.ravel()
    cols = cols.ravel()
    geo = np.array(np.vstack([rows, cols]).T)
    lst = []
    for idx_r in range(grid_shape[0] - 1):
        for idx_c in range(grid_shape[1] - 1):
            start = idx_r * grid_shape[1] + idx_c
            lst.append([start, start + 1, start + grid_shape[1] + 1, start + grid_shape[1]])
        
    return (geo, np.array(lst))


# In[8]:


##################################################################
# Part 2 - Generate triangulations of elevation maps

def make_trimesh_fixed(z_grid, diagonal="ul_lr"):
    """
    Input: 2D numpy array z_grid of elevation values, optional string diagonal
    that specifies direction of quad diagonal - "ul_lr" or "ll_ur"
    
    Output: Tuple consisting of 2D numpy array of 3D vertex positions
    and a list of triangles represented as tuples of vertex indices
    """
    #print(z_grid)
    geo, topos = make_quads(tuple([z_grid.shape[0], z_grid.shape[1]]))
    #print(geo)
    geometry = []
    for cord in geo:
        geometry.append([cord[0], cord[1], z_grid[int(cord[1])][int(cord[0])]])
    topology = []
    if diagonal == "ul_lr":
        for topo in topos:
            topology.append([topo[0], topo[1], topo[2]])
            topology.append([topo[2], topo[3], topo[0]])
    else:
        for topo in topos:
            topology.append([topo[3], topo[0], topo[1]])
            topology.append([topo[1], topo[2], topo[3]])        
    return (np.array(geometry), np.array(topology))


# In[10]:


def make_trimesh_feature(z_grid):
    """
    Input: 2D numpy array z_grid of elevation values
    
    Output: Tuple consisting of 2D numpy array of 3D vertex positions
    and a list of triangles represented as tuples of vertex indices 
    for raveled TRIMMED grid
    """
    feature = compute_features(z_grid)
    trimmed_grid = z_grid[1:-1, 1:-1]
    def make_trimesh_ullr(topo):
        return [[topo[0], topo[1], topo[2]],
                [topo[2], topo[3], topo[0]]]
    def make_trimesh_llur(topo):
        return [[topo[3], topo[0], topo[1]],
                [topo[1], topo[2], topo[3]]]  
    geo, topos = make_quads(tuple([feature.shape[0], feature.shape[1]]))
    geometry = []
    for cord in geo:
        geometry.append([cord[0], cord[1], trimmed_grid[int(cord[1])][int(cord[0])]])
    topology = []
    feature = feature.ravel()
    for topo in topos: 
        ullr = feature[topo[0]] + feature[topo[2]]
        llur = feature[topo[3]] + feature[topo[1]]
        if ullr > llur:            
            topology += make_trimesh_ullr(topo)
        else:           
            topology += make_trimesh_llur(topo)
    return (np.array(geometry), np.array(topology))


# In[13]:


##################################################################################
# Part 3 - Plot triangular meshes computed from elevation maps


def plot_mesh3d(verts, tris, title="3D plot of a triangular mesh", camera=None):
    """
    Input: 2D numpy array verts of 3D vertex positions, list tris of tuples of vertex indices,
    optional string title, optional dictionary camera
    
    Output: plotly figure corresponing to a triangular mesh created via Mesh3D() 
    using the colorscale ELEV and the specified camera position. 
    The aspectio of the 3D plots should be similar that use in plot_elevation().
    """
    np_tris = np.array(tris)
    data = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                      i=np_tris[:, 0], j=np_tris[:, 1], k=np_tris[:, 2],
                      intensity=verts[:, 2], colorscale=ELEV)]
    fig = go.Figure(data=data)
    fig.update_layout(title=title, scene_camera=camera)
    fig.update_scenes(aspectratio=dict(x=1, y=1, z=0.1),
                      aspectmode="manual")
    return fig


# In[15]:


def plot_trisurf(verts, tris, title="3D plot of a triangular mesh", camera=None):
    """
    Input: 2D numpy array verts of 3D vertex positions, list tris of tuples of vertex indices,
    optional string title, optional dictionary camera
    
    Output: plotly figure corresponding to a triangular mesh create via create_trisurf() 
    using the colormap ELEV and the specified camera position. 
    The aspectio of the 3D plots should be similar that use in plot_elevation().
    """
    np_tris = np.array(tris)
    fig = ff.create_trisurf(x=verts[:, 0], y=verts[:, 1], 
                            z=verts[:, 2], simplices=tris, colormap=ELEV)
    fig.update_layout(title=title, scene_camera=camera)
    fig.update_scenes(aspectratio=dict(x=1, y=1, z=0.1),
                      aspectmode="manual")
    return fig


# In[17]:


# #Throws an error in create_trisurf()
# print(DEMS["error"])
# verts, tris = make_trimesh_fixed(DEMS["error"], diagonal="ul_lr")
# plot_trisurf(verts, tris, title="Fixed triangulation of error examples")

