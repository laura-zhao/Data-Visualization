
# coding: utf-8

# In[52]:


"""
Template for week 12 project in Data Visualization

Plot various visualizations of digital elavation model data for Grand Canyon
https://pubs.usgs.gov/ds/121/grand/grand.html
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tifffile


# In[53]:


# Resource paths
PLOTS_PATH = "plots/"
DATA_PATH = "data/"
GC_DEM = DATA_PATH + "gc_dem.tiff"

DEMS = {}

# Sub-regions of interest defined as numpy slice objects
ALL = np.s_[:, :]
REGION1 = np.s_[1400:1900, 1200:1700]
REGION2 = np.s_[1250:1750, 1750:2250]

# Min and max heights in meters for Grand Canyon DEM
ZMIN = 489
ZMAX = 2770

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


# In[54]:


############################################################
# Part 1 - Load, plot, and contour images

def load_dem(dem_file):
    """
    Input: String dem_file
    
    Output: Numpy array of uint16 heights
    
    NOTE: The loaded height are in decimeters. Divide by 10 to
    return integer heights in meters
    """
    original_array = tifffile.imread(dem_file)     # tifffile correctly reads old format
    new_array = np.array(original_array / 10, dtype=np.uint16)

    return new_array

load_dem(GC_DEM)


# In[56]:


def plot_image(image_array, title="Image plot of a DEM", vmin=None, vmax=None, cmap=None):
    """
    Input: Numpy array image_array, optional string title, optional 
    integers vmin, vmax, optional colormap cmap
    
    Output: matplotlib figure consisting of a plot of the 
    image with values in image_array
    """
    #print(image_array,shape)
    fig, axs = plt.subplots()
    show = axs.imshow(image_array, cmap=cmap, vmin=vmin, vmax=vmax)
    axs.set_title(title)
    fig.colorbar(show)
    
    return fig


# In[59]:


def plot_contour(dem_array, title="Contour plot of a DEM", filled=False, 
                 vmin=None, vmax=None, cmap=None):
    """
    Input: Numpy array dem_array, optional string title, optional boolean filled
    optional integers vmin, vmax, optional colormap cmap
    
    Output: mapltolib figure consisting of a plot of the contours 
    of image in dem_array, use contour() or contourf() based on value of filled
    """
    dem = dem_array.copy()
    dem = dem[::-1]
    fig, axs = plt.subplots()
    axs.set_title(title)
    if filled:
        css = axs.contourf(dem, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        fig.colorbar(css)
    else:
        css = axs.contour(dem, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        fig.colorbar(css)
    return fig


# In[61]:


####################################################################
# Part 2 - Compute and plot features of dems

def compute_features(dem_array):
    """
    Input: Numpy array dem_array
    
    Output: Numpy array of float64 with boundary rows and columns trimmed
    """    
    dem_array = np.array(dem_array, dtype='float')
    first_abs = np.array([abs(dem_array[i - 1, :] - 2 * dem_array[i, :] + dem_array[i + 1, :]) 
                          for i in range(1, dem_array.shape[0] - 1)])
    second_abs = np.array([abs(dem_array[:, i - 1] - 2 * dem_array[:, i] + dem_array[:, i + 1]) 
                           for i in range(1, dem_array.shape[1] - 1)])
    second_abs = second_abs.transpose()
    feat = first_abs[:, 1 : -1] +  second_abs[1 : -1, :]
    return np.array(feat, dtype='float')


# In[62]:


def plot_features(dem_array, title="DEM and its features", vmin=None, vmax=None, cmap=None):
    """
    Input: Numpy array dem_array, optional string title,
    optional integers vmin, vmax, optional colormap cmap
    
    Output: matplotlib figure consisting of a 1 x 2 array 
    of plots of DEM image and its features
    """
    features = compute_features(dem_array)
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(dem_array[1:-1, 1:-1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.imshow(features, cmap="Greys_r")
    fig.suptitle(title)
    return fig


# In[65]:


#######################################################################
# Part 3 - Compute downsampled version of dem and plot as a 3D elevation map

def downsample_dem(dem_array):
    """
    Input: Numpy array dem_array whose dimensions are divisible by two
    
    Output: Numpy array of integers with each dimension being halved
    """
    dem_array = np.array(dem_array, dtype="float")
    first = np.array([dem_array[i, :] + dem_array[i + 1, :] 
                      for i in range(0, dem_array.shape[0], 2)])
    #print(first)
    second = np.array([(first[:, i] + first[:, i + 1])/4 
                       for i in range(0, first.shape[1], 2)])
    second = second.transpose()
    #print(second)
    return np.array(second, dtype='int')


# In[67]:


def plot_elevation(dem_array, cmin=None, cmax=None, title="Grand Canyon surface plot"):
    """
    Input: Numpy array dem_array, optional numbers cmin, cmax, optional string title
    
    Output: plotly figure corresponding to 3D elevation map of dem_array using the 
    colorscale ELEV with specified mininum and maximum elevations using go.Surface()
    
    The aspect ratio of the 3D plot should preserve the relative lengths of the x and y 
    coordinate ranges while the length of the z range should be scaled to be
    10% of the minimum of the lengths of the x and y ranges.
    
    NOTE: The function should also write the figure to HTML in the files linked in the
    markdown cell at the end of this notebook.
    """    
    dem = dem_array.copy()
    dem = dem[::-1]
    fig = go.Figure(data=[go.Surface(z=dem, colorscale=ELEV, cmin=cmin, cmax=cmax)])
    fig.update_layout(title=title)
    fig.update_scenes(aspectratio=dict(x=1, y=1, z=0.1),
                      aspectmode="manual")

    fig.write_html(PLOTS_PATH + title + '.html')
    return fig

