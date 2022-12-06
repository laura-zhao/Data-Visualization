
# coding: utf-8

# In[1]:


"""
Template for week 8 project in Data Visualization

Compute county centers from an SVG image of USA that includes county boundaries
Output a CSV file with FIPS code and county centers
"""

import math
import csv
from xml.dom import minidom
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Resource paths
PLOTS_PATH = "plots/"
DATA_PATH = "data/"

ATTRIBUTES = {}


# In[3]:


################################################################################
# Provided code

def dist(vert1, vert2):
    """
    Input: Tuples vert1, vert2 of floats
    
    Output: Euclidean distance between vert1 and vert2
    """
    return math.sqrt((vert1[0] - vert2[0]) ** 2 + (vert1[1] - vert2[1]) ** 2)


# In[4]:


def compute_verts_center(verts):
    """
    Input: List verts of vertices (tuples of two floats) on a path
    
    Output: Estimated center of the path as a tuple of two floats
    
    NOTE: Does not include an edge from the last vertex back to the first vertex
    """
    centroid = [0, 0]
    perimeter = 0
    for idx in range(len(verts) - 1):
        edge_length = dist(verts[idx], verts[idx + 1])
        centroid[0] += 0.5 * (verts[idx][0] + verts[idx + 1][0]) * edge_length
        centroid[1] += 0.5 * (verts[idx][1] + verts[idx + 1][1]) * edge_length
        perimeter += edge_length
    
    if perimeter == 0:
        center = verts[0]
    else:
        center = ((centroid[0] / perimeter), (centroid[1] / perimeter))
    return center


# In[5]:


#########################################################################
# Part 1 - Extract and display the "d" attribute of path elements in XML

def get_path_attributes(xml_file):
    """
    Input: String xml_file corresponding to an xml file containing a sequence of path elements
    
    Output: Dictionary whose keys are "id" attributes (FIPS code)
    and whose values are the corresponding "d" attributes (county boundary as a string)
    """
    xml_doc = minidom.parse(xml_file)     
    paths = {}
    for path in xml_doc.getElementsByTagName("path"):
        idnum = path.getAttribute("id")
        paths[idnum] = path.getAttribute("d")
    return paths     


# In[7]:


def get_d_verts(path_d, commands=('M', 'L', 'Z', 'z')):
    """
    Input: String path_d correspond to the "d" attribute of a path element
    
    Output: List of vertices (tuples of floats) for corresponding path
    
    NOTE: Ignores absolute path commands in path_d by default. Some floats may 
    include the character "e" due to scientific notation.
    """
    
    for chs in commands:
        path_d = path_d.replace(chs, ' ')
        
    lst = path_d.split() 

    verts = []
    for pair in lst:
        pairlst = pair.split(',')
        verts.append(tuple([float(pairlst[0]), float(pairlst[1])]))
    return verts


# In[9]:


def plot_paths(path_attributes, title="Paths extracted from an XML file"):
    """
    Input: Dictionary whose items are (path_id, path_d) pairs
    
    Output: matplotlib figure consisting of plot of 
    the corresponding paths encoded by the string path_d
    """
    
    fig, axs = plt.subplots()
    for dummy, path_d in path_attributes.items():
        bound = get_d_verts(path_d)
        x_ax = []
        y_ax = []
        for cor in bound:
            x_ax.append(cor[0])
            y_ax.append(cor[1])
        axs.plot(x_ax, y_ax)
        axs.set_title(title)
        axs.set_aspect('equal')
    return fig


# In[12]:


##############################################################################                                
# Part 2 - Create pandas dataframe for centers of paths and plots its contents

def make_centers_df(path_attributes):
    """
    Input: Dictionary path_attributes with items (path_id, path_d)
    
    Output: Dataframe with index path_id and columns x_coord, and
    y_coord where (x_coord, y_coord) is center of path encoded by path_d 
    """
    dic = {}
    dic['path_id'] = []
    dic['x_coord'] = []
    dic['y_coord'] = [] 
    for path_id, path_d in path_attributes.items():
        verts = get_d_verts(path_d)
        center = compute_verts_center(verts)
        dic['x_coord'].append(center[0])
        dic['y_coord'].append(center[1])
        dic['path_id'].append(path_id)
    frame = pd.DataFrame(dic, index=dic['path_id'])
    return frame[['x_coord', 'y_coord']]


# In[18]:


def write_centers_df(centers_df, file_name):
    """
    Input: Dataframe centers_df, string file_name
    
    Action: Write dataframe to specified file
    
    NOTE: The output file should have no headers
    """
    centers_df.to_csv(file_name, header=None)


# In[19]:


def plot_centers_df(centers_df, title="Path centers"):
    """
    Input: Dataframe centers_df
    
    Output: matplotlib figure consisting of plot of
    centers of counties using pandas scatter()
    """
    axs = centers_df.plot.scatter(x='x_coord', y='y_coord')
    axs.set_aspect('equal')
    axs.set_title(title)

