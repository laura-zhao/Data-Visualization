
# coding: utf-8

# In[11]:


"""
Template for week 14 project in Data Visualization

Load binary CT data and plot the contours of the resulting volume
http://graphics.cs.ucdavis.edu/~okreylos/PhDStudies/Spring2000/ECS277/DataSets.html
"""

import struct
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import skimage


# In[12]:


# Note that file names are caps-sensitive on Unix
PLOTS_PATH = "plots/"
DATA_PATH = "data/"
SIMPLE = DATA_PATH + "simple.vol"
C60_64 = DATA_PATH + "C60_64.vol"
C60_128 = DATA_PATH + "C60_128.vol"
FOOT = DATA_PATH + "Foot.vol"
SKULL = DATA_PATH + "Skull.vol"

VOLS = {}

# Custom colorscale for CT volumes
PL_BONE = [
    [0.0, 'rgb(0, 0, 0)'],
    [0.05, 'rgb(10, 10, 14)'],
    [0.1, 'rgb(21, 21, 30)'],
    [0.15, 'rgb(33, 33, 46)'],
    [0.2, 'rgb(44, 44, 62)'],
    [0.25, 'rgb(56, 55, 77)'],
    [0.3, 'rgb(66, 66, 92)'],
    [0.35, 'rgb(77, 77, 108)'],
    [0.4, 'rgb(89, 92, 121)'],
    [0.45, 'rgb(100, 107, 132)'],
    [0.5, 'rgb(112, 123, 143)'],
    [0.55, 'rgb(122, 137, 154)'],
    [0.6, 'rgb(133, 153, 165)'],
    [0.65, 'rgb(145, 169, 177)'],
    [0.7, 'rgb(156, 184, 188)'],
    [0.75, 'rgb(168, 199, 199)'],
    [0.8, 'rgb(185, 210, 210)'],
    [0.85, 'rgb(203, 221, 221)'],
    [0.9, 'rgb(220, 233, 233)'],
    [0.95, 'rgb(238, 244, 244)'],
    [1.0, 'rgb(255, 255, 255)']
]


# In[13]:


#########################################################################
# Part 1 - Implement Volume class and make_volume()

class Volume:
    """
    Container for CT volume data
    """
    
    def __init__(self, grid_values, grid_extents=(1, 1, 1)):
        """
        Input: 3D numpy array grid_values, optional tuple grid_extents
        that contains extent of grid in each dimension
        
        Ouput: Object storing grid of scalar data
        """
        self._extents = tuple([grid_extents[0], grid_extents[1], grid_extents[2]])
        self._data = grid_values.copy()
    
    # Implement during Part 3
    def plot_volume_contour(self, val, title="Contour plot of volume", plot_edges=True):
        """
        Input: Volume object self, number val, optional string title
        
        Output: plotly figure corresponding to contour plot of volume using Marching Cubes
        with specified value val.  Use grid extents to set aspect ratio.
        Also writes HTML to PLOTS_PATH + title + ".html".
        """
        verts, faces = skimage.measure.marching_cubes_classic(self._data, val)
        x_ax, y_ax, z_ax = zip(*verts)  
        fig = ff.create_trisurf(x=z_ax, y=y_ax, z=x_ax,
                                plot_edges=plot_edges,
                                simplices=faces, title=title)
        fig.update_scenes(aspectratio=dict(x=self._extents[2],
                                           y=self._extents[1],
                                           z=self._extents[0]),
                          aspectmode="manual")
        fig.write_html(PLOTS_PATH + title + ".html")

        return fig
        
    # Implement during Part 4
    def plot_volume_slice(self, title):
        """
        Input: Volume object self, optional string title
    
        Output: plotly figure corresponding to 3D slices of volume 
        perpendicular to z-axis. The vertical position of this slice 
        should be controlled by buttons and a slider
        Also writes HTML to PLOTS_PATH + title + ".html".
        """   
        volume = self._data
        nb_frames = len(volume)

        fig = go.Figure(frames=[go.Frame(data=go.Surface(
            z=k * np.ones((len(volume[0]), len(volume[0][0]))),
            surfacecolor=np.flipud(volume[k]), cmin=volume.min(),
            cmax=volume.max()), name=str(k)) for k in range(nb_frames)])
        # Add data to be displayed before animation starts

        fig.add_trace(go.Surface(z=np.zeros((len(volume[0]), len(volume[0][0]))),
                                 surfacecolor=np.flipud(volume[0]),
                                 colorscale=PL_BONE, cmin=volume.min(),
                                 cmax=volume.max(), colorbar=dict(thickness=20, ticklen=4)))
        
        def frame_args(duration):
            return {"frame": {"duration": duration}, "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"}}


        sliders = [{"pad": {"b": 10, "t": 60},
                    "len": 0.9, "x": 0.1, "y": 0,
                    "steps": [{"args": [[f.name], frame_args(0)],
                               "label": str(k), "method": "animate"}
                              for k, f in enumerate(fig.frames)]}]

        

        # Layout

        fig.update_layout(title=title,
                          width=600, height=600,
                          scene=dict(zaxis=dict(range=[-0.1, nb_frames],
                                                autorange=False),
                                     aspectratio=dict(x=self._extents[2],
                                                      y=self._extents[1],
                                                      z=self._extents[0])),
                          updatemenus=[{"buttons": [{"args": [None, frame_args(50)],
                                                     "label": "&#9654;",
                                                     "method": "animate"},
                                                    {"args": [[None], frame_args(0)],
                                                     "label": "&#9724;",
                                                     "method": "animate"}],
                                        "direction": "left",
                                        "pad": {"r": 10, "t": 70},
                                        "type": "buttons",
                                        "x": 0.1,
                                        "y": 0}],
                          sliders=sliders)
        fig.write_html(PLOTS_PATH + title + ".html")
        return fig


# In[15]:


def make_volume(z_coords, y_coords, x_coords, grid_fun):
    """
    Input: Numpy arrays z_coords, y_coords, x_coords,
    function grid_fun that takes 3 scalar parameters

    Output: Volume object whose grid values as grid_fun(z, y, x)
    taken at points of the grid defined by the coordinate arrays
    """
    volume= []
    for dummy1 in range(len(z_coords)):
        volume_z = []
        for dummy2 in range(len(y_coords)):
            volume_y = []
            for dummy3 in range(len(x_coords)):
                volume_y.append(grid_fun(z_coords[dummy1], y_coords[dummy2], x_coords[dummy3]))
            volume_z.append(volume_y)
        volume.append(volume_z)
    z_ext = z_coords.max()-z_coords.min()
    y_ext = y_coords.max()-y_coords.min()
    x_ext = x_coords.max()-x_coords.min()
    return Volume(np.array(volume), grid_extents=tuple([z_ext, y_ext, x_ext]))


# In[17]:


############################################################################
# Part 2 - Read binary CT data from a file and create a Volume object

def read_volume(vol_name):
    """
    Input: String vol_name 
    
    Output: Volume object read from binary-encoded CT volume data in given file
    
    NOTE: Use struct module - https://docs.python.org/3/library/struct.html
    """
    with open(vol_name, "rb") as vol:
        vol_binary = vol.read()
    
    #print(vol_binary)
    shape_fmt = "!3i"
    vol_shape = struct.unpack_from(shape_fmt, vol_binary)
    #print("shape", vol_shape)
    shape_offset = struct.calcsize(shape_fmt)
    boarder_fmt = "!i"
    boarder = struct.unpack_from(boarder_fmt, vol_binary, shape_offset)
    #print("boarder", boarder)
    boarder_offset = shape_offset + struct.calcsize(boarder_fmt)
    true_size_fmt = "!3f"
    true_size = struct.unpack_from(true_size_fmt, vol_binary, boarder_offset)
    #print("true size", true_size)
    vol_size = vol_shape[0] * vol_shape[1] * vol_shape[2]
    vol_offset = struct.calcsize(true_size_fmt) + boarder_offset
    vol_fmt = "!" + str(vol_size) + "B"
    vol_bytes = struct.unpack_from(vol_fmt, vol_binary, vol_offset)
    flat_vol = np.array(vol_bytes, dtype=np.uint8)
    volume = np.reshape(flat_vol, vol_shape)
    #print(volume)
    return Volume(volume, np.array(true_size))

