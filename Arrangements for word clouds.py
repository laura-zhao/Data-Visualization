
# coding: utf-8

# In[1]:


"""
Template for week 6 project in Data Visualization

Create a word cloud from a list words with frequency counts
"""

import math
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dill as pickle


# In[2]:


# Resource paths
PLOTS_PATH = "plots/"
DATA_PATH = "data/"


# In[3]:


############################################################################
# Provided code 

PLOT_RANGE = [-1, 1]
FIG_SIZE = 8

with open(DATA_PATH + "project6_boxes.pickle", 'rb') as file:
    BOXES = pickle.load(file)


# In[4]:


def init_plot(title):
    """
    Input: string title
    
    Ouput/action: Initialize plot in matplotlib and
    return figure/axis tuple
    """
    
    fig = plt.figure()
    fig.set_figheight(FIG_SIZE)
    fig.set_figwidth(FIG_SIZE)
    
    axs = plt.subplot()
    axs.set_xlim(PLOT_RANGE)
    axs.set_ylim(PLOT_RANGE)
    
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_title(title)
    
    return fig, axs


# In[5]:


def random_box_pos(box_size):
    """
    Input: Tuple box_size of floats
    
    Output: Tuple of floats corresponding to random lowerleft position
    of rectangle that lies in specified PLOT_RANGE
    """
    box_pos = (random.uniform(PLOT_RANGE[0], PLOT_RANGE[1] - box_size[0]),
               random.uniform(PLOT_RANGE[0], PLOT_RANGE[1] - box_size[1]))
    return box_pos


# In[6]:


############################################################################
# Part 1 - Compute random arrangements and plot the word cloud


def random_arrangement(word_boxes, seed=None):
    """
    Input: OrderedDict word_boxes with items (word, (font_size, box_size, word_off)),
    optional integer seed
    
    Output: OrderedDict with items (word, (font_size, box_pos, box_size, word_off))
    where box_pos is computed using random_box_pos()
    """
    word_boxes_pos = collections.OrderedDict()
    if seed:
        random.seed(seed)
    for word in word_boxes:
        box_pos = random_box_pos(word_boxes[word][1])
        tup = word_boxes[word]
        word_boxes_pos[word] = (tup[0], box_pos, tup[1], tup[2])
    return word_boxes_pos


# In[7]:


def plot_wordcloud(word_arrangement, title="Word cloud"):
    """
    Input: Dictionary word_arrangements with items (word, (font_size, box_pos, box_size, word_off)),
    optional string title
    
    Output: matplotlib figure containing plot of words using the specified font size and positions
    
    NOTE: The word's position is box_pos + word_off
    """
    
    fig, axs = init_plot(title)  
    for word in word_arrangement: 
        tup = word_arrangement[word]
        word_pos = (tup[1][0] + tup[3][0], tup[1][1] + tup[3][1])
        axs.text(word_pos[0], word_pos[1], word, fontsize=tup[0], 
                 bbox=dict(facecolor='yellow', edgecolor='red', pad=0.0))
    return fig


# In[9]:


#######################################################################
# Part 2 - Compute Monte Carlo arrangement


def intersect_intervals(interval1, interval2):
    """
    Input: Two pairs of numbers interval1 and interval2 as tuples
    
    Output: Boolean that indicates whether intervals overlap
    
    NOTE: Sharing a common endpoint is treated as NOT overlapping
    """
    if interval1[1] <= interval2[0] or interval1[0] >= interval2[1]:
        return False
    else:
        return True
    


# In[10]:


def intersect_boxes(box1, box2):
    """
    Input: Boxes box1, box2 of the form (box_pos, box_size)
    
    Output: Boolean indicating whether box1 and box2 intersect
    """
    box1_x_interval = (box1[0][0], box1[0][0] + box1[1][0])
    box1_y_interval = (box1[0][1], box1[0][1] + box1[1][1])
    box2_x_interval = (box2[0][0], box2[0][0] + box2[1][0])
    box2_y_interval = (box2[0][1], box2[0][1] + box2[1][1])
    x_tell = intersect_intervals(box1_x_interval, box2_x_interval)
    y_tell = intersect_intervals(box1_y_interval, box2_y_interval) 
    return x_tell and y_tell


# In[11]:


def intersect_box_arrangement(test_box, word_arrangement):
    """
    Input: Box test_box specified as tuple of position and size,
    OrderedDict word_arrangement with items (word, (font_size, box_pos, box_size, word_off))
    
    Output: Boolean indicating whether test_box intersects
    ANY of the boxes in word_arrangement
    """
    for word in word_arrangement:
        arr = (word_arrangement[word][1], word_arrangement[word][2])
        if intersect_boxes(test_box, arr):
            return True
    return False


# In[18]:


def montecarlo_arrangement(word_boxes, max_tries=100, seed=None):
    """
    Input: OrderedDict word_boxes with items (word, (font_size, box_size, word_off)) 
    integer max_tries, optional integer seed
    
    Output: OrderedDict with items (word, (font_size, box_pos, box_size, word_off))
    
    NOTE: Returned dictionary is computed one word at time
    using at most max_tries calls to random_box_pos() for each box.
    If no non-intersecting position is found vs. current arrangement,
    box is added to current arrangement using last position
    """
    word_boxes_pos = collections.OrderedDict()
    if seed:
        random.seed(seed)
    for word in word_boxes:
        for dummy in range(max_tries):
            box_pos = random_box_pos(word_boxes[word][1])
            if not(intersect_box_arrangement((box_pos, word_boxes[word][1]), word_boxes_pos)):
                break;
        tup = word_boxes[word]
        word_boxes_pos[word] = (tup[0], box_pos, tup[1], tup[2])
    return word_boxes_pos


# In[20]:


###################################################################
# Provided code for part 3 - Compute spiral arrangement

TURNS = 8
STEPS = np.linspace(-2 * math.pi * TURNS, 0, 129)


# In[21]:


def spiral_pos(theta, yoff):
    """
    Input: Floats theta, yoff
    
    Output: Parametric coordinates of point on logarthmic spiral
    translated vertically by yoff
    """
    
    rad = math.exp(theta / (2 * TURNS))
    return [rad * math.cos(theta), rad * math.sin(theta) + yoff]


# In[22]:


def initial_yoff():
    """
    Output: Float corresponding to random vertical offset of initial box 
    in spiral arrangement
    """

    yoff = 0.25 * random.uniform(PLOT_RANGE[0], PLOT_RANGE[1])
    return yoff


# In[23]:


def plot_spiral():
    """
    Action: plot logarithmic spiral use to create a spiral arrangement
    """
    
    title = "Parametric plot of spiral"
    fig, axs = init_plot(title)
                        
    points = [spiral_pos(theta, 0) for theta in STEPS]
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    axs.plot(x_coords, y_coords)
    return fig
    
#plot_spiral()


# In[24]:


############################################################
# Student code for part 3 

def spiral_arrangement(word_boxes, seed=None):
    """
    Input: OrderedDict with items (word, (font_size, box_size, word_off)) 
    optional integer seed
    
    Output: OrderedDict with items (word, (font_size, box_pos, box_size, word_off))
    
    NOTE: Returned dictionary is computed one word at time (in order)
    using non-intersecting boxes whose centers are positioned along the spiral.
    If no non-intersecting position is found vs. current arrangement,
    box is added to current arrangement using last position on the spiral
    """
    word_boxes_pos = collections.OrderedDict()
    if seed:
        random.seed(seed)
    for word in word_boxes:
        yoff = initial_yoff()
        for theta in STEPS:
            box_center_pos = spiral_pos(theta, yoff)
            box_pos_lx = box_center_pos[0] - 1/2 * word_boxes[word][1][0]
            box_pos_ly = box_center_pos[1] - 1/2 * word_boxes[word][1][1]
            box_pos = (box_pos_lx , box_pos_ly)
            if not(intersect_box_arrangement((box_pos, word_boxes[word][1]), word_boxes_pos)):
                break;
        tup = word_boxes[word]
        word_boxes_pos[word] = (tup[0], box_pos, tup[1], tup[2])
    return word_boxes_pos

