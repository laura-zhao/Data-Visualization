
# coding: utf-8

# In[4]:


"""
Template for week 5 project in Data Visualization

Create raster images using three types of dynamical systems
"""

import random
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


# Resource paths
PLOTS_PATH = "plots/"
DATA_PATH = "data/"


# In[6]:


#######################################################################################
# Part 1 - Compute and plot Julia sets


def invert_fun(complex_fun, complex_val):
    """
    Input: np.poly1d object complex_fun and complex complex_val
    
    Output: List of n possible complex solutions to the equation complex_val = complex_fun(inv)
    The roots of complex_poly are the n possible values for complex_fun^(-1)(complex_val)
    """
    if(isinstance(complex_val, complex)):
        new_c = np.array(complex_fun.c, dtype='complex')
    else:
        new_c = complex_fun.c
    new_c[-1] -= complex_val
    root = np.poly1d(new_c).r
    return np.sort(root)
#print(invert_fun(np.poly1d([-3,  3,  0]),0.10367460967663507+0.08410597490646866j))


# In[8]:


def julia_set(lmbd, z_0, num_returned, num_dropped, seed=None):
    """
    Input: complex lmbd, z_0, integers num_returned, num_dropped,
    optional integer seed
    
    Output: List of complex numbers (starting with z_0)
    generated by Julia iteration with given lmbd
    """
    
    if seed:
        random.seed(seed)
    z_k = z_0
    zk_list = [z_0]
    func = np.poly1d([-lmbd, lmbd, 0])
    for dummy in range(num_returned+num_dropped-1):
        roots = invert_fun(func, z_k)
        root = random.choice(roots)
        zk_list.append(root)
        z_k = root
    # Remember to use random.choice() when selecting an inverse
    
    return zk_list[num_dropped:]


# In[10]:


def plot_julia(julia_points, lmbd):
    """
    Input: List of complex numbers that lie on the Julia set for given lmbd
    
    Output: Scatter plot of complex number in list julia_points
    """
    x_real = []
    y_imag = []
    for point in julia_points:
        x_real.append(point.real)
        y_imag.append(point.imag)
    
    fig, axs = plt.subplots()
    axs.scatter(x_real, y_imag)
    axs.set_xlabel("real component of complex")
    axs.set_ylabel("imaginary component of complex")
    axs.set_title("Julia set when "+chr(955)+"="+str(lmbd))
    return fig


# In[12]:


##############################################################################
# Part 2 - Compute and plot Mandelbrot sets

MAX_ITERS = 20


def iterate_mandel(z_0):
    """
    Input: complex z_0
    
    Output: number of iterations for Mandelbrot function f(z) = z ** 2 + z_0 to diverge,
    maximum number of iterations is capped at MAX_ITERS
    """
    z_k = z_0
    max_it_num = 0
    for dummy in range(MAX_ITERS):
        if(abs(z_k) > 2): 
            break
        z_k = z_k ** 2 + z_0
        max_it_num += 1
    return max_it_num


# In[14]:


def mandel_table(real_values, imag_values):
    """
    Input: arrays real_values, imag_values of floats
    
    Output: 2D numpy array indicating number of iterations for
    divergence of Mandelbrot iteration initialized at
    corresponding complex values
    """
    table = []
    for imag in imag_values:
        line = []
        for real in real_values:
            num = iterate_mandel(complex(real, imag))
            line.append(num)
        table.append(line)
    return np.array(table)


# In[16]:


def plot_mandelbrot(real_values, imag_values):
    """
    Input: lists (or numpy arrays) real_values and imag_values
    
    Output: matplotlib figure of image plot of mandel_table()
    applied to real_values, imag_values
    """
    madel = mandel_table(real_values, imag_values)
    fig, axs = plt.subplots()
    axs.imshow(madel, origin="lower")
    axs.set_title("Mandelbrot Set")
    axs.set_xlabel("real")
    axs.set_ylabel("imag")
    return fig


# In[18]:


#########################################################################################
# Part 3 - Compute and plot Newton basins 

EPSILON = 0.000001

def newton_index(roots, z_0):
    """
    Input: List of complex roots, complex initial value z_0
    
    Output: Index of root in roots to which Newton iteration with initial
    value z_0 converges (with tolerance EPSILON)
    
    NOTE: If the iteration encounters a zero derivative, return len(roots)
    """
    g_z = np.poly1d(roots, r=True)
    g_z_d = g_z.deriv(1)
    z_k = z_0
    g_z_k = g_z(z_k)
    g_d_z_k = g_z_d(z_k)
    z_k_f = z_0
    while abs(g_z_k) > EPSILON and abs(g_d_z_k) > EPSILON:
        z_k_f = z_k
        z_k = z_k - g_z_k/g_d_z_k
        g_z_k = g_z(z_k)
        g_d_z_k = g_z_d(z_k)
    if abs(g_d_z_k) <= EPSILON:
        return len(roots)
    if abs(g_z_k) <= EPSILON:
        min_val = abs(roots[0] - z_k_f)
        min_idx = 0
        for idx in range(len(roots)):
            if abs(roots[idx] - z_k_f) < min_val:
                min_idx = idx
                min_val = abs(roots[idx] - z_k_f) 
        return min_idx
                
    return 0


# In[22]:


def newton_table(roots, real_values, imag_values):
    """
    Input: List of complex numbers roots, two lists of floats that define a grid
    of complex numbers
    
    Output:  2D numpy array of Newton indices where z_0 takes on values in
    the grid of complex numbers
    """
    ans = []
    for imag in imag_values:
        row = []
        for real in real_values:
            z_0 = complex(real, imag)
            row.append(newton_index(roots, z_0))
        ans.append(row)
    return np.array(ans)


# In[ ]:


#########################################################################################
# Student code for plotting Newton basins (peer-graded)

def plot_newton(roots, real_values, imag_values):
    """
    Input: List of complex numbers roots, two lists of floats that define a grid
    of complex numbers
    
    Output: matplotlib figure of mage plot of mandel_table()
    applied to roots, real_values, imag_values
    """
    
    fig, axs = plt.subplots()
    return fig
