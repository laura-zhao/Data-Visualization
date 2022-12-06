
# coding: utf-8

# In[3]:


"""
Template for week 4 project in Data Visualization

Solve the differential equations for Earth/Sun orbits using scipy
Plot the result orbits using matplotlib
"""

import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation


# In[4]:


# Define some constants - units are in kilograms, kilometers, and days
GRAV_CON = 4.981 * 10 ** (-10) 
MASS_SUN = 1.989 * 10 ** 30
MASS_EARTH = 5.977 * 10 ** 24
INIT_EARTH = (1.495979 * 10 ** 8, 0, 0, 2.56721625 * 10 ** 6)
ORBITS = {}

# Resource paths
DATA_PATH = "data/"
PLOTS_PATH = "plots/"


# In[5]:


#########################################################################
# Part 1 -  Solve the orbital equations and plot the resulting orbits


def orbital_eqs(e_0, t_0):
    """
    Input: Tuple e_0 of floats that represent Earth's position (p_x, p_y)
    and velocity (v_x, v_y) at time t_0
    
    Output: Tuple that contains approximation to d(e)/dt,
    computed from the orbital equations
    
    NOTE: The variable t_0 is not used in computing the
    returned value for the orbital equations
    """
    p_x, p_y, v_x, v_y = e_0
    r_t = math.sqrt(p_x**2 + p_y**2)
    par = -1 * GRAV_CON * MASS_SUN / r_t**3
    dpxdt = v_x
    dpydt = v_y
    dvxdt = par * p_x
    dvydt = par * p_y
    dedt = [dpxdt, dpydt, dvxdt, dvydt]
    
    return tuple(dedt)


# In[6]:


def solve_orbital_eqs(time_steps, speed=1.0):
    """
    Input: numpy array time_steps, optional float speed
    that scales the Earth's initial velocity
    
    Output: Returns a 2D numpy array whose rows are the
    x and y position of the Earth's orbit at specified times
    """
    init = (INIT_EARTH[0], INIT_EARTH[1], INIT_EARTH[2], INIT_EARTH[3]*speed)
    sol = odeint(orbital_eqs, init, time_steps)
    return sol[:,0:2].transpose()


# In[8]:


def plot_earth_coords(orbit, time_steps, title="(p_x(t),p_y(t)) plots"):
    """
    Input: Numpy array orbit whose rows are numpy arrays containing x and y positions
    for the Earth orbit for specified times in the numpy array time_steps
    
    Action: Plot both x positions and y positions versus time_steps
    using matplotlib
    """
    
    fig, axs = plt.subplots()
    axs.plot(time_steps,orbit[0,:], label = "x_coordinate", color='blue')
    axs.plot(time_steps,orbit[1,:], label ="y_coordinate", color='green')
    axs.set_title(title)
    axs.set_xlabel("time_steps/days")
    axs.set_ylabel("coordinate/km")
    axs.legend()

    return fig


# In[12]:


def plot_earth_orbit(orbit, title="Plotted orbit"):
    """
    Input: Numpy array orbit whose rows are numpy arrays containing x and y positions
    for the Earth orbit
    
    Action: Plot x positions versus y positions using matplotlib
    """
    fig, axs = plt.subplots()
    axs.plot(orbit[0,:],orbit[1,:])
    axs.set_title(title)
    axs.set_xlabel("x posistions/km")
    axs.set_ylabel("y positions/km")
    axs.set_aspect('equal')
    return fig


# In[ ]:


#########################################################################
# Part 2 - Animate the computed orbits


def extend_limits(limits, pad=0.1):
    """
    Input: Tuple limits = (min_x, max_x, min_y, max_y),
    float pad treated as a percentage

    Output: Tuple of four floats that represent new ranges
    extended by pad (as percentage of range length) in both directions
    """
    x_pad_min = limits[0] - pad* (limits[1]-limits[0])
    x_pad_max = limits[1] + pad* (limits[1]-limits[0])
    y_pad_min = limits[2] - pad* (limits[3]-limits[2])
    y_pad_max = limits[3] + pad* (limits[3]-limits[2])
    pad_range = (x_pad_min, x_pad_max, y_pad_min, y_pad_max)
    return pad_range


# In[ ]:


def animate_earth_orbit(orbit, title="Animated orbit"):
    """
    Input: Numpy array orbit whose rows are numpy arrays containing x and y positions
    for the Earth orbit
    
    Output: matplotlib Animation object corresponding to
    plot of x positions versus y positions using matplotlib
    """
    fig, axs = plt.subplots()
    x_ax = orbit[0,:]
    y_ax = orbit[1,:]
    x_min = min(x_ax)
    x_max = max(x_ax)
    y_min = min(y_ax)
    y_max = max(y_ax)
    
    org_range = (x_min, x_max, y_min, y_max)
    pad_range = extend_limits(org_range)
    line = axs.plot(x_ax, y_ax)
    dot, = axs.plot([], [], 'bo')
    dot2, = axs.plot(0, 0, 'yo')
    
    def init():
        axs.set_xlim(pad_range[0], pad_range[1])
        axs.set_ylim(pad_range[2], pad_range[3])
    return line

    def gen_dot(orbit):
        for i in orbit:
            newdot = orbit[i]
            yield newdot

    def update_dot(newd):
        dot.set_data(newd[0], newd[1])
        return dot,
    
    ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot(orbit), interval = 100, init_func=init)
    
    
    if file_name:
        anim.save(file_name, fps=30, extra_args=['-vcodec', 'libx264'])
        
    return anim

