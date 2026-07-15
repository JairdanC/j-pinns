#Libraries
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
import numpy as np
import jax
from jaxtyping import Array, Int, Float

FIG_SIZE_LONG = (15, 4) 
FIG_SIZE_SHORT = (8, 4)

def plot_profile(solution: Array,
                 label: str,
                 domain: Array
                 ) -> Figure:
    
    NotImplemented

def plot_profile(solution: Array,
                 label: str,
                 fig: Figure
                 ) -> Figure:
    """
    Plots the specified profiles of a given solution, figure must exist,
    it can be passed such that the new profile is graphed on top, for compatiblity,
    the passed figure should be a profile
    """

    NotImplemented

def plot_profiles(domain: Array,
                  solutions: tuple[Array, ...],
                  labels: tuple[str]
                  ) -> Figure:

    """
    Plots the specified profiles of a given set of solutions
    """

    NotImplemented

def plot_losses():
    NotImplemented
 


