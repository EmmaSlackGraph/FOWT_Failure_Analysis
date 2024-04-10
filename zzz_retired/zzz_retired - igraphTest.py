import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from dash import Dash, dcc, html
import dash_cytoscape as cyto
import igraph as ig
from graphBuilder import *
from igraph import Graph

arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix (2)")

g = ig.Graph.Adjacency(arr)
g.vs["mode"] = [False for i in range(arr[:26].shape[0])] + [True for i in range(arr[26:].shape[0])]
color_dict = {True: "#fabc98", False:"#98c5ed"}

ig.config['plotting.backend'] = 'matplotlib'
ig.plot(g,
        layout="random",
        vertex_size=20,
        vertex_color=[color_dict[mode] for mode in g.vs["mode"]],
        vertex_label=[i for i in range(arr.shape[0])],
        edge_width=[1])

plt.show()

def graph_trees(G):
    g = ig.from_networkx(G)
    ig.config['plotting.backend'] = 'matplotlib'
    ig.plot(g,
        layout="random",
        vertex_size=20,
        edge_width=[1])
