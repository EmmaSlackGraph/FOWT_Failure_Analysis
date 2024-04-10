import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from dash import Dash, dcc, html
import dash_cytoscape as cyto
from graphBuilder import *
import easygraph as eg
import torch

arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")

G = eg.DiGraph()
labels = get_node_array(arr, nodeNames)
edges = get_graph_edges(arr, 'tuple', 0.5)
G.add_nodes([i for i in range(arr.shape[0])], nodes_attr= labels)
G.add_edges(edges)

'''pos = eg.random_position(G)
eg.draw_easygraph_nodes(G, pos)
eg.draw_easygraph_edges(G, pos)
plt.show()'''

eg.draw_kamada_kawai(G)