import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from dash import Dash, dcc, html
import dash_cytoscape as cyto
from graphBuilder import *
from graphDiagnostics import *
from graphml2nwx import *
from smallWorldMeasures import *
from spectralAnalysis import *
from searchMethods import *
from physicsPositions import *
from fiedlerVoronoi import *
from plotClusters import *
from clusterAnalysis import *
from failureProbabilities import *
from sympy import *
import copy

'''arr, nodeNames = excel2Matrix("failureData.xlsx", "modes2effects")
names = []
for i in range(len(nodeNames)):
            names.append(nodeNames[i].replace("\n", " "))
G, arr = matrix2Graph(arr, names, True)
plot_graph(G, "bipartite", 26, names)
# draw_bfs_multipartite(arr, nodeNames, 16, "child")
# draw_bfs_multipartite(arr, nodeNames, 16, "parent")'''



'''arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
G, arr = matrix2Graph(arr, nodeNames, True)
draw_bfs_multipartite(arr, nodeNames, 20)'''

'''arr, nodeNames = excel2Matrix("Task49Graph.xlsx", "AlteredSheet-noCause")
G, arr = matrix2Graph(arr, nodeNames, True)

draw_bfs_multipartite(arr, nodeNames, [1,3,7], "multi-parent")'''


'''arr, nodeNames = excel2Matrix("Task49Graph.xlsx", "AlteredSheet-noCause")
G, arr = matrix2Graph(arr, nodeNames, False)
nodenames2 = nodeNames

gens = {}
nodeNames = [i for i in range(arr.shape[0])]

for i in range(len(nodeNames)):
    if i < 0:
        gens.update({nodeNames[i]: {"layer": 0}})
        continue
    elif i < 49:
        gens.update({nodeNames[i]: {"layer": 1}})
        continue
    gens.update({nodeNames[i]: {"layer": 2}})

nx.set_node_attributes(G, gens)

print(nx.is_planar(G))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, nodelist=nodeNames[:49], node_color="#fabc98", node_size=200, edgecolors="#c89679")
nx.draw_networkx_nodes(G, pos, nodelist=nodeNames[49:], node_color="#98c5ed", node_size=200, edgecolors="#799dbd")
for edge in G.edges:
    # print(edge[0])
    # print(np.where(np.array(names) == edge[0])[0][0])
    if edge[0] > 47:
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color="red")
        continue
    else:
        nx.draw_networkx_edges(G, pos, edgelist=[edge])
nx.draw_networkx_labels(G, pos, font_size=5, verticalalignment='center_baseline')
plt.box(False)
plt.show()
#nx.draw_networkx_nodes(G, pos, nodelist=nodeNames[:7], node_color="#F2F2F2", node_size=200, edgecolors="#c1c1c1")

print(nodenames2)
names = []
for node in nodenames2:
    names.append(node.replace("\n", " "))

K, arr = matrix2Graph(arr, names, True)
plot_graph(K, "bipartite", 48, names)

effects_mark = 48

pos = nx.bipartite_layout(K, names[:effects_mark])  # positions for all nodes
nx.draw_networkx_nodes(K, pos, nodelist=names[effects_mark:], node_color="#98c5ed", edgecolors="#799dbd")
nx.draw_networkx_nodes(K, pos, nodelist=names[:effects_mark], node_color="#fabc98", edgecolors="#c89679")

# This for loop places labels outside of the nodes for easier visualization
for i in pos:
    x, y = pos[i]
    if x <= 0:
        plt.text(x-0.075,y,s=i, horizontalalignment='right')

    else:
        plt.text(x+0.075,y,s=i, horizontalalignment='left')

#nx.draw_networkx_labels(G, pos, horizontalalignment='center')
for edge in K.edges:
    # print(edge[0])
    # print(np.where(np.array(names) == edge[0])[0][0])
    if np.where(np.array(names) == edge[0])[0][0] > 47:
        nx.draw_networkx_edges(K, pos, edgelist=[edge], edge_color="red")
        continue
    else:
        nx.draw_networkx_edges(K, pos, edgelist=[edge])
plt.box(False)
plt.show()'''