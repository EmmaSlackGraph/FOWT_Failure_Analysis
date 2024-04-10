import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Graph_Analysis.graphBuilder import *
from Graph_Analysis.graphDiagnostics import *
from Graph_Analysis.graphml2nwx import *
from Graph_Analysis.smallWorldMeasures import *
from Graph_Plotting.physicsPositions import *

def multipartiteClusterPlot(arr, nodeNames, clusters):
    values = {}

    for i in range(len(np.unique(clusters))):
        nodes = nodeNames[np.where(clusters == i)]
        for node in nodes:
            values.update({node: {"layer": i}})

    G, arr = matrix2Graph(arr, nodeNames, True)
    nx.set_node_attributes(G, values)

    pos = nx.multipartite_layout(G, subset_key='layer')
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.box(False)
    plt.show()
    return G