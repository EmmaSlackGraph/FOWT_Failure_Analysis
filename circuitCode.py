import math
import networkx as nx
import matplotlib.pyplot as plt
from graphBuilder import *
from spectralAnalysis import *
from searchMethods import *
from allPaths import *

arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
arr = make_binary(arr)
G, arr = matrix2Graph(arr, nodeNames, True)

start = 1
target = 14
resistors = 10
voltage = 20

threshold = nx.dijkstra_path_length(G, nodeNames[start - 1], nodeNames[target - 1])+1
print("threshold", threshold)
title = False

G = nx.DiGraph()

edge_labels = {}
paths = dfs(arr, diagonal_nodes(arr), start, target, threshold = threshold)

if len(paths) == 1:
    current = voltage / ((threshold - 1) * resistors)
else: current = math.pi

# Initialize effects and modes arrays for different colors when we plot
effects = []
modes = []

# Walk along each path and look at every node. Add the node to the graph and the correct effects/modes array. Add an
# edge to the next node in the path only if the current node is not the target node (otherwise move on to next path).
for path in paths:
    for i in range(len(path)):
        G.add_node(nodeNames[path[i] - 1])
        if path[i] < 27: effects.append(nodeNames[path[i]-1])
        else: modes.append(nodeNames[path[i]-1])

        if path[i] == target: 
            continue
        G.add_weighted_edges_from([(nodeNames[path[i]-1], nodeNames[path[i+1]-1], current)])
        edge_labels.update({(nodeNames[path[i]-1], nodeNames[path[i+1]-1]): "%.2f Amp" % current})

# Position the nodes in a planar layout (i.e. no edges cross) if the graph can be plotted as such, otherwise use shell layout
if nx.is_planar(G):
    pos = nx.planar_layout(G.subgraph(set(G) - set(nodeNames[[start - 1, target-1]])))
else:
    pos = nx.shell_layout(G.subgraph(set(G) - set(nodeNames[[start - 1, target-1]])))

# Position start and end nodes
pos[nodeNames[start - 1]] = [-1.5, 0]
pos[nodeNames[target-1]] = [1.5, 0]

# Plot graph
nx.draw_networkx_nodes(G, pos, nodelist=effects, node_color="#98c5ed", node_size=2700, edgecolors="#799dbd", node_shape="s")
nx.draw_networkx_nodes(G, pos, nodelist=modes, node_color="#fabc98", node_size=2700, edgecolors="#c89679", node_shape="s")
nx.draw_networkx_nodes(G, pos, nodelist=[nodeNames[start-1], nodeNames[target-1]], node_size = 2500, node_color="#fcf8d9", node_shape="s")
nx.draw_networkx_labels(G, pos, font_size=10, verticalalignment='center_baseline')
nx.draw_networkx_edges(G, pos, arrowsize=60)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Figure title
plt_string = "Paths of length " + str(threshold - 1) + " from\n" + ' '.join(nodeNames[start - 1].splitlines()) + " to\n" + ' '.join(nodeNames[target - 1].splitlines())

# Show graph (and title if the user choses to)
plt.box(False)
if title:
    plt.title(plt_string)
filename = "Figures/allPaths/allPaths" + str(threshold) + "." + str(start) + "." + str(target) + ".png"
plt.savefig(filename)
plt.show()