import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graphBuilder import *
from graphDiagnostics import *
from graphml2nwx import *
from smallWorldMeasures import *
from physicsPositions import *
from allNeighbors import *

path = '/Users/eslack/Documents/Code/'
filelist = os.listdir(path)
path2 = []
for x in filelist:
    if x.startswith('Graph_Analysis'):
        path2.append(path + x + '/')
print(path2)

''' allPaths-------------------------------------------------------------------------------------------------------

            ****************************** Last Updated: 26 February 2024 ******************************

 Methods:
 1) dfs: inputs adjacency matrix, diagonal matrix of nodes, current node, target node, list of nodes in the current
 path, threshold, list of paths already found --> outputs list of paths found

2) draw_all_paths: inputs adjacency matrix, list of node names, start node, target node, threshold for path length,
 boolean for title --> plots the graph (no return)
 
--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------


    dfs Documentation
 ----------------------
 This method takes in an adjacency matrix, diagonal matrix of nodes, current node, target node, list of nodes
 in the current path, threshold for maximum path length, and a list of paths already found. In this case, DFS stands
 for 'Depth First Search'. This method uses the fundamental aspects of a depth first search to find all the paths
 between two nodes (that are less than or equal to a threshold length). To do this, we recursively call the dfs
 function and return the list of paths found.'''

def dfs(adj, nodes, current, target, path = [], threshold = 47, paths = []):
    # Add the current node to the list of nodes in the path
    path.append(current)

    # If the path is too large, then return nothing and try a different path
    if len(path) > threshold-1:
        return
    
    # Find all the children of the current node (refer to breadth_first_search for more explanataion)
    children_bool = adj[current-1] @ nodes
    children = children_bool[np.nonzero(children_bool)] #list of just child names (numerical names)
    #print(path)
    
    # Look at each child of the current node
    for child in children:

        # If the child is the target node, then we have finished our path. Print the path and append the 
        # completed path to the list of paths already found. Make sure to add the target node to the end
        # of both (but do not append to the current path)
        if child == target:
            print("target found ------------>", path + [child])
            paths.append(path + [child])

        # Print if the child is already in the current path
        #elif child in path:
            #print("child in path")

        # If the child is not already in the path and is not the target node, then call the dfs function
        # again, but have the child as the current node.
        elif child not in path and child != target:
            dfs(adj, nodes, child, target, path, threshold, paths)

            # Useful print statements for debugging - uncomment to use
            #print("current, child:", (current, child))
            #print("old path", path)

            # Once the path is complete, remove the child from the path, since we have already exhausted
            # the path that includes this child.
            path.remove(child)

            # Another debugging statement
            #print("adjusted path", path)

    # Return the list of all paths from source to target less than or equal to the threshold length
    return paths



''' draw_all_paths Documentation
 --------------------------------
 This method takes in an adjacency matrix, list of node names, start node, target node, threshold for path length,
 and boolean for printing the title of the graph. This method plots the graph from the dfs method and plots the
 graph. Nothing is returned.'''

def draw_all_paths(arr, nodeNames, start, target, threshold = 10, title = False):
    # Initialize graph, obtain binary adjacency matrix, and obtain paths from dfs
    G = nx.DiGraph()
    arr = make_binary(arr)
    paths = dfs(arr, diagonal_nodes(arr), start, target, threshold = threshold)

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
            G.add_edge(nodeNames[path[i]-1], nodeNames[path[i+1]-1])

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

    # Figure title
    plt_string = "Paths of length " + str(threshold - 1) + " from\n" + ' '.join(nodeNames[start - 1].splitlines()) + " to\n" + ' '.join(nodeNames[target - 1].splitlines())

    # Show graph (and title if the user choses to)
    plt.box(False)
    if title:
        plt.title(plt_string)
    filename = "Figures/allPaths/allPaths" + str(threshold) + "." + str(start) + "." + str(target) + ".png"
    plt.savefig(filename)
    plt.show()
    return G, paths