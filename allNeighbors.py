import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graphBuilder import *
from graphDiagnostics import *
from graphml2nwx import *
from smallWorldMeasures import *
from physicsPositions import *

''' allNeighbors----------------------------------------------------------------------------------------------------

            ****************************** Last Updated: 8 March 2024 ******************************

 Methods:
 1) diagonal_nodes: inputs adjacency matrix --> outputs nodes diagonal matrix

 2) short_path_child: inputs adjacency matrix, list of node names, target node, start node --> outputs new graph, new
   adjacency matrix, names of nodes in new graph, node path, edge path, effect nodes, mode nodes

 3) short_path_parent: inputs adjacency matrix, list of node names, target node, start node --> outputs new graph, new
   adjacency matrix, names of nodes in new graph, node path, edge path, effect nodes, mode nodes

 4) short_paths_neighbors: inputs an adjacency matrix, list of node names, a target node, and a start node --> outputs graph

 4) draw_short_child: inputs adjacency matrix, list of node names, target node, start node --> plots the graph (no return)

 5) draw_short_parent: inputs adjacency matrix, list of node names, target node, start node --> plots the graph (no return)

 6) draw_short_neighbors: inputs adjacency matrix, list of node names, target node, start node --> plots the graph (no return)
 
--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------


    diagonal_nodes Documentation
 --------------------------------
 This method takes in an adjacency matrix and outputs a diagonal matrix. For an adjacency matrix of length k, the
 outputted diagonal matrix places values 1-k on the diagonal.'''

def diagonal_nodes(arr):
    # Return the diagonal matrix with i+1 in the i,i spot and zero elsewhere
    return np.diag([i+1 for i in range(arr.shape[0])])



''' short_path Documentation
 ----------------------------
 This method inputs an adjacency matrix, list of node names, a target node, and a start node. We use an altered
 breadth first searh to find the shortest path between two nodes. We then generate a graph which includes all of
 the nodes along this path, as well as all the neighbors (parents and children) of nodes in the path. This method
 plots this new graph, and returns the graph, its adjacency matrix, and the names of the nodes in the graph.'''

def short_path_child(arr, nodeNames, target, start):
    # Initialize a directed graph
    G = nx.DiGraph()

    # We binarize the adjacency matrix so that relationships either exist or not (rather than having an intensity)
    adj = make_binary(arr).astype(int)

    # Create an array of booleans such that the ith entry indicates if the node has already been visited. Nodes
    # that have been visited are set to TRUE.
    nodeList = np.reshape(np.repeat(False, arr.shape[0]), (arr.shape[0], 1))

    # Create a diagonal matrix such that the value of the i,i entry is 1+1, referencing the node with name i+1
    nodes = diagonal_nodes(adj)

    # Visit the starting node and add it to the queue
    nodeList[start-1] = True
    queue = [start]

    # Create a dictionary to hold information about the children of each node. The keys will children and the values
    # will be who their parent is.
    kiddos = {}

# Continue while there are still nodes in the queue (reference algorithms for a breadth first search for more info)
    while len(queue) > 0:
        # Set the node we are looking at equal to the node next in line in the queue
        current = queue[0]
        
        # Determine the children of the current node
        children_bool = adj[current-1] @ nodes # vector of zeros and child names (numerical names)
        children = children_bool[np.nonzero(children_bool)] #list of just the child names (numerical names)

        for child in children: # For every child of the current node that was found above...
            if nodeList[child - 1] == False: # Check if the child has been visited. If not, continue with the following:
                queue.append(child) # Append the child to the queue
                nodeList[child - 1] = True # Change the status of the child to say we have visited it
                kiddos.update({child: current}) # Add a key/value pair of the child (as key) and the current node (as value)
            
            # If the child is the target, then we have found the shortest path (any other paths will be the same length
            # or longer). So break the while loop and continue on to below.
            if child == target:
                break
            
        # Remove the current node from the queue
        queue = queue[1:]
    
    # Start a trace of the path with the target node. We will iterate backwards through the path
    trace = target

    # Add the target node to the graph
    G.add_node(nodeNames[trace - 1])

    # Initialize a list of edges in the path, initialize a list of nodes (numerical) in the path and add the target node, 
    # initialize a list of nodes (names - i.e. string type) and add the target node, and initialize effect and mode arrays
    edge_path = []
    node_path = [target]
    sig_nodes = [nodeNames[trace - 1]]
    effects = []
    modes = []

    # Iterate backwards through the path
    while trace != start:
        parent = kiddos[trace] # Find the parent of the current node
        edge_path.append((nodeNames[parent - 1], nodeNames[trace- 1])) # Append the edge between the current node and its parent
        node_path.append(parent) # Append the parent to the list of nodes in the path
        trace = parent # Update the current node to the parent node

    # Traverse through the path moving forward (recall that the path list is backwards, hence the reverse command)
    for node in reversed(node_path):

        # Determine if current node is effect or mode, add to correct array, and add to the graph
        if node < 27:
            effects.append(nodeNames[node - 1])
        else:
            modes.append(nodeNames[node - 1])
        G.add_node(nodeNames[node - 1])

        # Add the current node to the list of nodes in our graph
        sig_nodes.append(nodeNames[node - 1])

        # Find the children of the current node
        children_bool = adj[node-1] @ nodes
        children = children_bool[np.nonzero(children_bool)] #list of child names (numerical names)
        
        # Iterate through the list of parents and child nodes
        for child in children:
            # Determine if the child is a failure effect or failure mode and add to correct list
            if child < 27:
                effects.append(nodeNames[child - 1])
            else:
                modes.append(nodeNames[child - 1])
            G.add_node(nodeNames[child - 1]) # Add the child to the graph
            sig_nodes.append(nodeNames[child - 1]) # Add the child to the list of nodes in the graph
            G.add_edge(nodeNames[node - 1], nodeNames[child - 1]) # Add the edge between the current node and child

    # Obtain the adjacency matrix of the graph we constructed
    new_adj = nx.to_numpy_array(G)

    # Returns the graph, its adjacency matrix, list of nodes in graph, paths (node and edges), and effects and modes arrays
    return G, new_adj, sig_nodes, node_path, edge_path, effects, modes



''' short_path_parent Documentation
 -----------------------------------
 This method inputs an adjacency matrix, list of node names, a target node, and a start node. We use an altered
 breadth first searh to find the shortest path between two nodes. We then generate a graph which includes all of
 the nodes along this path, as well as all the neighbors (parents and children) of nodes in the path. This method
 plots this new graph, and returns the graph, its adjacency matrix, and the names of the nodes in the graph.'''

def short_path_parent(arr, nodeNames, target, start):
    # Initialize the graph, obtain the adjacency matrix and nodes matrix, set the trace to the start for forward
    # walking on the path, and add the start node to the graph
    G = nx.DiGraph()
    adj = make_binary(arr).astype(int)
    nodes = diagonal_nodes(adj)
    trace = start
    G.add_node(nodeNames[trace - 1])

    # Obtain the path (node information and edge information) from the short_path_child method. This allows us to
    # compare between parent and child paths, even if there is more than one shortest path
    K, kid_adj, kid_nodes, node_path, edge_path, effs, mods = short_path_child(arr, nodeNames, target, start)

    # Initialize the array of nodes in our graph, the number of nodes, and the effects nodes and modes nodes arrays
    sig_nodes = [nodeNames[trace - 1]]
    effects = []
    modes = []
    
    # Traverse through the path moving forward
    for node in reversed(node_path):

        # Determine if a node is a failure effect or failure mode and add to the appropriate array
        if node < 27:
            effects.append(nodeNames[node - 1])
        else:
            modes.append(nodeNames[node - 1])

        # Add the node to the graph and the list of nodes in teh graph
        G.add_node(nodeNames[node - 1])
        sig_nodes.append(nodeNames[node - 1])
        
        # Find all the parents of the current node in the path
        parent_bool = nodes @ adj[:, node-1]
        parents = parent_bool[np.nonzero(parent_bool)]

        # For each parent of the current node, determine if it is a failure mode or failure effect
        for parent in parents:
            if parent < 27:
                effects.append(nodeNames[parent - 1])
            else:
                modes.append(nodeNames[parent - 1])
            G.add_node(nodeNames[parent - 1]) # Add the parent to the graph
            sig_nodes.append(nodeNames[parent - 1]) # Add the parent to the list of nodes in the graph
            G.add_edge(nodeNames[parent - 1], nodeNames[node - 1]) # Add the edge between the current node and parent

    # Obtain the adjacency matrix of the graph we constructed
    new_adj = nx.to_numpy_array(G)

    # Returns the graph, its adjacency matrix, list of nodes in graph, paths (node and edges), and effects and modes arrays
    return G, new_adj, sig_nodes, node_path, edge_path, effects, modes
    
    
    
''' short_paths_neighbors Documentation
 --------------------------------------
 This method inputs an adjacency matrix, list of node names, a target node, and a start node. We use an altered
 breadth first searh to find the shortest path between two nodes. We then generate a graph which includes all of
 the nodes along this path and return this graph.'''

def short_paths_neighbors(arr, nodeNames, target, start):
    # Obtain path information from both the parent and child methods
    H, new_adj2, sig_nodes2, npath2, epath2, effs, mods = short_path_parent(arr, nodeNames, target, start)
    G, new_adj, sig_nodes, npath, epath, effs2, mods2 = short_path_child(arr, nodeNames, target, start)

    # Combine the effects, modes, and significant nodes from the above graphs
    effects = effs + effs2
    modes = mods + mods2
    significant_nodes = sig_nodes + sig_nodes2

    # Take the union of the two graphs so that it has both the parents and children
    R = nx.compose(G, H)
    return R, nx.to_numpy_array(R), significant_nodes, npath, epath, effects, modes



''' draw_short_child Documentation
 ----------------------------
 This method inputs an adjacency matrix, list of node names, a target node, and a start node. We use an altered
 breadth first searh to find the shortest path between two nodes. We then generate a graph which includes all of
 the nodes along this path, as well as all the children of nodes in the path. This method plots the graph, but
 nothing is returned.'''

def draw_short_child(arr, nodeNames, target, start):
    # Obtain path information from method
    G, new_adj, sig_nodes, npath, epath, effects, modes = short_path_child(arr, nodeNames, target, start)

    # Set position of all the nodes not on the path
    pos = nx.kamada_kawai_layout(G.subgraph(set(G) - set(nodeNames[npath[::-1] - np.repeat(1, len(npath))])))

    # Set the position of the nodes on the path to be in a straight line
    for n in range(len(npath)):
        pos[nodeNames[npath[-(n+1)] - 1]] = [(-0.5 + n/(len(npath)-1)), 0]
    
    # Plot the nodes
    nx.draw_networkx_nodes(G, pos, nodelist=effects, node_color="#98c5ed", node_size=2700, edgecolors="#799dbd", node_shape="s")
    nx.draw_networkx_nodes(G, pos, nodelist=modes, node_color="#fabc98", node_size=2700, edgecolors="#c89679", node_shape="s")

    # Plot the source and target nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[nodeNames[start-1], nodeNames[target-1]], node_size = 2500, node_color="#fcf8d9", node_shape="s")

    # Plot labels and edges
    nx.draw_networkx_labels(G, pos, font_size=10, verticalalignment='center_baseline')
    nx.draw_networkx_edges(G, pos, arrowsize=60)

    # Plot path edges and show plot
    nx.draw_networkx_edges(G, pos, arrowsize=60, edgelist=epath, edge_color="red", width=4)
    plt.box(False)
    filename = "Figures/allFriends/shortChild" + str(start) + "." + str(target) + ".png"
    plt.savefig(filename)
    plt.show()



''' draw_short_parent Documentation
 ----------------------------
 This method inputs an adjacency matrix, list of node names, a target node, and a start node. We use an altered
 breadth first searh to find the shortest path between two nodes. We then generate a graph which includes all of
 the nodes along this path, as well as all the parents of nodes in the path. This method plots the graph, but
 nothing is returned.'''

def draw_short_parent(arr, nodeNames, target, start):
    # Obtain path information from method
    G, new_adj, sig_nodes, npath, epath, effects, modes = short_path_parent(arr, nodeNames, target, start)

    # Set position of all the nodes not on the path
    pos = nx.kamada_kawai_layout(G.subgraph(set(G) - set(nodeNames[npath[::-1] - np.repeat(1, len(npath))])))

    # Set the position of the nodes on the path to be in a straight line
    for n in range(len(npath)):
        pos[nodeNames[npath[-(n+1)] - 1]] = [(-0.5 + n/(len(npath)-1)), 0]
    
    # Plot the nodes
    nx.draw_networkx_nodes(G, pos, nodelist=effects, node_color="#98c5ed", node_size=2700, edgecolors="#799dbd", node_shape="s")
    nx.draw_networkx_nodes(G, pos, nodelist=modes, node_color="#fabc98", node_size=2700, edgecolors="#c89679", node_shape="s")

    # Plot the source and target nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[nodeNames[start-1], nodeNames[target-1]], node_size = 2500, node_color="#fcf8d9", node_shape="s")

    # Plot labels and edges
    nx.draw_networkx_labels(G, pos, font_size=10, verticalalignment='center_baseline')
    nx.draw_networkx_edges(G, pos, arrowsize=60)

    # Plot path edges and show plot
    nx.draw_networkx_edges(G, pos, arrowsize=60, edgelist=epath, edge_color="red", width=4)
    plt.box(False)
    filename = "Figures/allFriends/shortParent" + str(start) + "." + str(target) + ".png"
    plt.savefig(filename)
    plt.show()
    
    
    
''' draw_short_neighbors Documentation
 --------------------------------------
 This method inputs an adjacency matrix, list of node names, a target node, and a start node. We use an altered
 breadth first searh to find the shortest path between two nodes. We then generate a graph which includes all of
 the nodes along this path, as well as all the neighbors (parents and children) of nodes in the path. This method
 plots the graph, but nothing is returned.'''

def draw_short_neighbors(arr, nodeNames, target, start):
    # Obtain path information from both the parent and child methods
    H, new_adj2, sig_nodes2, npath2, epath2, effs, mods = short_path_parent(arr, nodeNames, target, start)
    G, new_adj, sig_nodes, npath, epath, effs2, mods2 = short_path_child(arr, nodeNames, target, start)

    # Take the union of the two graphs so that it has both the parents and children
    R = nx.compose(G, H)

    # Comine the effects and modes lists
    effects = effs + effs2
    modes = mods + mods2
    # Debug statement: print(mods, mods2)

    # Position all the nodes not in the path
    pos = nx.spring_layout(R.subgraph(set(R) - set(nodeNames[npath2 - np.repeat(1, len(npath))])))

    # Position the nodes on the path. If the paths are not the same, plot both paths (one above the other)
    if npath != npath2:
        for n in range(len(npath)):
            if n == 0:
                pos[nodeNames[npath2[n] - 1]] = [-0.5, 0]
            elif n == len(npath) - 1:
                pos[nodeNames[npath2[n] - 1]] = [0.5, 0]
            else:
                pos[nodeNames[npath2[n] - 1]] = [(-0.5 + n/(len(npath)-1)), 0.1]
                pos[nodeNames[npath[-(n+1)] - 1]] = [(-0.5 + n/(len(npath)-1)), -0.1]
    else:
        for n in range(len(npath)):
            pos[nodeNames[npath2[-(n+1)] - 1]] = [(-0.5 + n/(len(npath)-1)), 0]

    # Plot the nodes
    nx.draw_networkx_nodes(R, pos, nodelist=effects, node_color="#98c5ed", node_size=2700, edgecolors="#799dbd", node_shape="s")
    nx.draw_networkx_nodes(R, pos, nodelist=modes, node_color="#fabc98", node_size=2700, edgecolors="#c89679", node_shape="s")

    # Plot the source and target nodes
    nx.draw_networkx_nodes(R, pos, nodelist=[nodeNames[start-1], nodeNames[target-1]], node_size = 2500, node_color="#fcf8d9", node_shape="s")

    # Plot labels and edges
    nx.draw_networkx_labels(R, pos, font_size=10, verticalalignment='center_baseline')
    nx.draw_networkx_edges(R, pos, arrowsize=60)

    # Plot path edges and show plot
    nx.draw_networkx_edges(R, pos, arrowsize=60, edgelist=epath, edge_color="red", width=4)
    plt.box(False)
    filename = "Figures/allFriends/shortNeighbor" + str(start) + "." + str(target) + ".png"
    plt.savefig(filename)
    plt.show()