
from graphBuilder import *
from Graph_Analysis.smallWorldMeasures import *
from graphDiagnostics import *
from searchMethods import *

arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix", False)

G, arr = matrix2Graph(arr, nodeNames, False)

plot_graph(G, "multipartite", nodeNames, True)

''' short_path Documentation
 ----------------------------
 This method inputs an adjacency matrix, list of node names, a target node, and a start node. We use an altered
 breadth first searh to find the shortest path between two nodes. We then generate a graph which includes all of
 the nodes along this path, as well as all the neighbors (parents and children) of nodes in the path. This method
 plots this new graph, and returns the graph, its adjacency matrix, and the names of the nodes in the graph.'''

def short_path_parent(arr, nodeNames, target, start):
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
    nodeList[target-1] = True
    queue = [target]

    # Create a dictionary to hold information about the children of each node. The keys will children and the values
    # will be who their parent is.
    folks = {}

# Continue while there are still nodes in the queue (reference algorithms for a breadth first search for more info)
    while len(queue) > 0:
        # Set the node we are looking at equal to the node next in line in the queue
        current = queue[0]
        
        # Determine the children of the current node
        parent_bool =  np.reshape(nodes @ adj[:, current-1], (1, nodes.shape[0])) # vector of zeros and child names (numerical names)
        parents = parent_bool[np.nonzero(parent_bool)] #list of just the child names (numerical names)

        for parent in parents: # For every child of the current node that was found above...
            if nodeList[parent - 1] == False: # Check if the child has been visited. If not, continue with the following:
                queue.append(parent) # Append the child to the queue
                nodeList[parent - 1] = True # Change the status of the child to say we have visited it
                folks.update({parent: current}) # Add a key/value pair of the child (as key) and the current node (as value)
            
            # If the child is the target, then we have found the shortest path (any other paths will be the same length
            # or longer). So break the while loop and continue on to below.
            if parent == start:
                break
            
        # Remove the current node from the queue
        queue = queue[1:]
    
    # Start a trace of the path with the target node. We will iterate backwards through the path
    trace = start
    print(trace)

    # Add the target node to the graph
    G.add_node(nodeNames[trace - 1])

    # Initialize a list of edges in the path, initialize a list of nodes (numerical) in the path and add the target node, 
    # initialize a list of nodes (names - i.e. string type) and add the target node
    edge_path = []
    node_path = [start]
    sig_nodes = [nodeNames[trace - 1]]

    # Initialize a variable which counts how many nodes are in the graph (for plotting purposes)
    num_nodes = 0

    # Iterate backwards through the path
    while trace != target:
        child = folks[trace] # Find the child of the current node
        edge_path.append((nodeNames[trace- 1], nodeNames[child - 1])) # Append the edge between the current node and its parent
        node_path.append(child) # Append the parent to the list of nodes in the path
        trace = child # Update the current node to the parent node

    # Traverse through the path moving forward (recall that the path list is backwards, hence the reverse command)
    for node in reversed(node_path):

        # Add the curent node of the path to the graph
        G.add_node(nodeNames[node - 1])

        # Update the number of nodes in the path and add the current node to the list of nodes in our graph
        num_nodes += 1
        sig_nodes.append(nodeNames[node - 1])

        # Find the children of the current node
        children_bool = adj[node-1] @ nodes
        children = children_bool[np.nonzero(children_bool)] #list of child names (numerical names)

        # Find the parents of the current node
        parent_bool = nodes @ adj[:, node-1]
        parents = parent_bool[np.nonzero(parent_bool)] #list of parent names (numerical names)

        # Combine the list of child and parent nodes
        neighbors = np.concatenate((children, parents))
        
        # Iterate through the list of parents and child nodes
        for i in range(len(neighbors)):

            # If the neighbor node is a child node, then do the following:
            if i >= len(children):
                G.add_node(nodeNames[neighbors[i] - 1]) # Add the child to the graph
                num_nodes.append(nodeNames[neighbors[i] - 1]) # Add the child to the list of nodes in the graph
                G.add_edge(nodeNames[neighbors[i] - 1], nodeNames[node - 1]) # Add the edge between the current node and child
            else:
                continue

    # Obtain the adjacency matrix of the graph we constructed
    new_adj = nx.to_numpy_array(G)

    # Returns the graph and its adjacency matrix
    return G, new_adj, sig_nodes, node_path, edge_path