import networkx as nx
from graphBuilder import *

''' graphDiagnostics--------------------------------------------------------------------------------------------

            ****************************** Last Updated: 21 February 2024 ******************************

 Methods:
 1) graph_diagnostics: input graph and adjacency matrix --> no returned value (values printed)

 2) efficiency: input graph and adjacency matrix --> output efficienty value

 3) get_degrees: input  adjacency matrix, the names of the nodes, and threshold --> outputs an array of dictionaries
 for each node's degrees.

 4) short_path_length: inputs adjacency matrix, target node, start node --> outputs length of shortest path
 
------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------


   graph_diagnostics Documentation
 ----------------------------------
 This method takes in a graph and its adjacency matrix and prints out several measurements of the graph. These
 measurements include if the graph is a directed acyclic graph, if it is eulerian, its efficiency, its flow
 hierarchy, its reciprocity, the voronoi cells for centers 11 and 12 [feel free to alter these centers to your
 desire!!], the number of each triad, and the average degrees for predecessors, succesors, and total neighbors of
 each node.'''

def graph_diagnostics(G, arr):

    # Using the function from networkx, we see if the inputted graph is directed and acyclic. We then print this
    # result.
    dag = nx.is_directed_acyclic_graph(G)
    print("Directed Acyclic Graph:", dag)

    # A graph is Eulerian if all the vertices have an even degree - meaning that it is traversable
    eulerian = nx.is_eulerian(G)
    print("Eulierian:", eulerian)

    # Efficiency, as it implies, is how well the network can exchange information. It is inversely related to
    # distances between nodes.
    efficient = efficiency(G, arr)
    print("Efficiency:", efficient)

    # Flow hierarchy is the fraction of graphs not in cycles
    flow_hierarchy = nx.flow_hierarchy(G)
    print("Flow Hierarchy:", flow_hierarchy)

    # Reciprocity is the likelihood that vertices are mutually linked
    reciprocity = nx.reciprocity(G)
    print("Overall Reciprocity:", reciprocity)

    # Voronoi cells tell us which nodes are closest to each node in the 'center.' In this case, I am setting
    # node 11 (capsize) and node 12 (sink) as the centers.
    voronoi = nx.voronoi_cells(G, [11, 12])
    print("Voronoi Cells:", voronoi)

    # There are 16 types of triads that can be made in a directed graph. This function measures how many of 
    # each triad there exists in the graph G.
    triad = nx.triadic_census(G)
    print("Triadic Census:", triad)

    # These three methods find average the degree of neighboring nodes for each node in the graph. A neighboring
    # node to a node i is defined as a node that is connected by an edge to node i.
    average_predecessors = nx.average_neighbor_degree(G, "in")
    average_successors = nx.average_neighbor_degree(G)
    average_neighbors = nx.average_neighbor_degree(G, "in+out")
    print("Average Predecessor Degree", average_predecessors)
    print("Average Successor Degree", average_successors)
    print("Average Neighbor Degree", average_neighbors)
    
    return



''' efficiency Documentation
 ----------------------------
 This method takes in a graph and its adjacency matrix. It outputs the efficiency of the graph (in this case, 
 since the graph is mostly unweighted, it is also equal to the global efficiency). From literature, the efficiency
 of a graph is defined as the sum of the inverse distances between all pairs of nodes in the graph all of which is 
 divided by the number of nodes times the number of nodes minus one.'''

def efficiency(G, arr):
    sum = 0

    # For each node in the graph, iterate through all the nodes. If 
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):

            # There was some issues since there was no shortest path between nodes 0 & 5 and 0 & 19. In this case,
            # I am assuming the distance to be 0.
            if ((i == 0 and j==5) or j==19) or (i == 5 or i == 25):
                continue

            # Add the inverse of the length of the shortest distance between the two nodes
            elif nx.shortest_path_length(G, i, j) ==0:
                continue
            else:
                sum += 1 / nx.shortest_path_length(G, i, j)

    # Return the efficiency of the graph
    return sum/(arr.shape[0] * (arr.shape[0] - 1))



''' get_degrees Documentation
 --------------------------------
 This method takes in an adjacency matrix, the names of the nodes, and threshold. It outputs an array of dictionaries for
 each node. The dictionaries consist of the node's in-degree (accessed by key 'inDeg'), out-degree (accessed by
 key 'outDeg'), and the name of the failure effect/mode (corresponding key is an integer specific to the node).'''

def get_degrees(arr, nodeNames, threshold):
    
    # Create copy of the adjacency matrix so that we can alter it without losing information about
    # our original adjacency matrix.
    arr_altered = arr

    # Binarization of adjacency matrix. The threshold determines the cutoff for what will be labeled
    # as a 1 versus a 0.  Anything above the threshold will be a 1, and anything below the threshold
    # will be set to 0.
    for i in range(0, arr_altered.shape[0]):
        for j in range(arr_altered.shape[0]):
            if arr[i,j] > threshold:
                arr_altered[i,j] = 1
            else:
                arr_altered[i,j] = 0

    # Calculating out degrees
    out_degrees = np.sum(arr_altered, axis=1)

    # Calculating in degrees
    in_degrees = np.sum(arr_altered, axis=0)

    # Initialize the array of degrees
    degrees = []

    # Iterate through each node in the graph
    for i in range(arr_altered.shape[0]):

        # Add a dictionary with the in-degree, out-degree, number, and name for each node
        degrees.append({'inDeg': in_degrees[i], 'outDeg': out_degrees[i], i:nodeNames[i]})

        # Print the information just added to the array
        print(degrees[-1])

    # Return the entire array of degree information
    return  degrees



''' short_path_length Documentation
 ----------------------------
 This method inputs an adjacency matrix, a target node, and a start node. We use an altered
 breadth first searh to find the shortest path between two nodes. We output the length of this path.'''

def short_path_length(arr, target, start):

    # We binarize the adjacency matrix so that relationships either exist or not (rather than having an intensity)
    adj = make_binary(arr).astype(int)

    # Create an array of booleans such that the ith entry indicates if the node has already been visited. Nodes
    # that have been visited are set to TRUE.
    nodeList = np.reshape(np.repeat(False, arr.shape[0]), (arr.shape[0], 1))

    # Create a diagonal matrix such that the value of the i,i entry is 1+1, referencing the node with name i+1
    nodes = np.diag([i+1 for i in range(adj.shape[0])])

    # Visit the starting node and add it to the queue
    nodeList[start-1] = True
    queue = [[start, 0]]

    # Continue while there are still nodes in the queue (reference algorithms for a breadth first search for more info)
    while len(queue) > 0:
        # Set the node we are looking at equal to the node next in line in the queue
        current = queue[0]
        
        # Determine the children of the current node
        children_bool = adj[current[0]-1] @ nodes # vector of zeros and child names (numerical names)
        children = children_bool[np.nonzero(children_bool)] #list of just the child names (numerical names)

        for child in children: # For every child of the current node that was found above...
            if nodeList[child - 1] == False: # Check if the child has been visited. If not, continue with the following:
                queue.append([child, current[1]+1]) # Append the child to the queue
                nodeList[child - 1] = True # Change the status of the child to say we have visited it
            
            # If the child is the target, then we have found the shortest path (any other paths will be the same length
            # or longer). So break the while loop and continue on to below.
            if child == target:
                #print(current[1]+1)
                return current[1]+1
            
        # Remove the current node from the queue
        queue = queue[1:]

def cluster_diagonstics(G, cluster):
    # Compute the adjacency matrix of the cluster
    arr = nx.to_numpy_array(G)

    # Calculate the max/min of the in/out/overall degrees
    in_degs = np.sum(arr, axis=1)
    max_in_deg = max(in_degs)
    max_in_deg_node = cluster[np.where(in_degs == max(in_degs))]
    print("Max in-degree of " + str(max_in_deg) + " at node(s) " + max_in_deg_node)

    min_in_deg = min(in_degs)
    min_in_deg_node = cluster[np.where(in_degs == min(in_degs))]
    print("Min in-degree of " + str(min_in_deg) + " at node(s) " + min_in_deg_node + "/n")

    out_degs = np.sum(arr, axis=0)
    max_out_deg = max(out_degs)
    max_out_deg_node = cluster[np.where(out_degs == max(out_degs))]
    print("Max out-degree of " + str(max_out_deg) + " at node(s) " + max_out_deg_node)

    min_out_deg = min(out_degs)
    min_out_deg_node = cluster[np.where(out_degs == min(out_degs))]
    print("Min out-degree of " + str(min_out_deg) + " at node(s) " + min_out_deg_node + "/n")

    all_degs = np.sum((arr + arr.T), axis=1)
    max_deg = max(all_degs)
    max_deg_node = cluster[np.where(all_degs == max(all_degs))]
    print("Max overall degree of " + str(max_deg) + " at node(s) " + max_deg_node)

    min_deg = max(all_degs)
    min_deg_node = cluster[np.where(all_degs == min(all_degs))]
    print("Min overall degree of " + str(min_deg) + " at node(s) " + min_deg_node + "/n")

    # Determine if the cluster is acyclic (we already know it is directed)
    dag = nx.is_directed_acyclic_graph(G)
    print("Directed Acyclic Graph:", dag)

    return # Return nothing