import numpy as np
import networkx as nx
import scipy
from numpy import random
from graphBuilder import *

''' smallWorldMeasures--------------------------------------------------------------------------------------------

            ****************************** Last Updated: 19 February 2024 ******************************

 Methods:
 1) apl_and_cc: input the adjacency matrix --> output the average path length and average clustering coefficient

 2) total_degree: input the adjacency matrix --> output the total number of edges (i.e. total degree)

 3) generate_lattice: input the adjacency matrix --> output the adjacency matrix of the corresponding lattice graph

 4) generate_random: input the adjacency matrix --> output the adjacency matrix of the corresponding random graph

 5) sigma_and_omega: input the adjacency matrix --> output the values of sigma and omega (as tuple)

 6) is_directed_small_world: input value (either sigma or omega) and type (either sigma or omega) --> output
 boolean of True if small world and False if not smallworld.
 
------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------


   apl_and_cc Documentation
 -------------------------------
 This method takes in an adjacency matrix and outputs the average path length and clustering coefficient of the
 graph inputted. The average clustering coefficient has a method through networkx, so we make use of this method.
 There is no function for the average path length, so we compute it ourselves with the help of other networkx
 commands. Although average path length and clustering coefficient do not directly indicate if a graph is small
 world, they are used in the calculation of small world measures.'''

def apl_and_cc(arr):

    # Create an array of the numeric names of the nodes and use the matrix2Graph method from excel2Graph.py in 
    # order to create a graph for the inputted adjacency matrix. For simplicity, we do not want node labels.
    nodeNames = range(arr.shape[0])
    G, var = matrix2Graph(arr, nodeNames, nodeLabels = False)

    # networkx already has an average clustering coefficient method to use, so we are making use of it here
    average_clustering_coefficient = nx.average_clustering(G)

    # Unfortunately, the average path length for directed graphs is not built into networkx's package. We thus find
    # all the shortest paths between nodes. Then we will average these paths.
    k = nx.all_pairs_dijkstra_path_length(G)

    # To start, we are creating two blank arrays. The 'dictionary' array will be used to access the information from
    # the iterator that is spit-out from 'nx.all_pairs_dijkstra_path_length()'. The 'shortest_paths' array will collect
    # the numerical values for each path length.
    dictionary = []
    shortest_paths = []

    # For each node 'i', access the dictionary containing all shortest paths such that 'i' is the source node.
    for i in range(arr.shape[0]):
        dictionary.append(next(k))

        # For each node 'j' in the dictionary of shortest paths, access the value of the path length
        for j in range(len(dictionary[i][1])):
            if j in dictionary[i][1].keys():

                # Append the value of this path length to the shortest_paths array
                shortest_paths.append(dictionary[i][1][j])

    # Take the average of the shortest path lengths by summing over shortest_paths and dividing by the number of 
    # shortest paths (i.e. the length of shortest_paths)
    average_path_length = sum(shortest_paths)/len(shortest_paths)

    # Return the average path length and clustering coefficient
    return average_path_length, average_clustering_coefficient



''' total_degree Documentation
 --------------------------------
 This method takes in an adjacency matrix and outputs the total degree of the matrix (i.e. the number of the edges).
 We begin by binarizing the adjacency matrix and then summing up across the entire matrix (this only works for directed
 graphs - in undirected graphs, we would sum across the entire matrix and then divide by two since directionality does
 not matter).'''

def total_degree(arr):

    # Create a copy of the adjacency matrix so not to ruin the original adjacency matrix
    arr_altered = arr

    # For every row 'i' and column 'j', binarize the adjacency matrix based on the threshold that anying above 0.5
    # should be treated as a 1 while everything else should be set to zero.
    for i in range(0, arr_altered.shape[0]):
        for j in range(arr_altered.shape[0]):
            if arr[i,j] > 0.5:
                arr_altered[i,j] = 1
            else:
                arr_altered[i,j] = 0

    # Return the sum of all the ones in the adjacency matrix, which is analogus to the total degree of the graph
    return np.sum(arr_altered)



''' generate_lattice Documentation
 ----------------------------------
 This method takes in an adjacency matrix and outputs an adjacency matrix for the corresponding lattice graph 
 (which means that the lattice graph has the same number of nodes and roughly the same number of edges as the given 
 graph). 
 
 We do this by creating a 1-dimensional array and using scipy's 'linalg.circulant' method to cycle through
 this array and create a matrix. What this means is that every new row matrix is created by shifting all the numbers of
 the previous row one space to the right (and whatever is on the end becomes the number to the farthest on the left of 
 the row). By making sure that the second entry of the array is not equal to the last entry in the array, we are 
 ensuring that the resulting graph will be directed. 
 
 Also, by placing ones on each end of the array (so to speak), we have more control over the average degree of the 
 lattice graph and ensure that the lattice graph better mimics the original graph. This latter property is due to fact
 that our original graph has edges pointing to both high-valued and low-valued nodes (kind of like saying that the node
 points to numbers ahead of and behind it).'''

def generate_lattice(arr):

    # Find the total number of edges in the original graph
    num_edges = total_degree(arr)

    # Find the average degree (round to the nearest integer) by dividing the number of edges by the number of nodes
    average_degree = int(num_edges/arr.shape[0])

    # Initialize the first row (the one we will circulate through) with all zeros. We will change certain zeros with
    # ones in order ot draw certain edges.
    circ_col = np.zeros(arr.shape[1])
    
    # If the average degree is odd, then add one. This is because odd degrees are harder to work with and we only need
    # an approximation of what a lattice-version of the inputted graph would look like. Each node should have this new
    # number as its out-degree in our calculated lattice matrix.
    if (average_degree % 2) == 1:
        average_degree += 1
    
    # Place ones in the first row so that each node will point to average_degree/2 nodes directly after it (in numerical
    # order) and average_degree/2 nodes directly before it (but not including the node directly behind it).
    for i in range(int(average_degree / 2)):
        circ_col[i + 1] = 1
        circ_col[arr.shape[1] - i - 2] = 1

    # Cycle the first row in order to create the lattice graph's adjacency matrix
    latt_adj = scipy.linalg.circulant(circ_col)

    # Return the adjacency matrix of the lattice graph
    return latt_adj



''' generate_random Documentation
 ----------------------------------
 This method takes in an adjacency matrix and outputs an adjacency matrix for the corresponding random graph 
 (which means that the random graph has the same number of nodes and roughly the same number of edges as the given 
 graph). We do this by randomly choosing x (x being equal to the average degree of the inputted graph) entries in the 
 i-th row of the adjacency matrix to be equal to 1. It is important that these values be random and not equal to i.'''

def generate_random(arr):

    # Find the total number of edges in the given graph
    num_edges = total_degree(arr)

    # Calculate the average degree of the given graph
    average_degree = int(num_edges/arr.shape[0])

    # Initialize the adjacency matrix of the random graph with zeros
    rand_adj = np.zeros(arr.shape)
    
    # In order to keep in line with what we did with the lattice graphs, round up the average degree to the nearest
    # even integer
    if (average_degree % 2) == 1:
        average_degree += 1
    
    # For each row in the matrix, choose some number of indices to set to 1. If any of the indices are the same number
    # as the row index, then select new indices. Keep doing this until you have indices not equal to the row index. Then
    # put 1s into those indices.
    for i in range(arr.shape[0]):
        rand_edges = random.randint(arr.shape[0], size = average_degree)
        while i in rand_edges:
            rand_edges = random.randint(arr.shape[0], size = average_degree)
        rand_adj[i][rand_edges] = 1

    # Return the adjacency matrix of the random graph
    return rand_adj



''' sigma_and_omega Documentation
 ----------------------------------
 This method takes in an adjacency matrix and outputs sigma and omega values. Recall that sigma and omega are measures
 of smallworldness. As defined in literature, a graph is smallworld if its sigma value is greater than one or if its 
 omega value is near zero.'''

def sigma_and_omega(arr):

    # We first generate our random and lattice counterparts, as we need them for our calculations
    random_counterpart = generate_random(arr)
    lattice_counterpart = generate_lattice(arr)

    # We find the average path length and clustering coefficient for both random and lattice graphs, as well as our graph
    # in question.
    random_apl, random_cc = apl_and_cc(random_counterpart)
    lattice_apl, lattice_cc = apl_and_cc(lattice_counterpart)
    graph_apl, graph_cc = apl_and_cc(arr)

    # From the equations in literature, we calculate the values for sigma and omega.
    sigma = (graph_cc/random_cc)/(graph_apl/random_apl)
    omega = (random_apl/graph_apl) - (graph_cc/lattice_cc)

    # We return the values for sigma and omega
    return sigma, omega



''' is_directed_small_world Documentation
 -----------------------------------------
 This method takes in a value and a string. The value is either a sigma or omega value. The string indicates which
 of the two is inputted into the method. Then from literature, we return True (that the graph in question is small
 world) if the sigma value is greater than one or it the omega value is near zero.'''

def is_directed_small_world(value, type):

    # If the sigma value is greater than 1, then return True
    if type == "sigma" and value > 1.0:
        return True
    
    # If the omega value is near zero, then return True
    elif type == "omega":
        if value < 0.5 and value > -0.5:
            return True
        
    # If the wrong type of value was inputted, return a string that says there was an error
    elif type != "sigma" and type != "omega":
        return "Incompatible Value Type: Please enter \"sigma\" or \"omega\""
    
    # Otherwise, return False
    return False