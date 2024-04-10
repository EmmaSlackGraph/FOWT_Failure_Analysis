import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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
import random
from sympy import *

''' clusterAnalysis-----------------------------------------------------------------------------------------------

            ****************************** Last Updated: 6 March 2024 ******************************

 Methods:
 1) max_flow_mat: input adjacency matrix, names of nodes --> output matrix of maximum flow values between nodes

 2) hitting_time_mat: input adjacency matrix --> output matrix of expected hitting times between nodes

 3) page_rank_mat: input adjacency matrix --> output matrix of personalized page ranks/hitting probabilities between nodes

 4) shortest_path_mat: input adjacency matrix, names of nodes --> output matrix of shortest path lengths between nodes

 5) all_failures: input adjacency matrix, names of nodes --> output True if all the nodes can be visited, False otherwise

 6) connectivity_matrix: input adjacency matrix, names of nodes, type of comparison --> output desired connectivity matrix

 7) get_connectivity_histograms: input adjacency matrix, names of nodes --> plot connectivities for all measurements

------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------


   max_flow_mat Documentation
 -----------------------------------
 This method inputs an adjacency matrix and array of node names (strings). The function computs the maximum flow
 between each pair of nodes and places this value in a matrix. In this case, maximum flow refers to the minimum number
 of edges needed to be cut in order to disconnect the two nodes. We ouput the matrix of maximum flow values.'''


# 1) minimum number of edges that have to be removed from G to destroy every path between two vertices u and v (max flow alg)

def max_flow_mat(arr, nodeNames):
    # Find all pairs of node connectivity from Networkx (uses max flow calculation)
    G, arr = matrix2Graph(arr, nodeNames, True)
    short_dict = nx.all_pairs_node_connectivity(G)

    # Initialize a matrix to put the connectivity values into
    shortest_matrix = np.zeros(arr.shape)

    # For each pair of nodes, find the connectivity calculated in Networkx and add to the connectivity matrix
    for node_i in short_dict:
        connections = short_dict[node_i]

        for node_j in connections:
            i = np.where(nodeNames == node_i)[0][0]
            j = np.where(nodeNames == node_j)[0][0]

            shortest_matrix[i,j] = connections[node_j]

    # Return the connectivity matrix
    return shortest_matrix

''' hitting_time_matr Documentation
 ------------------------------------
 This method inputs an adjacency martrix. We compute the expected hitting time for each pair of nodes. Expected hitting
 time is the average number of steps needed when taking a random walk from the given start node to the end node. We return
 the matrix of all hitting times.'''

# 3) Hitting Time or Commute Time (random walks)

def hitting_time_mat(arr):
    # Initialize a binary version of the adjacency matrix and a matrix that we will but the expected hit times in
    adj = make_binary(arr, 0)
    hit_mat = np.zeros(arr.shape)

    # For each node A, we will find the expected hit times for all nodes when we start from A
    for A in range(arr.shape[0]):
        # Create the transition matrix for the graph and subtract it from the identity matrix
        num_sum = np.sum(adj, axis = 1)
        num_sum[num_sum == 0] = 1
        P = np.diag([1 for i in range(adj.shape[0])]) - np.array(adj) / (np.reshape(np.repeat(num_sum, adj.shape[0]), adj.shape))
        # Debugging --> print(np.array(adj) / (np.reshape(np.repeat(num_sum, adj.shape[0]), adj.shape)))
        # Debugging --> print(np.diag([1 for i in range(adj.shape[0])]))

        # Delete the Ath row and columns from the matrix so that we can find its inverse
        P = P.astype(float64)
        P = np.delete(P, A, 0)
        P = np.delete(P, A, 1)

        # Find vector v such that Pn = v where n is the expected hit times. v will always be the all ones matrix.
        ones = np.ones((adj.shape[0] - 1, 1))
        # Debugging --> print(linalg.inv(P) @ ones)

        # Solve for the expected hit times and add to the matrix of expected hit times.
        vect = np.reshape(linalg.inv(P) @ ones, (1, arr.shape[1]-1))
        hit_mat[A, :A] = vect[0, :A]
        hit_mat[A, A+1:] = vect[0, A:]

    # Debugging --> rint(np.where(hit_mat[0] == max(hit_mat[0])))
    # Return the matrix of expected hit times.
    return hit_mat

''' hitting_probability_mat Documentation
 -----------------------------------------
 This method inputs an adjacency martrix. We compute the hitting probability for each pair of nodes. Hitting probability
 is the probability that a random walk starting from a given node will end at a second given node. We put each hitting
 probability in a matrix and return this matrix.'''

def hitting_probability_mat(arr):
    # Initialize the adjacency matrix and connectivity matrix
    adj = make_binary(arr, 0)
    conn_mat = np.zeros(adj.shape)

    # For each node A, find the transition matrix associated with the graph and set the Ath row equal to zeros with a
    # 1 in the [A, A] entry.
    for A in range(arr.shape[0]):
        num_sum = np.sum(adj, axis = 1)
        num_sum[num_sum == 0] = 1
        P = np.array(adj) / (np.reshape(np.repeat(num_sum, adj.shape[0]), adj.shape))
        P[A, :] = np.zeros((1, adj.shape[0]))
        P[A, A] = 1

        # Find the identity matrix, but set the [A,A] entry equal to zero.
        I = np.diag([1 for i in range(adj.shape[0])])
        I[A, A] = 0

        # Compute M, which is a system of equations for finding the hitting probabilities
        M = (P - I).astype(float64)

        # Find the vector v such that Mh = v, where h is the vector of hitting probabilities. v will always be an
        # indicator vector (1 at the Ath spot and zeros everywhere else).
        zeros = np.zeros((adj.shape[0], 1))
        zeros[A] = 1

        # Compute the inverse of M and multiply by v to find the values of h (the hitting probabilities)
        answr = linalg.inv(M) @ zeros

        # Reshape the hitting probabilities into a horizontal (rather than vertical) vector and add to the connectivity matrix
        conn_mat[A] = np.reshape(answr, (1, arr.shape[1]))

    # Return the connectivity matrix when we have found the hitting probabilities for each node
    return conn_mat

''' page_rank_mat Documentation
 ------------------------------------
 This method inputs an adjacency martrix. We will attempt to find the probability that we land on a node j if we take
 a random walk starting at node i. This probability is calculated for each pair of nodes for 10 iterations (or whatever
 you set iterations equal to), averaged over the number of iterations, and added to a matrix of probabilities. We then
 return this matrix.'''

def page_rank_mat(arr):
    # Initialize the number of iterations, adjacency matrix, node (numerical) names as a diagonal array, likelihood of stopping
    # (alpha), and a matrix of probabilties
    iterations = 10
    adj = make_binary(arr)
    nodes = np.diag([i+1 for i in range(arr.shape[0])])
    alpha = 0.9
    shortest_matrix = np.zeros(arr.shape)

    # Debugging --> print(arr.shape)

    # For each pair of nodes, initialize the number of times we land on the target node equal to zero
    for start in range(arr.shape[0]):
        for target in range(arr.shape[1]):
            num = 0

            # Initialize the current node as the start node and declare that we are walking (have not stopped)
            current = start
            walking = True

            # For the number of iterations stated, ...
            for i in range(iterations):

                # ... if we are still walking ...
                while walking:

                    # ... find the nodes that we could move to next.
                    children_bool = adj[current - 1] @ nodes # vector of zeros and child names (numerical names)
                    children = children_bool[np.nonzero(children_bool)]

                    # At some probability, stop walking. Also stop walking if there is nowhere to walk to.
                    if random.random() > alpha or len(children) < 1:
                        walking = False

                    # Otherwise, find the next node to walk to.
                    elif len(children) > 0:
                        current = random.choice(children)

                # If we have stopped walking and it turns out that we landed on our target, add 1 to our counter.
                if current == target:
                    num += 1
                    # Debugging --> print("target found! - num =", num)
                
                # Start walking again, but once we start from a new node
                walking = True
            
            # After each node pair, find the average number of times we ended at the target and append to the matrix
            shortest_matrix[start][target] = num/iterations
            print("start:", start, "target:", target, "prob:", num/iterations)

    # Return the matrix of probabilities that we land on a given target node when starting from a given starting node
    return shortest_matrix

''' shortest_path_mat Documentation
 ------------------------------------
 This method inputs an adjacency martrix and list of node names. It returns a matrix with the shortest path length between 
 each pair of nodes. We make use of the breadth_first_child method from searchMethods.py. We return the matrix of path lengths.'''

def shortest_path_mat(arr, nodeNames):
    # Initialize a matrix of zeros
    shortest_matrix = np.zeros(arr.shape)

    # Go through each node and find all the other nodes it can reach using the breadth first child method. Use the layer
    # identification to obtain the length of each path, and put this length in a matrix.
    for start in range(arr.shape[0]):
        G, adj, gens, effects, modes = breadth_first_child(arr, nodeNames, start)
        for item in gens:
            # print(np.where(nodeNames == item)[0][0])
            # print(gens[item]["layer"])
            shortest_matrix[start - 1][np.where(nodeNames == item)[0][0]] = gens[item]["layer"]

    # Return the matrix containing all the shortest path lengths
    return shortest_matrix

''' all_failures Documentation
 ------------------------------------
 This method inputs an adjacency martrix. We determine if we can reach all the failures by computting the shortest paths
 between nodes for all node pairs. If at least one node can propagate to all the others, we return true. Otherwise, we
 return false.'''
 
def all_failures(arr, nodeNames):
    # Find all the shortest path lengths between nodes
    sm = shortest_path_mat(arr, nodeNames)

    for i in range(arr.shape[0]):

        # Since the shortest path matrix was initialized as an array of zeros, any zero that is not the node that
        # we are starting from (because a path of length zero gets you to yourself) tells us that the starting node
        # cannot reach the node with a 0 path length. But if the only node path with length zero is the one that
        # starts and ends in the same place, that node can reach all the others. So we return true.
        lst = np.where(sm[i] == 0)[0]
        if len(lst) < 2:
            return True
        
    # Return false otherwise
    return False

''' connectivity_matrix Documentation
 ------------------------------------
 This method inputs an adjacency martrix, list of node names, and a string indicating the type of connectivity
 calculation we would like to run. Then we call the appropriate connectivitity method above, and return the 
 adjacency matrix as a default. We return the connectivity matrix computed.'''

def connectivity_matrix(arr, nodeNames, type):
    # For the given type of connectivity calculation, compute the connectivity for each pair of nodes and return
    # a matrix such that the [i,j] entry is the connectivity from the ith node to the jth node
    if type == "max flow":
        return max_flow_mat(arr, nodeNames)
    elif type == "expected hitting time":
        return hitting_time_mat(arr)
    elif type == "hitting probability":
        return hitting_probability_mat(arr)
    elif type == "page rank":
        return page_rank_mat(arr)
    elif type == "shortest path":
        return shortest_path_mat(arr, nodeNames)
    else:
        return arr

''' get_connectivity_histograms Documentation
 ---------------------------------------------
 This method inputs an adjacency martrix and list of node nanes. For each kind of connectivity calculation, we
 compute the connectivity matrix, flatten it, and plot a histogram that counts all the connectivities from the 
 connectivity matrix. We plot and save the histogram for each of types of connectivity measurements outlined above,
 but we do not return anything.'''
    
def get_connectivity_histograms(arr, nodeNames):
    # List each of the different types of connectivity matrices we want
    types = ["max flow", "hitting probability", "expected hitting time", "shortest path", "adjacency"]

    for type in types:
        # Get the connectivity matrix and flatten
        conn_mat = np.reshape(connectivity_matrix(arr, nodeNames, type), (1, arr.shape[0]*arr.shape[0])).flatten()

        # Name the histogram file
        fig_name = 'Figures/nodeConnectivity/' + type + ' connectivity' + '.png'

        # Plot and save figure
        plt.hist(conn_mat, bins=15)
        plt.title("Histogram of " + type.title() + " Between Node Pairs in Graph")
        plt.ylabel("Number of Node Pairs")
        plt.xlabel(type.title() + " Value of Node Pair")
        plt.savefig(fig_name)
        plt.show()


arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
H, carr = matrix2Graph(arr, nodeNames, False)
nodeNames = np.array([i for i in range(arr.shape[0])])
target = 11
start = 16

G, clusters = get_one_cluster(arr, nodeNames, 2, "spectral", False, "unnormal", cluster = 0)
arr = nx.to_numpy_array(G)

# 
'''
G = nx.complete_graph(5, nx.DiGraph())
arr = nx.to_numpy_array(G)'''

'''G = nx.DiGraph()
G.add_edges_from([(0,1), (2,1), (1,3), (3,2), (3,4), (4,1), (2,4), (4,5), (5,2)])
arr = nx.to_numpy_array(G)'''
# print(arr)

# edges = []

incidence_matrix = np.zeros((arr.shape[0], len(G.edges)))
edges = list(G.edges)
# print("edges", edges)
# print(np.array(G.edges))
# print(arr.shape)
# print(edges.index((5, 2)))

for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i,j] > 0:
            index = edges.index((clusters[i], clusters[j]))
            # print("Index of", [i,j], "at", edges.index([i,j]))
            incidence_matrix[i, index] = -1
            incidence_matrix[j, index] = 1

cycles = nx.simple_cycles(G)
cycle_edge_array = []

i = 0
cycle = next(cycles)
has_next = True
cycle_array = np.zeros((incidence_matrix.shape[1] - incidence_matrix.shape[0], incidence_matrix.shape[1]))

while has_next and incidence_matrix.shape[0] < incidence_matrix.shape[1]:
    # print(cycle)
    cycle_edges = []
    cycle_row = np.zeros((1, incidence_matrix.shape[1]))
    for m in range(len(cycle)-1):
        cycle_edges.append([cycle[m], cycle[m+1]])
        index = edges.index((cycle[m], cycle[m+1]))
        cycle_row[0, index] = 10
    cycle_edges.append([cycle[-1], cycle[0]])
    index = edges.index((cycle[-1], cycle[0]))
    cycle_row[0, index] = -10

    new_matrix = np.vstack((incidence_matrix, cycle_row))
    if  np.linalg.matrix_rank(new_matrix.T) < incidence_matrix.shape[0]: # (np.array(Matrix(new_matrix).rref()[0])[-1] == np.ones((1, new_matrix.shape[1]))).any():#
        print(cycle)
        print("Added to matrix!")
        incidence_matrix = new_matrix
        cycle_edge_array.append(cycle_edges)
    else:
        print("Zeros!", i)
        i += 1
        

    # print(cycle)

    cycle = next(cycles, 14)
    if cycle == 14:
        has_next = False
        
# cycle_array = np.zeros((len(cycle_edge_array), len(G.edges)))

'''for i in range(len(cycle_edge_array)):
    # print(edge_list)
    for edge in cycle_edge_array[i]:
        
        if edge == cycle_edge_array[i][0]:
            cycle_array[i, index] = 10
        else:
            cycle_array[i, index] = 10'''

start_node = 0
start_current = 40

is_and_vs = incidence_matrix
# is_and_vs= np.vstack((incidence_matrix, cycle_array))
# print(is_and_vs.shape)


num_eqns_diff = is_and_vs.shape[0] - (is_and_vs.shape[1])
# print(is_and_vs.shape)
# print(num_eqns_diff)

solns = np.vstack((np.zeros((arr.shape[0], 1)), 100 * np.ones((is_and_vs.shape[0] - arr.shape[0], 1)))) #np.random.randint(0, 400, (is_and_vs.shape[0] - arr.shape[0], 1)))) # 
# solns[np.random.randint(incidence_matrix.shape[0], is_and_vs.shape[0] - num_eqns_diff - 1)] = 20 # np.random.randint(-5, 60)
solns[start_node] = -start_current
lin_alg_mat = np.hstack((is_and_vs, solns))
lam_for_rref = Matrix((lin_alg_mat[:is_and_vs.shape[1]]))
# print(is_and_vs)
# print(solns)
rref_mat = np.array(lam_for_rref.rref()[0])
print(rref_mat[:,-1])
# print(linalg.det(is_and_vs))
# print(linalg.inv(is_and_vs[:-num_eqns_diff]) @ solns)