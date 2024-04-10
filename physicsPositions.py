import numpy as np
import networkx as nx
from graphBuilder import *
from graphDiagnostics import *
from graphml2nwx import *
from smallWorldMeasures import *
from spectralAnalysis import *
from clusterAnalysis import *
from plotClusters import *

''' smallWorldMeasures--------------------------------------------------------------------------------------------

            ****************************** Last Updated: 20 February 2024 ******************************

 Methods:
 1) mag: inputs vector  --> outputs magnitude of vector

 2) fa: inputs distances x and k --> outputs attractive force

 3) fr: inputs distances x and k --> outputs negative repulsive force

 4) cool: inputs temperature t and constant k --> outputs cooled temperature

 5) barycenter_draw: inputs adjacency matrix and list of node names --> outputs an dictionary of nonde positions

 6) fruchtermanReingoldMethod: inputs graph, adjacency matrix, list of node names, number of iterations,
 temperature contraint --> outputs a dictionary of node positions
 
------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------


   mag Documentation
 ---------------------
 This method inputs a vector (1D array) and outputs the magnitude of that vector.'''

def mag(v):
    # Return the magnitude of the vector
    return np.sqrt(np.sum(np.square(v)))



''' fa Documentation
 ---------------------
 This method takes in x (distance between two nodes) and k (optimal distance between the two nodes) and outputs 
 the attractive force between the two nodes'''

def fa(x, k):
    # Return the attactive force between the nodes based on the distance and optimal distance between the nodes
    return x**2/k



''' fa Documentation
 ---------------------
 This method takes in x (distance between two nodes) and k (optimal distance between the two nodes) and outputs 
 the negative repulsive force between the two nodes'''

def fr(x, k):
    # Return the negative repulsive force between the nodes based on the distance and optimal distance between the nodes
    return k**2/x



''' cool Documentation
 ---------------------
 This method takes in a temperature and constant k, and it outputs the cooled temperature after one time step.'''

def cool(t, k):
    # Return the cooler temperature (linear descend throughout number of iterations)
    return t - k/10



''' barycenter_draw Documentation
 -----------------------------------
 W. T. Tutte is often credited with formulating the first 'force-directed' algorithm for graph visualization. The
 Barycentric Method implemented here attempts to find an equilibrium between attractie and repulsive forces of the
 nodes of the graph. This method assumes that the ideal length of an edge is zero and no explicit repulsive forces.
 Since this method would eventually converge where all the nodes are at the same position, we fix some of the nodes
 before starting. Then we find the best position for all the other nodes by placing them in the center of all its 
 neighbors.
 
 So, this method takes in an adjacency matrix and list of node names, and outputs an dictionaary of node positions.'''

def barycenter_draw(arr, nodeNames):
    # Make the graph undirected and unweighted for our calculation of forces between nodes (we don't care the direction
    # of the connection, just that there is a connection because the edges are thought of as springs)
    mat = make_undirected_unweighted(arr)

    # Calculate the unnormalized graph Laplacian
    D = np.diag(np.sum(mat, axis=0))
    L = D - mat

    # Partition the graph into two clusters. We will fix one cluster as a circle and plot the second cluster inside the
    # circle.
    centers, clusters = get_clusters(2, 1000, arr, nodeNames, sim_measure = None, plot=False)

    # Create an array of the names (numerical) of the nodes
    nodes = np.array(range(arr.shape[0]))

    # Find which nodes are in which cluster
    v0 = nodes[np.where(clusters == 0)]
    v1 = nodes[np.where(clusters == 1)]

    # Generate a circular graph with the same number of nodes as in the first cluster and find the positions of the nodes
    P = nx.circulant_graph(len(v0), [1])
    pos = nx.circular_layout(P)

    # Initialize a position dictionary
    v_pos = {}

    # Initialize a sum of the x and y values of the positions of the fixed nodes
    sum_x = 0
    sum_y = 0

    # Iterate through all the nodes in the first cluster
    for i in reversed(range(len(v0))):
        
        # Assign the positions from the circular graph to the nodes in the first cluster and add to position dictionary
        v_pos.update({nodeNames[v0[i]]: pos[i]})

        # Remove the row and column of the fixed node from the graph Laplacian
        L = np.delete(L, v0[i], 0)
        L = np.delete(L, v0[i], 1)

        # Add the x and y value of the fixed node's position to the running total
        sum_x += pos[i][0]
        sum_y += pos[i][1]

    # Create vertical vectors that repeat the sum of the total x positions and total y positions of the fixed nodes
    vect_bx = np.reshape(np.repeat(sum_x, L.shape[0]), (L.shape[0], 1))
    vect_by = np.reshape(np.repeat(sum_y, L.shape[0]), (L.shape[0], 1))

    # Multiply the inverse of the adjusted Laplacian by the vectors just calculated to find the positions of the
    # free nodes (free nodes are the nodes that are not fixed)
    xs = linalg.inv(L) @ vect_bx # x-positions of free nodes (ith value in vector is x-position for ith free node)
    ys = linalg.inv(L) @ vect_by # y-positions of free nodes (ith value in vector is y-position for ith free node)

    # Go through the x and y vectors calculated for the free nodes and add them to the positions dictionary
    for index in range(len(v1)):
        v_pos.update({nodeNames[v1[index]]: np.hstack((xs[index], ys[index]))})
    
    # Return the position dictionary
    return v_pos



''' fruchtermanReingoldMethod Documentation
 ------------------------------------------
 This method inputs a graph, adjacency matrix, list of node names, number of iterations,and boolean for plotting graphs. 
 This method outputs a dictionary of node positions. This method uses the Fruchterman-Reingold method of visualizing 
 graphs via spring embeddings (i.e. treat edges as springs).'''

def fruchtermanReingoldMethod(G, arr, nodeNames, iterations, graph_trace=False):
    # Set the size of the plotting frame
    w = 2 # frame width
    l= 2 # frame length
    area = w * l # frame area

    t = w / 10

    # Initialize the positions of the nodes to random positions
    pos = nx.random_layout(G)

    # Initialize an array for the displacement of nodes from their original position
    disps = np.zeros([arr.shape[0], 2])

    # Set the optimal distance between nodes as the squareroot of the area of the frame divided by the number of nodes
    k = np.sqrt(area/arr.shape[0])

    # Repeat the following for the inputted number of iterations
    for i in range(iterations):

        # If the user wants to see all the plots along the way, plot the graphs
        if graph_trace:
            nx.draw_networkx_nodes(G, pos)
            nx.draw_networkx_edges(G, pos)
            nx.draw_networkx_labels(G, pos)
            plt.show()
        
        # For each node v in the graph...
        for v in range(arr.shape[0]):

            # Set the new displacement equal to zero
            disps[v] = [0., 0.]

            # For all other nodes in the graph that are not node v...
            for u in range(arr.shape[0]):
                if u != v:

                    # Find the difference between the posotions of the two nodes v and u
                    delta = np.array(pos[nodeNames[v]]) - np.array(pos[nodeNames[u]])
                    #print(i, np.array(pos[nodeNames[v]]), np.array(pos[nodeNames[u]]), delta)

                    # Update the displacement based on difference of nodes and repulsive force between the nodes
                    disps[v] = disps[v] + (delta/mag(delta)) * fr(mag(delta), k)
        
        # For each edge e in the graph...
        for e in get_graph_edges(arr, 'array', 0):

            # Find the nodes the edge connects
            v = e[0]
            u = e[1]

            # Find the length of the edge
            delta = np.array(pos[nodeNames[v]]) - np.array(pos[nodeNames[u]])

            # Update the displacement of the edge's nodes based on attactive forces
            disps[v] = disps[v] - (delta/mag(delta))*fa(mag(delta), k)
            disps[u] = disps[u] + (delta/mag(delta))*fa(mag(delta), k)

        # For each node v in the graph...
        for v in range(arr.shape[0]):
            # Update the position of the node v based on the displacement values (limiting max displacement to
            # temperature t)
            new_pos_v = np.array(pos[nodeNames[v]]) + (disps[v]/mag(disps[v]))*min([mag(disps[v]), t])

            # Prevent from displacement outside of the frame
            new_x = min([w/2, max([-w/2, new_pos_v[0]])])
            new_y = min([l/2, max([-l/2, new_pos_v[1]])])

            # Update the new position in the position dictionary
            pos[nodeNames[v]] = (new_x, new_y)
        
        # Reduce the temperature as the positions reach better configurations
        t = cool(t, w/iterations)

    # Return the position dictionary
    return pos

