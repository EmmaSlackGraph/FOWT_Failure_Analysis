import numpy as np
import networkx as nx
from graphBuilder import *
from graphDiagnostics import *
from graphml2nwx import *
from smallWorldMeasures import *
from numpy import *
from spectralAnalysis import *

''' spectralAnalysis----------------------------------------------------------------------------------------------

            ****************************** Last Updated: 23 February 2024 ******************************

 Methods:
 1) fiedler_graph_partition: inputs adjacency matrix --> outputs two arrays of nodes (clusters)

 3) extendedFiedlerPartition: inputs adjacency matrix, integer l --> outputs (up to 2**l) clusters

 2) voronoi_split: inputs graph, adjacency matrix, names of the nodes, centers --> outputs the voronoi clusters
 
------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------


    fielder_graph_partition Documentation
 -----------------------------------------
 This method inputs an adjacency matrix and outputs two clusters of nodes (as arrays of integers).'''

def fiedler_graph_partition(arr):
    # Make the graph unweighted and undirected
    arr = make_undirected_unweighted(arr)

    # Calculate the graph Laplacian by subtracting the new adjacency matrix from the matrix with the degrees of
    # each node on the diagonal.
    D = np.diag(np.sum(arr, axis=0))
    L = D - arr

    # Find the eigenvalues and eigenvectors of the graph Laplacian
    lambdas, vects = linalg.eig(L)

    # Find the minimum eigenvalue. Delete the minimum eigenvalue from the list of eigenvalues and delete its 
    # corresponding eigenvector from the list of eigenvectors
    minimum = np.where(lambdas == min(lambdas))[0][0]
    lambdas = np.delete(lambdas, minimum, 0)
    vects = np.delete(vects, minimum, 0)

    # Find the second smallest eigenvalue and its corresponding eigenvector
    second_min = np.where(lambdas == min(lambdas))[0][0]
    v2 = vects[second_min]

    # Initialize two lists - these will be the lists for each of our two clusters
    c1 = []
    c2 = []

    # Iterate through each node (the nodes are numbered)
    for i in range(arr.shape[0]):

        # If the ith entry in the eigenvector corresponding to the second smallest eigenvector is positive,
        # then add the ith node to the first cluster of nodes.
        if v2[i] > 0:
            c1.append(i)
        
        # Otherwise, add the ith node to the second cluster of nodes.
        else:
            c2.append(i)

    # Return the two clusters of nodes
    return c1, c2


''' extendedFiedlerPartitions Documentation
 -----------------------------------------
 This method inputs an adjacency array and a value l, which indicates how many eigenvectors we are interested in.
 This method extends the algorithm for finding Fiedler Partitions, where there are 2**l maximum partitions
 (see https://shainarace.github.io/LinearAlgebra/chap1-5.html#fiedler-partitioning. The method returns the clusters
 generated from this approach.'''

def extendedFiedlerPartitions(arr, l):
    # Generate the graph's unnormalized graph Laplacian (from an undirected and unweighted version of itself)
    A = make_undirected_unweighted(arr)
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    # Calculate the eigenvectors of the graph Laplacian and find the first l eigenvectors (not including the very first)
    eigenval, eigenvects = linalg.eig(L)
    v2vl = np.hstack([np.reshape(eigenvects[i+1], (arr.shape[0], 1)) for i in range(l)])

    # Initialize an array that has the maximum buckets in it
    clusters = [[] for i in range(2 ** l)]

    # For each node (row in the first l eigenvects)...
    for i in range(arr.shape[0]):
        parity = v2vl[i] > 0 # Find where the values in the row are positive
        binary_array = np.multiply(parity, 1) # Convert this row of TRUE/FAlse to 1s and 0s (1s mean entry is positive)

        # Convert the row of 1s and 0s (binary representation) to its base 10 representation
        binary_val = int(np.multiply(parity, 1) @ np.reshape(np.logspace(0, l, num = l, endpoint = False, base = 2.0), (len(binary_array), 1)))
        
        # Add the node to the correct array based on its associated binary value
        clusters[binary_val].append(i)

    # Return the clusters
    return clusters


''' voronoi_split Documentation
 ------------------------------------
 This method inputs a Networkx graph G, an adjacency matrix, names of the nodes, and centers desired for the voronoi
 calculation. This method outputs the clusters from the voronoi calculation.'''

def voronoi_split(G, cells):
    #Find the voronoi cells from the built-in Networkx method
    voronoi = nx.voronoi_cells(G, cells)

    # Convert the dictionaries of voronoi clusters into lists
    cell_division = [list(voronoi[i]) for i in voronoi]

    # Return the clusters
    return cell_division