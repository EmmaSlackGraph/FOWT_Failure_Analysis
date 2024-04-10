import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graphBuilder import *
from graphDiagnostics import *
from graphml2nwx import *
from Graph_Analysis.smallWorldMeasures import *
from numpy import *

''' spectralAnalysis----------------------------------------------------------------------------------------------

            ****************************** Last Updated: 21 February 2024 ******************************

 Methods:
 1) cosine_similarity: inputs vector one and vector two --> outputs cosine similarity of vector one and two

 2) make_undirected_unweighted: inputs adjacency matrix and (optional) threshold --> outputs undirected, unweighted
 version of the adjacency matrix

 3) similarity_matrix: inputs an adjacency matrix, measure of similarity --> outputs similarity matrix

 4) eigenval_eigenvects: inputs similarity matrix and boolean for calulation preferences --> outputs eigenvectors
 of the graph Laplacian

 5) kmeans: inputs square matrix, k (int), and number of desired iterations (int) --> outbuts centers, sums of 
 squared error, and clusters

 6) plot_clusters: inputs k (int), adjacency matrix, palette, names of nodes, clusters --> plots clusters 
 (nothing returned)
 
 7) get_clusters: inputs k (int), number of iterations, adjacency matrix, names of the nodes, similarity measure
 boolean for similarity matrix type (optional), boolean for plotting (optional) --> outputs clusters and centers

 8) fiedler_graph_partition: inputs adjacency matrix --> outputs two arrays of nodes (clusters)

 9) plot_fgp: inputs fielder-graph partition clusters, the adjacency matrix, and the names of the nodes --> plots 
 the two clusters (Nothing returned)

 10) voronoi_split: inputs graph, adjacency matrix, names of the nodes, centers --> outputs the voronoi clusters

 11) multipartiteClusterPlot: inputs graph, list of node names, clusters --> plots a multipartite graph and 
 outputs the graph

 12) extendedFiedlerPartition: inputs adjacency matrix, integer l --> outputs (up to 2**l) clusters

 13) spectralClustering: inputs adjacency matrix, integer k, type of laplacian desired --> outputs k clusters
 
------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------


   cosine_similarity Documentation
 -----------------------------------
 This method takes in two vectors and outputs the cosine similarity between them. In linear algebra, cosine
 similarity is defined as the measure of how much two vectors are pointing in the same direction. It can also
 be thought of cosine of the angle between the two vectors.'''

def cosine_similarity(v1, v2):
    # Find the dot product between the two vectors inputted
    dot_prod = np.transpose(v1) @ v2

    # Calculate and return the cosine similarity of the two vectors
    return (dot_prod) / ((np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2)))))



''' make_undirected_unweighted Documentation
 ---------------------------------------------
 This method takes in an adjacency matrix and threshold. It outputs an adjusted adjacency matrix that
 corresponds to the undirected and unweighted graph of the one inputted.'''

def make_undirected_unweighted(arr, threshold = 0):
    # Make sure that the array is a numpy array
    arr = np.array(arr)

    # Add the array to the transpose of itself. This makes the graph undirected.
    arr = arr + arr.T

    # Use the method from graphBuilder.py to binarize the matrix. This makes the graph unweighted.
    make_binary(arr, threshold)

    # Since the array now only consists of zeros and ones, make sure that it is of integer type
    arr = arr.astype(int)

    # Return the adjusted matrix
    return arr



''' similarity_matrix Documentation
 ------------------------------------
 This method inputs an adjacency matrix and similarity measure and outputs a similarity matrix. By default, this
 method will calculate similarity on shared children. But we can also calculate based on shared neighbors, parents,
 or distance to capsize and sink.'''

def similarity_matrix(arr, measure=None):
    # Initialize a matrix that is the same size as the adjacency matrix
    sim_matrix = np.zeros(arr.shape)

    # Calculate based on shared parents and children
    if measure == "neighbors":
        arr = make_undirected_unweighted(arr)

    # Calculate based on shared parents
    elif measure == "parents":
        arr = arr.T

    # Calculate based on shared distances to "capsize" and "sink" nodes
    elif measure == "disaster":
        arr_adj = np.zeros((arr.shape[0], 2))
        for row in range(arr.shape[0]):
            arr_adj[row]=[short_path_length(arr, 10, row), short_path_length(arr, 11, row)]#([diance to sink, distance to capsize])
            

    # For each pair i,j of nodes, calculate the cosine similarity using the rows of the inputted adjacency matrix
    # Update the i,j entry in the similarity matrix based off of the cosine similarity just calculated.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            cs = cosine_similarity(arr[i], arr[j])
            sim_matrix[i,j] = cs

    # Return the similarity matrix.
    return sim_matrix



''' eigenval_eigenvects Documentation
 -------------------------------------
 This method inputs a similarity matrix and (optional) boolean to determine exact method of calulation. If input
 TRUE, then only vector similaries in which the vectors are somewhat parellel (not at all anti-parellel or orthogonal)
 will be considered. This often comes in handy when trying to calculate eigenvalues and eigenvectors of the selected
 matrices. If the input is FALSE, then all similarities are considered. The method outputs a matrix of eigenvectors for
 the (adjusted) similarity matrix.'''

def eigenval_eigenvects(sim_matrix, w_or_sim=True):
    # Initially set the mat variable equal to the similarity matrix
    mat = sim_matrix

    # If the we want to only focus on positive similarities (or don't specify our wishes), then we find a matrix
    # with only the positive similarities from the similiarity matrix (and zero everywhere else).
    if w_or_sim:

        #Initialize this new matrix with zeros
        W = np.zeros(sim_matrix.shape)

        # Find where the similiaries are positive and copy those over to the same places in the new matrix
        W[np.where(sim_matrix>0)] = sim_matrix[np.where(sim_matrix > 0)]

        # Set the mat variable equal to the matrix we just created
        mat = W

    # Find a diagonal matrix such that the i,i entries is the degree of vertex i and everywhere else is zero
    D = np.diag(np.sum(mat, axis=0))

    # Find the unnormalized graph Laplacian by subtracting the matrix (acting like an adjacency matrix) from the 
    # diagonal matrix we just created.
    L = D - mat

    # Use the linear algebra package from numpy to find the eigenvalues and corresponding eigenvectors for the
    # unnormalized graph Laplacian we just calculated.
    lambdas, vects = linalg.eig(L)

    # Return the eigenvectors
    return vects



''' kmeans Documentation
 --------------------------------
 This method takes in a square matrix, integer k, and an integer indicating the number of desired iterations. The
 method outbuts the centers, sum of squared error, and clusters. The kmeans algorithm is a very popular algorithm
 which randomly places k data points (centers) among the data and finds which real data points are closests to
 which center. This is process is repeated in order to update the nodes in each cluster. Although not perfect, 
 typically more iterations mean a higher degree of accuracy.'''

def kmeans(data, k, n_ints):
    # Initialize an array J which will hold the sum of distances between each node and all the centers.
    J = []

    # Randomly choose k centers
    k_means = data[np.random.choice(range(data.shape[0]), k, replace=False)]
    
    # Repeat the following process for the number of iterations inputted
    for it in range(n_ints):

        # Calculate the sum of squared distances between each center and each node
        sqdist = np.sum((k_means[:, np.newaxis, :] - data)**2, axis = 2)

        # Find the shortest distance for each node (this shows us which center is closest to each node)
        closest = np.argmin(sqdist, axis = 0)

        # Append the sum of squared distances (like an error measure) to J. This allows us to keep track of
        # how well our kmeans algorithm progressed through each iteration
        J.append(np.sum(np.min(sqdist)))

        # Find average value of the nodes around each center
        for i in range(k):
            k_means[i, :] = data[closest == i, :].mean()

    # Once we have gone through all the iterations, append the final measure of error to J
    J.append(np.sum(np.min(np.sum((k_means[:, np.newaxis, :] - data)**2, axis = 2))))

    # Return the latest centers, sums of squared error, and clusters
    return k_means, J, closest



''' plot_clusters Documentation
 --------------------------------
 This method inputs k, an adjacency matrix, a pallette of colors (array type), names of the nodes, and clusters from
 the k-means algorithm. The method prints the clusters with the different colors. Nothing is returned.'''

def plot_clusters(k, arr, palette, nodeNames, clusters):
    # Construct the graph from the inputted adjacency matrix and names of nodes
    G, arr = matrix2Graph(arr, nodeNames, True)

    # Divide up the region for plotting into sectors based on the number of nodes in each cluster.
    thetas = [(2 * math.pi)/arr.shape[0]*len(np.where(clusters == i)[0]) for i in range(k)]

    # Find the polar coordinates based on the sectors just found. This creates a list of poslitions for each node,
    # seperated by which node is in which cluster.
    polor_coors = [[[0.55, thetas[i]/(len(np.where(clusters == i)[0]))*(j+1)+np.sum(thetas[:i])] for j in range(len(np.where(clusters == i)[0]))] for i in range(k)]

    pos = {}

    # For each cluster, find the nodes in the cluster and find the list of polar coordinates for that cluster
    for i in range(k):
        nodes = nodeNames[np.where(clusters == i)]
        positions = polor_coors[i]

        # For each node in the cluster, find the Cartesian coordinates (based on the polar coordinates
        # calculated above) and add them to a dictionary with the node's name as the key.
        for j in range(len(np.where(clusters == i)[0])):
            pos.update({nodes[j]: (positions[j][0]*math.cos(positions[j][1]), positions[j][0]*math.sin(positions[j][1]))})

    # For each cluster, plot the nodes in the cluster so all the nodes within a cluster are the same color, but
    # nodes in different clusters are different colors
    for i in range(k):
        nx.draw_networkx_nodes(G, pos, nodelist=nodeNames[np.where(clusters == i)], node_color=palette[i])

    # Draw the rest of the graph and plot it.
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.box(False)
    plt.show()

    # Nothing is returned
    return



''' get_clusters Documentation
 ------------------------------------
 This method inputs k (numnber of clusters), iterations desired, adjacency matrix, name of the nodes,
 similarity measure, and booleans for computation type and whether or not to plot the clusters. This 
 method outputs the clusters from the k-means algorithm'''

def get_clusters(k, iterations, arr, nodeNames, sim_measure = None, w_or_sim = True, plot = True):

    # Compute the similarity matrix
    sim_matrix = similarity_matrix(arr, sim_measure)

    # Find the eigenvectors from the similarity matrix
    U = eigenval_eigenvects(sim_matrix, w_or_sim)

    # Find the centers, errors, and clusters from the k-means algorithm
    centers, j_trace, clusters = kmeans(U, k, iterations)

    # Generate a graph from the adjacency matrix
    G, arr = matrix2Graph(arr, nodeNames, True)

    # These are the palette colors I chose. There are 47 of them, so k can be anywhere from 1-47
    palette = ["#f6f390", "#ffaeca", "#d799ff", "#84dfff", "#90f6b8", 
            "#ff0a9c", "#ae1857", "#18da8d", "#2033c1", "#40e331",
            "#f9583c", "#dd355b", "#143d55", "#1a1b52", "#312b78", 
            "#ff2323", "#feff5b", "#2a83ff", "#16254d", "#bcbcbc",
            "#f0e3ca", "#40a1c1", "#faeb57", "#647483", "#db000d",
            "#9f0733", "#720428", "#161515", "#04533a", "#05507c",
            "#0084ff", "#44bec7", "#ffc300", "#fa4f5d", "#d696bb",
            "#ff2323", "#feff5b", "#2a83ff", "#16254d", "#a17f7a",
            "#ceaca1", "#ece7e8", "#aab09d", "#909878", "#65737e", "#c99789", "#dde6d5"]
    
    # If we want to plot the clusters, then we will use the plot_clusters method.
    if plot:
        plot_clusters(k, arr, palette, nodeNames, clusters)

    # Return the centers and clusters from the k-means-algorithm
    return centers, clusters



''' fielder_graph_partition Documentation
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



''' plot_fgp Documentation
 ------------------------------------
 This method is meant to plot the clusters found from the fielder-graph partition. It inputs the clusters, the adjacency
 matrix, and the names of the nodes. This method plots the two clusters with different colors. Nothing is returned.'''

def plot_fgp(cs, arr, nodeNames):
    # Generate a graph from the adjacency matrix inputted
    G, arr = matrix2Graph(arr, nodeNames, True)

    # Divide up the region for plotting into sectors based on the number of nodes in each cluster.
    thetas = [(2 * math.pi)/arr.shape[0]*len(cs[i]) for i in range(2)]

    # Find the polar coordinates based on the sectors just found. This creates a list of poslitions for each node,
    # seperated by which node is in which cluster.
    polor_coors = [[[0.55, thetas[i]/(len(cs[i]))*(j+1)+np.sum(thetas[:i])] for j in range(len(cs[i]))] for i in range(2)]

    # Initialize a position dictionary
    pos = {}

    # There are only two clusters, so we only need to choose two colors
    palette = ["#f6f390", "#ffaeca"]

    # For each of the two clusters, find the names of the nodes in the clusters, find the positions for these nodes, and
    # add them the position dictionary (keys are node names and values are Cartesian coordinates of nodes).
    for i in range(2):
        nodes = nodeNames[cs[i]]
        positions = polor_coors[i]
        for j in range(len(cs[i])):
            pos.update({nodes[j]: (positions[j][0]*math.cos(positions[j][1]), positions[j][0]*math.sin(positions[j][1]))})

    # Draw the two clusters seperately, specifying the color for each cluster
    for i in range(2):
        nx.draw_networkx_nodes(G, pos, nodelist=nodeNames[cs[i]], node_color=palette[i])

    # Plot the rest of the graph
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.box(False)
    plt.show()

    # Nothing is returned
    return



''' voronoi_split Documentation
 ------------------------------------
 This method inputs a Networkx graph G, an adjacency matrix, names of the nodes, and centers desired for the voronoi
 calculation. This method outputs the clusters from the voronoi calculation.'''

def voronoi_split(G, arr, nodeNames, cells):
    #Find the voronoi cells from the built-in Networkx method
    voronoi = nx.voronoi_cells(G, cells)

    # Convert the dictionaries of voronoi clusters into lists
    cell_division = [list(voronoi[i]) for i in voronoi]
    
    # Create a list of the centers from the voronoi calculation
    labels = [str(x) for x in voronoi]

    # Divide up the region for plotting into sectors based on the number of nodes in each cluster.
    thetas = [(2 * math.pi)/arr.shape[0]*len(cell_division[i]) for i in range(len(cell_division))]

    # Find the polar coordinates based on the sectors just found. This creates a list of poslitions for each node,
    # seperated by which node is in which cluster.
    polor_coors = [[[0.55, thetas[i]/(len(cell_division[i]))*(j+1)+np.sum(thetas[:i])] for j in range(len(cell_division[i]))] for i in range(len(cell_division))]

    # Initialize a position dictionary
    pos = {}

    # These are the palette colors I chose. There are 47 of them, so k can be anywhere from 1-47
    palette = ["#f6f390", "#ffaeca", "#d799ff", "#84dfff", "#90f6b8", 
            "#ff0a9c", "#ae1857", "#18da8d", "#2033c1", "#40e331",
            "#f9583c", "#dd355b", "#143d55", "#1a1b52", "#312b78", 
            "#ff2323", "#feff5b", "#2a83ff", "#16254d", "#bcbcbc",
            "#f0e3ca", "#40a1c1", "#faeb57", "#647483", "#db000d",
            "#9f0733", "#720428", "#161515", "#04533a", "#05507c",
            "#0084ff", "#44bec7", "#ffc300", "#fa4f5d", "#d696bb",
            "#ff2323", "#feff5b", "#2a83ff", "#16254d", "#a17f7a",
            "#ceaca1", "#ece7e8", "#aab09d", "#909878", "#65737e", "#c99789", "#dde6d5"]
    
    # For each of the clusters, find the names of the nodes in the clusters, find the positions for these nodes, and
    # add them the position dictionary (keys are node names and values are Cartesian coordinates of nodes).
    for i in range(len(cell_division)):
        nodes = nodeNames[cell_division[i]]
        positions = polor_coors[i]
        for j in range(len(cell_division[i])):
            pos.update({nodes[j]: (positions[j][0]*math.cos(positions[j][1]), positions[j][0]*math.sin(positions[j][1]))})

    # For each cluster, plot the nodes in the cluster so all the nodes within a cluster are the same color, but
    # nodes in different clusters are different colors
    for i in range(len(cell_division)):
        nx.draw_networkx_nodes(G, pos, nodelist=nodeNames[cell_division[i]], node_color=palette[i])
    
    # Draw the rest of the graph (and unlike before, label which nodes are closest to which voronoi cells)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.box(False)
    plt.legend(labels)
    plt.show()

    # Return the clusters
    return cell_division



''' multipartiteClusterPlot Documentation
 -----------------------------------------
 This method inputs a Networkx graph, list of node names, and array of clusters (integers show which node is in which 
 cluster). This method plots a multipartite version of the graph, given the partition indicated in the clusters, and 
 outputs the graph.'''

def multipartiteClusterPlot(G, nodeNames, clusters):
    # Initialize a dictionary - we will use it to label each node with a specific cluster
    values = {}

    # Iterate through the clusters
    for i in range(len(np.unique(clusters))):

        # Find and iterate through the nodes in each cluster
        nodes = nodeNames[np.where(clusters == i)]

        # For each node in the cluster, add the label of the cluster to the node to the values dictionary
        for node in nodes:
            values.update({node: {"layer": i}})

    # Add the attributes determined above to the graph
    nx.set_node_attributes(G, values)

    # Plot the graph using a multipartite layout
    pos = nx.multipartite_layout(G, subset_key='layer')
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.box(False)
    plt.show()

    # Return the graph
    return G



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



''' spectralClustering Documentation
 -----------------------------------------
 This method inputs an adjacency array, k (how many clusters we want), and the type of Laplacian matrix that
 we want. This method uses eigenvectors and the k-means algorithm to cluster nodes. It outputs the clusters.'''

def spectralClustering(arr, k, laplacian = None):
    # Generate the graph's unnormalized graph Laplacian (from an undirected and unweighted version of itself)
    A = make_undirected_unweighted(arr)
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    # Depending on type of Laplacian desired, calculate the correct Laplacian matrix
    if laplacian == "normalized" or laplacian == "u2n": # u2n refers to unit 2-norm
        L = np.linalg.inv(np.sqrt(D)) @ L @ np.linalg.inv(np.sqrt(D))
    elif laplacian == "nrw":
        L = np.linalg.inv(D) @ L

    # Calculate the eigenvectors of the graph Laplacian and find the first l eigenvectors (not including the very first)
    eigenval, eigenvects = linalg.eig(L)
    vk = np.hstack([np.reshape(eigenvects[i], (arr.shape[0], 1)) for i in range(k)])

    # If we want all the rows to have unit 2-norm, update the rows to have unit 2-norm
    if laplacian == "u2n": # u2n refers to unit 2-norm
        for row in range(vk.shape[0]):
            vk[row] = vk[row]/(np.sqrt(np.sum(np.square(vk[row]))))


    # Use the k-means algorithm to cluster the rows into k clusters
    centers, J, clusters = kmeans(vk, k, 1000)

    # Return the clusters
    return clusters