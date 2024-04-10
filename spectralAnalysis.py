import numpy as np
from graphBuilder import *
from graphDiagnostics import *
from graphml2nwx import *
from smallWorldMeasures import *
from numpy import *

''' spectralAnalysis----------------------------------------------------------------------------------------------

            ****************************** Last Updated: 23 February 2024 ******************************

 Methods:
 1) cosine_similarity: inputs vector one and vector two --> outputs cosine similarity of vector one and two

 2) make_undirected_unweighted: inputs adjacency matrix and (optional) threshold --> outputs undirected, unweighted
 version of the adjacency matrix

 3) similarity_matrix: inputs an adjacency matrix, measure of similarity --> outputs similarity matrix

 4) eigenval_eigenvects: inputs similarity matrix and boolean for calulation preferences --> outputs eigenvectors
 of the graph Laplacian

 5) kmeans: inputs square matrix, k (int), and number of desired iterations (int) --> outbuts centers, sums of 
 squared error, and clusters

 6) spectralClustering: inputs adjacency matrix, integer k, type of laplacian desired --> outputs k clusters
 
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