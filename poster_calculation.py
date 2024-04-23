import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graphBuilder import *
from spectralAnalysis import *
from searchMethods import *
from failureProbabilities import *
import copy
import time
import math
from arrayPropagation import *

''' poster_calculation.py -----------------------------------------------------------------------------------------------

            ****************************** Last Updated: 23 April 2024 ******************************

No methods. Run this code to calculate all pairwise failure probabilities in turbine (i.e. probability of A given B for
all nodes A and B). This document is seperate to cut down run time.

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------'''


# Initialize the adjacency matrix, list of node names, number of turbines, array layout, and initial failure probabilities.
arr, nodeNames = excel2Matrix("ExcelFiles/failureData.xlsx", "bigMatrix")
num_turbines = 10
effects_mark = 27
array_layout = [[0, [1, 2], [1], [1], [1], [2]], 
                [1, [0, 2, 3], [0, 2], [0, 2], [2], [3]],
                [2, [0,1,3,4], [1, 3], [1, 3], [3], [0, 4]], 
                [3, [1,2,4,5], [2, 4], [2, 4], [4], [1, 5]],
                [4, [2,3,5,6], [3, 5], [3, 5], [5], [2, 6]],
                [5, [3,4,6,7], [4, 6], [4, 6], [6], [3, 7]],
                [6, [4,5,7,8], [5, 7], [5, 7], [7], [4, 8]],
                [7, [5,6,8,9], [6, 8], [6, 8], [8], [5, 9]],
                [8, [6, 7, 9], [7, 9], [7, 9], [9], [6]],
                [9, [7,8], [8], [8], [8], [7]]]
probabilities = np.array([0.0195, 0.0195, 0.013625, 0.0055, 0.0175, 0.2075, 0.001, 0.001, 0.001, 0.093185, 0.001, 0.001,
                                        0.027310938, 0.033968125, 0.033968125, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375,
                                        0.0205, 0.0205, 0.02, 0.01, 0.01, 0.233, 0.288, 0.543374, 0.1285, 0.01, 0.01, 0.01, 0.015, 0.0155,
                                        0.015, 0.0155, 0.015, 0.0155, 0.015, 0.33, 0.025, 0.025, 0.025, 0.025, 0.025, 0.105]) #0.01375,

# For our calculations, we are calculating the probability of A given that B has already failed. So, we are interested in the forward
# propagation of failures. Thus, we calculate the "child" version of Bayesian inference.
parent_or_child = "child"

# Name the Excel file that we are writing to
filename = "turbineInference_" + "_" + parent_or_child + "blah.xlsx"


with pd.ExcelWriter(filename) as writer:
    all_probabilities = np.zeros((arr.shape[0]*num_turbines,arr.shape[1]*num_turbines)) # Initialize a large array to put all the pairwise probabilities in
    for start_turbine in range(num_turbines): # Iterate through each turbine in array
        for start_component in range(1,arr.shape[0]+1): # Iterate through each failure mode/effect in turbine
            print("Generating tree...")
            adj_mat = arr.copy()

            # Generate the forward propagation tree for the given starting turbine and starting failure mode/effect
            G, adj_array, gens, effects, modes, names_of_nodes = turbine_array_child(adj_mat, nodeNames, [start_component], num_turbines, array_layout, start_turbine, effects_mark, plot = False)

            
            nodeNamesArray = np.array([]) # Initialize array of names of nodes
            new_probabilities = [] # Initialize array of probabilities
            for node in G.nodes:
                name = node[3:]
                index = np.where(nodeNames == name)
                new_probabilities.append(probabilities[index]) # Add probabilitry to the array
                nodeNamesArray = np.append(nodeNamesArray, node) # Add name to the array

            nodeNamesArray = np.reshape(nodeNamesArray, (len(nodeNamesArray),)) # Make numpy array
            new_probabilities = np.reshape(np.array(new_probabilities), (len(new_probabilities), 1)) # Make numpy array

            print("------------", nodeNamesArray[0].replace("\n", " "), "------------")

            this_array = nx.to_numpy_array(G) # Initialize adjacency matrix for forward propagation tree
            a = this_array.copy()
            probabilitiy_table = np.zeros((2, this_array.shape[0])) # Initialize table of inference probabilities
            nodes = diagonal_nodes(a) # Diagonal matrix of node names (numerical +1)
            a = make_binary(a, 0.5) # Binarize adjacency table
            prblts = new_probabilities

        # Interence-----------------------------------------------------------------------
            for node in range(a.shape[0]):
                pts_bool = nodes @ a[:, node] # vector of zeros and child names (numerical names)
                pts = pts_bool[np.nonzero(pts_bool)] #list of just the child names (numerical names)

                if len(pts) < 1: # If no parents, add probability of failure happening to the probability table
                    probabilitiy_table[0][node] = prblts[node]
                    probabilitiy_table[1][node] = 1 - prblts[node]
                    continue

                parents, our_table = bayesian_table(a, node+1, True, nodeNamesArray, True, prblts) # Calculate the probability distribution table
                mlt_table = np.ones((our_table.shape[0],2)) # Initialize table for multiplying across rows of probability distribution table

            # Calculate Probability Table ------------------------------------------------------------
                for i in range(our_table.shape[0]):
                    for j in range(our_table.shape[1] - 2):
                        parent = int(parents[j])
                        if our_table[i,j] == 0:
                            our_table[i,j] = probabilitiy_table[0][parent - 1]
                            if probabilitiy_table[0][parent - 1] == 0:
                                break
                        else:
                            our_table[i,j] = probabilitiy_table[1][parent - 1]
                            if (parent-1 in evidence): # If the node's parent is the evidence, zero out the non-failing possibility
                                our_table[i,j] = 0
                        mlt_table[i,0] *= our_table[i,j] # Multiply the probabilities across the probability distribution table
                    mlt_table[i,1] = mlt_table[i,0] * our_table[i, -1] # Multiple by the probability of event not failing given combination of parent failure
                    mlt_table[i,0] *= our_table[i, -2] # Multiple by the probability of event failing given combination of parent failure
                sm_table = np.sum(mlt_table, axis = 0) #/np.sum(mlt_table) # Sum the products of probabilities across the columns
                probabilitiy_table[0][node] = sm_table[0] # Update the inference probability table with the probabilites just calculated
                probabilitiy_table[1][node] = sm_table[1]

                # Print and add probability of node to table
                print("Probability of ", nodeNamesArray[node].replace("\n", " "), "=", sm_table)
                index1 = int(nodeNamesArray[node][0])
                index2 = np.where(nodeNames == nodeNamesArray[node][3:])[0][0]
                all_probabilities[start_turbine * arr.shape[0] + start_component - 1][index1 * arr.shape[0] + index2] = sm_table[0]/np.sum(sm_table)

    # Write array to dataframe
    df3 = pd.DataFrame(all_probabilities)
    df3.to_excel(writer, sheet_name="allProbs")