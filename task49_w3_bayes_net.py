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
import random

''' task49_w3_bayes_net.py --------------------------------------------------------------------------------------------

            ****************************** Last Updated: 11 June 2024 ******************************

No methods. Run this code to calculate all pairwise failure probabilities in turbine (i.e. probability of A given B for
all nodes A and B). This document is seperate to cut down run time.

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------'''


# Initialize the adjacency matrix, list of node names, number of turbines and initial failure probabilities.
user_input = input("What type of probability do you want to use? (type 'r', 'o', or 'a') ")
arr, nodeNames = excel2Matrix("ExcelFiles/W3_FMEA_Tables_Adj_Mat.xlsx", "CombinedMatrix")

# Determine array layout
user_input2 = int(input("How many turbines are you using?  "))
if user_input2 == 1:
    num_turbines = 1
    effects_mark = 46
    array_layout = [[0, [0], [0], [0], [0], [0]]]
elif user_input2 == 10:
    num_turbines = 10
    effects_mark = 46
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

# Initialize occurrence intervals
occurrence_intervals = {1:[0, 1*10**(-6)], 2:[1*10**(-6),50*10**(-6)], 3:[50*10**(-6),100*10**(-6)], 4:[100*10**(-6),1*10**(-3)],
                        5:[1*10**(-3),2*10**(-3)], 6:[2*10**(-3),5*10**(-3)], 7:[5*10**(-3),10*10**(-3)], 
                        8:[10*10**(-3),20*10**(-3)], 9:[20*10**(-3),50*10**(-3)], 10:[50*10**(-3),1]}

# Determine type of probability calculation to use and get compute those probabilities
if user_input.lower() == 'r':
    probabilities, nodeNames = getProbabilities("ExcelFiles/W3_FMEA_Tables_Adj_Mat.xlsx", "abc")
elif user_input.lower() == 'o':
    probabilities, nodeNames = getProbabilities("ExcelFiles/W3_FMEA_Tables_Adj_Mat.xlsx", "def")
    for i in range(len(probabilities)):
        index = round(probabilities[i] * 2)
        prob_range = occurrence_intervals[index]
        probabilities[i] = (prob_range[1]-prob_range[0])*random.random() + prob_range[0]
        

print("Probabilities obtained!!")

# For our calculations, we are calculating the probability of A given that B has already failed. So, we are interested in the forward
# propagation of failures. Thus, we calculate the "child" version of Bayesian inference.
parent_or_child = "child"

# Name the Excel file that we are writing to
filename = "W3_FMEA_turbineInference_"+user_input.lower()+str(num_turbines)+"_linear.xlsx"

# List the names of the nodes in our graph and ask if ready to continue to Bayesian inference
for turbine_number in range(num_turbines):
    for name in nodeNames:
        print(str(turbine_number)+": " + name.replace('\n', ' '))
user_input = input('Ready to continue? (y/n) ')
if 'q' in user_input: quit()

with pd.ExcelWriter(filename) as writer:
    all_probabilities = np.zeros((arr.shape[0]*num_turbines,arr.shape[1]*num_turbines)) # Initialize a large array to put all the pairwise probabilities in
    for start_turbine in range(num_turbines): # Iterate through each turbine in array
        for start_component in range(1,arr.shape[0]+1): # Iterate through each failure mode/effect in turbine
            print("Generating tree...")
            adj_mat = arr.copy()

            # Generate the forward propagation tree for the given starting turbine and starting failure mode/effect
            G, adj_array, gens, effects, modes, names_of_nodes = turbine_array_child(adj_mat, nodeNames, [start_component], num_turbines, array_layout, start_turbine, effects_mark, plot = True)
            
            nodeNamesArray = np.array([]) # Initialize array of names of nodes
            new_probabilities = [] # Initialize array of probabilities
            for node in G.nodes:
                name = node[3:]
                index = np.where(nodeNames == name)
                new_probabilities.append(probabilities[index]) # Add probabilitry to the array
                nodeNamesArray = np.append(nodeNamesArray, node) # Add name to the array

            print(len(new_probabilities))
            nodeNamesArray = np.reshape(nodeNamesArray, (len(nodeNamesArray),)) # Make numpy array
            if len(new_probabilities) == 1:
                if isinstance(new_probabilities[0], float):
                    new_probabilities = np.array([[new_probabilities[0]]])
                elif isinstance(new_probabilities[0], ndarray):
                    new_probabilities = np.array([[new_probabilities[0][0]]])

                print("Length 1", new_probabilities)
            else:
                new_probabilities = np.reshape(np.array(new_probabilities), (len(new_probabilities), 1)) # Make numpy array
            print(new_probabilities.shape)

            print("------------", nodeNamesArray[0].replace("\n", " "), "------------")

            this_array = nx.to_numpy_array(G) # Initialize adjacency matrix for forward propagation tree
            a = this_array.copy()
            probabilitiy_table = np.zeros((2, this_array.shape[0])) # Initialize table of inference probabilities
            nodes = diagonal_nodes(a) # Diagonal matrix of node names (numerical +1)
            a = make_binary(a, 0.5) # Binarize adjacency table
            prblts = new_probabilities.copy()

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
                            if parent-1 == 0: # If the node's parent is the evidence, zero out the non-failing possibility
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

    # Set the probability of failure i given failure i equal to 1
    for i in range(all_probabilities.shape[0]):
        all_probabilities[i][i] = 1
    
    # Write array to dataframe
    df3 = pd.DataFrame(all_probabilities)
    df3.to_excel(writer, sheet_name="allProbs")