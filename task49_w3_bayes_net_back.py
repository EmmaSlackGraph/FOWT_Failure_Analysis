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
from new_w3_calc import circularBayNet

''' task49_w3_bayes_net.py --------------------------------------------------------------------------------------------

            ****************************** Last Updated: 11 June 2024 ******************************

No methods. Run this code to calculate all pairwise failure probabilities in turbine (i.e. probability of A given B for
all nodes A and B). This document is seperate to cut down run time.

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------'''

# Initialize the adjacency matrix, list of node names, number of turbines
user_input = input("What type of probability do you want to use? (type 'r', 'o', or 'a') ")

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

# Create Bayesian network and gather information about it
bayNet = circularBayNet()
bayNet.create_circular_graph(num_turbines, user_input.lower())
arr = nx.to_numpy_array(bayNet.G)
nodeNames = list(bayNet.G.nodes)

# Initialize occurrence intervals
occurrence_intervals = {1:[0, 1*10**(-6)], 2:[1*10**(-6),50*10**(-6)], 3:[50*10**(-6),100*10**(-6)], 4:[100*10**(-6),1*10**(-3)],
                        5:[1*10**(-3),2*10**(-3)], 6:[2*10**(-3),5*10**(-3)], 7:[5*10**(-3),10*10**(-3)], 
                        8:[10*10**(-3),20*10**(-3)], 9:[20*10**(-3),50*10**(-3)], 10:[50*10**(-3),1]}

# Print the nodes in the Bayesian network and ask if user is ready to continue to Bayesian inference
for node in nodeNames:
    if num_turbines == 1 and node[0] == '0': print(node.replace('\n', ' ')[3:])
    else: print(node.replace('\n', ' '))
user_input = input("Ready to continue? (y/n) ")
if 'q' in user_input: quit()

# For our calculations, we are calculating the probability of A given that B has already failed. So, we are interested in the forward
# propagation of failures. Thus, we calculate the "child" version of Bayesian inference.
poc = "child"

# Name the Excel file that we are writing to
filename = "W3_FMEA_turbineInference_"+user_input.lower()+str(num_turbines)+"_circular.xlsx"

# For plotting
plot = False
if plot:
    effect_colors = ["#ffd6ed", "#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff", "#b1adff", "#e4adff", "#e5e5e5", "#e8d9c5"]
    mode_colors = ["#e5c0d5", "#e5a1a7", "#e5c8a7", "#e5e5a7", "#a7e5b4", "#a7cae5", "#9f9be5", "#cd9be5", "#cecece", "#d0c3b1"]
    mode_edges = ["#CCABBD", "#cc8f94", "#ccb294", "#cccc94", "#94cca0", "#94b4cc", "#8d8acc", "#b68acc", "#b7b7b7", "#b9ad9d"]
    pos = nx.spiral_layout(bayNet.G) #nx.multipartite_layout(bayNet.G, subset_key='layer')
    for node in bayNet.G.nodes:
        if node in bayNet.effects:
            nx.draw_networkx_nodes(bayNet.G, pos, nodelist=[node], node_color=effect_colors[int(node[0])], node_size=750, edgecolors=mode_colors[int(node[0])], node_shape="s")
        elif node in bayNet.systems:
            nx.draw_networkx_nodes(bayNet.G, pos, nodelist=[node], node_color=effect_colors[int(node[0])], node_size=750, edgecolors=mode_colors[int(node[0])], node_shape="p")
        elif node in bayNet.causes:
            nx.draw_networkx_nodes(bayNet.G, pos, nodelist=[node], node_size=750, node_shape="D")
        elif node in bayNet.fowts:
            nx.draw_networkx_nodes(bayNet.G, pos, nodelist=[node], node_color=effect_colors[int(node[0])], node_size=750, edgecolors=mode_colors[int(node[0])], node_shape="h")
        elif node in bayNet.farm:
            nx.draw_networkx_nodes(bayNet.G, pos, nodelist=[node], node_size=750, node_shape="s")
        else:
            nx.draw_networkx_nodes(bayNet.G, pos, nodelist=[node], node_color=effect_colors[int(node[0])], node_size=750, edgecolors=mode_colors[int(node[0])])
    nx.draw_networkx_labels(bayNet.G, pos, font_size=5, verticalalignment='center_baseline')
    nx.draw_networkx_edges(bayNet.G, pos, arrowsize=20)
    plt.box(False)
    plt.show()

with pd.ExcelWriter(filename) as writer:
    all_probabilities = np.zeros((arr.shape[0],6)) # Initialize a large array to put all the pairwise probabilities in
    for start_component in range(1,arr.shape[0]+1): # Iterate through each failure mode/effect in turbine

        # Generate the forward propagation tree for the given starting turbine and starting failure mode/effect
        print("Generating tree...")
        a = arr.copy()
        non = nodeNames
        K, a, g, e, m, non = breadth_first_multi(a, nodeNames, [start_component], poc) # Generate tree for Bayesian network
        new_probabilities = [] # Initialize array of node probabilities (in order of appearance in graph)
        for node in non:
            new_probabilities.append(bayNet.G.nodes[node]['probability']) # Add nodes to array of node probabilities
        new_probabilities = np.array(new_probabilities)

        probabilitiy_table = np.zeros((2, a.shape[0])) # Initialize table of inference probabilities
        nodes = diagonal_nodes(a) # Diagonal matrix of node names (numerical +1)
        a = make_binary(a, 0.5) # Binarize adjacency table

        nodeNamesArray = np.array(list(K.nodes)) # Initialize array of names of nodes
        nodeNamesArray = np.reshape(nodeNamesArray, (len(nodeNamesArray),)) # Make numpy array

        print("------------", nodeNamesArray[0].replace("\n", " "), "------------")

        prblts = new_probabilities.copy()

        # Interence-----------------------------------------------------------------------
        runs = []
        for run in range(2):
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
                                our_table[i,j] = run
                        mlt_table[i,0] *= our_table[i,j] # Multiply the probabilities across the probability distribution table
                    mlt_table[i,1] = mlt_table[i,0] * our_table[i, -1] # Multiple by the probability of event not failing given combination of parent failure
                    mlt_table[i,0] *= our_table[i, -2] # Multiple by the probability of event failing given combination of parent failure
                sm_table = np.sum(mlt_table, axis = 0) #/np.sum(mlt_table) # Sum the products of probabilities across the columns
                probabilitiy_table[0][node] = sm_table[0] # Update the inference probability table with the probabilites just calculated
                probabilitiy_table[1][node] = sm_table[1]

                # Print and add probability of node to table
                print("Probability of ", nodeNamesArray[node].replace("\n", " "), "=", sm_table)
            runs.append(probabilitiy_table[0][-1])
        start_component_probability = runs[0]/(runs[0] + runs[1])
        start_component_name = list(bayNet.G.nodes)[start_component - 1]
        start_component_layer = bayNet.G.nodes[start_component_name]['layer']
        all_probabilities[start_component - 1][start_component_layer] = start_component_probability
            
    # Write array to dataframe
    df3 = pd.DataFrame(all_probabilities)
    df3.to_excel(writer, sheet_name="allProbs")