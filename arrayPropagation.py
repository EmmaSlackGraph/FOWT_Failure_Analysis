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

''' arrayPropagation.py -----------------------------------------------------------------------------------------------

            ****************************** Last Updated: 24 April 2024 ******************************

 Methods:
 1) turbine_array_child: input adjacency matrix, node names, starting node, number of turbines, effects_mark -->
 output graph

 2) turbine_array_parent:input adjacency matrix, node names, starting node, number of turbines, effects_mark -->
 output graph

 3) turbine_array_child_prob: input adjacency matrix, node names, starting node, number of turbines, effects_mark,
 update boolean --> output graph

 4) turbine_array_parent_prob:input adjacency matrix, node names, starting node, number of turbines, effects_mark,
 update boolean --> output graph

 5) monte_carlo_sim_array: inputs number of iterations, number of turbines, plotting boolean, starting node, adjacency matrix, 
 array of node names, random seed boolean, midpoint boolean --> output average probabilities, similarity of average and 
 conditional probabilities

 6) one_to_one_array_inference: input adjacency matrix, node names list, starting components list, evidence integer, 
 hypothesis integer, parent or child indicator, farm layout, list of probabilities, number of turbines, starting turbine
 integer, effects integer --> write probabilities to Excel (noting returned)

 7) altered_one_to_one_array_inference: input adjacency matrix, node names list, parent or child indicator, farm layout array, 
 list of probabilities, number of turbines, effects integer --> write probabilities to Excel (noting returned)

 8) turbine_to_one_array_inference: input adjacency matrix, node names list, hypothesis integer, parent or child indicator,
 farm layout array, list of probabilities, number of turbines, starting turbine integer, and effects integer --> write 
 probabilities to Excel (noting returned)

 9) one_to_turbine_array_inference: input adjacency matrix, node names list, hypothesis integer, parent or child indicator,
 farm layout array, list of probabilities, number of turbines, starting turbine integer, and effects integer --> write 
 probabilities to Excel (noting returned)

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------


   turbine_array_child Documentation
 -----------------------------------
 This method inputs an adjacency matrix, array of node names (strings), a starting node, the number of turbines we
 have, and the number of the node that is the last effect in our matrix (named the "effects_mark"). We then trace
 the forward propagation of failures through and between turbines. We return the new graph.'''

def turbine_array_child(arr, nodeNames, start_arr, num_turbines, turbine_info, start_turbine, effects_mark, plot = True):
    adjacency_matrix = copy.deepcopy(arr) # Create a duplicate of the adjacency matrix tp know which failures affect other turbines
    G = nx.DiGraph() # Initialize graph

    effects = [] # Initialize effects and modes arrays for plotting purposes
    modes = []
    names_of_nodes = []

    adj = make_binary(arr, threshold=0.5).astype(int) # Create binary matrix that tells us the children of the node

    # List that tells us which nodes have already been visited
    nodeList = np.reshape(np.repeat(True, arr.shape[0] * num_turbines), (arr.shape[0], num_turbines))
    nodes = diagonal_nodes(arr)

    queue = []
    gens = {}
    owt = {}

    for start in start_arr:
        G.add_node(str(start_turbine) + ": " + str(nodeNames[start-1]))
        if start < effects_mark: effects.append(str(start_turbine) + ": " + str(nodeNames[start-1]))
        else: modes.append(str(start_turbine) + ": " + str(nodeNames[start-1]))
        nodeList[start-1][start_turbine] = False
        names_of_nodes.append(str(start_turbine) + ": " + str(nodeNames[start-1]))
        queue.append([start, 0, start_turbine])
        gens.update({str(start_turbine) + ": " + str(nodeNames[start-1]): {"layer": 0}})
        owt.update({str(start_turbine) + ": " + str(nodeNames[start-1]): {"turbine": start_turbine}})

    while len(queue) > 0: # While there are still nodes left to visit
        current = queue[0] # Get the first node in teh queue
        # print("current", current) # --> For debugging, feel free to uncomment

        children_bool = adj[current[0]-1] @ nodes # vector of zeros and child names (numerical names)
        kids = children_bool[np.nonzero(children_bool)] #list of just the child names (numerical names)
        layers = np.reshape(np.repeat(current[1] + 1, len(kids)), (len(kids), 1)) # Create a list of the layer the children are in
        turbine_nums = np.reshape(np.repeat(current[2], len(kids)), (len(kids), 1)) # Create list of which turbine the children are in
        children = np.hstack((np.reshape(kids, (len(kids),1)), layers, turbine_nums)) # Combine the children names, layer, and turbine info

        for child in children: # For each child...
            if nodeList[child[0] - 1][child[2]] == True or gens[str(child[2]) + ": " + str(nodeNames[child[0]-1])]['layer'] == child[1]: # If the child has not been visited...
                G.add_node(str(child[2]) + ": " + str(nodeNames[child[0]-1])) # Add the node and the edge from parent to child to graph
                names_of_nodes.append(str(child[2]) + ": " + str(nodeNames[child[0]-1]))

                if str(child[2]) + ": " + str(nodeNames[child[0]-1]) in names_of_nodes:
                    x = 14
                else:
                    names_of_nodes.append(str(child[2]) + ": " + str(nodeNames[child[0]-1]))

                # Mark if the child is an effect or a mode
                if child[0] < effects_mark: effects.append(str(child[2]) + ": " + str(nodeNames[child[0]-1]))
                else: modes.append(str(child[2]) + ": " + str(nodeNames[child[0]-1]))

                G.add_edge(str(current[2]) + ": " + str(nodeNames[current[0]-1]), str(child[2]) + ": " + str(nodeNames[child[0]-1]))

                queue.append(child) # Add the child to the queue
                nodeList[child[0] - 1][child[2]] = False # Change the status of the child to say we have visited it
                gens.update({str(child[2]) + ": " + str(nodeNames[child[0]-1]): {"layer": child[1]}}) # Add child to dictionary of nodes
                owt.update({str(child[2]) + ": " + str(nodeNames[child[0]-1]): {"turbine": child[2]}}) # Add child to dictionary of nodes
                
                # print("mid-check", len(gens)) # --> For debugging, feel free to uncomment
                # print("mid-check", len(G.nodes))# --> For debugging, feel free to uncomment
                # print("-------------------") # --> For debugging, feel free to uncomment

            if adjacency_matrix[current[0] - 1][child[0] - 1] > 1: # If the current node affects a child in another turbine...
                if current[0] == 3:
                    turbine_index_val = 1
                elif current[0] == 7:
                    turbine_index_val = 2
                elif current[0] == 19:
                    turbine_index_val = 3
                elif current[0] == 23:
                    turbine_index_val = 4
                elif current[0] == 40:
                    turbine_index_val = 5
                else:
                    print("Error")
                    break
                turbines_affected = turbine_info[current[2]][turbine_index_val]
                # print("collision turbines", turbines_affected)
                for i in turbines_affected:
                    # print("child", child[0]) # --> For debugging, feel free to uncomment
                    if (i >=0 and i <= 9) and nodeList[child[0] - 1][i]: # If child unvisited... any(nodeList[child[0] - 1])
                        # print("passed child", child[0]) # --> For debugging, feel free to uncomment

                        # Mark if the child is an effect or a mode
                        if child[0] < effects_mark: effects.append(str(i) + ": " + str(nodeNames[child[0]-1]))
                        else: modes.append(str(i) + ": " + str(nodeNames[child[0]-1]))

                        G.add_node(str(i) + ": " + str(nodeNames[child[0]-1])) # Add the node and the edge from parent to child to graph
                        G.add_edge(str(current[2]) + ": " + str(nodeNames[current[0]-1]), str(i) + ": " + str(nodeNames[child[0]-1]))

                        names_of_nodes.append(str(i) + ": " + str(nodeNames[child[0]-1]))
                        
                        # print("INFO", child[2]+1, np.where(nodeList[child[0] - 1] == True)[0][0]) # --> For debugging, feel free to uncomment
                        queue.append([child[0], child[1], i])
                        # print(nodeList[child[0] - 1], child[2]) # --> For debugging, feel free to uncomment
                        nodeList[child[0] - 1][i] = False # Change the status of the child to say we have visited it
                        gens.update({str(i) + ": " + nodeNames[child[0]-1]: {"layer": child[1]}}) # Add child to dictionary of layers
                        owt.update({str(i) + ": " + nodeNames[child[0]-1]: {"turbine": i}}) # Add child to dictionary of turbines and update turbine number
                        
                        # print(queue[-1]) # --> For debugging, feel free to uncomment
                        # print("2nd turbine", len(gens)) # --> For debugging, feel free to uncomment
                        # print("2nd turbine", len(G.nodes)) # --> For debugging, feel free to uncomment
                        # print("-------------------") # --> For debugging, feel free to uncomment

        queue = queue[1:] # Remove current node from queue

    # print("end", len(gens)) # --> For debugging, feel free to uncomment
    # print("end", len(G.nodes)) # --> For debugging, feel free to uncomment

    nx.set_node_attributes(G, gens) # Set layer attributes

    if plot:
        # Set of colors (for up to 10 turbines)
        effect_colors = ["#ffd6ed", "#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff", "#b1adff", "#e4adff", "#e5e5e5", "#e8d9c5"]
        mode_colors = ["#e5c0d5", "#e5a1a7", "#e5c8a7", "#e5e5a7", "#a7e5b4", "#a7cae5", "#9f9be5", "#cd9be5", "#cecece", "#d0c3b1"]
        mode_edges = ["#CCABBD", "#cc8f94", "#ccb294", "#cccc94", "#94cca0", "#94b4cc", "#8d8acc", "#b68acc", "#b7b7b7", "#b9ad9d"]

        # Plot the graph
        pos = nx.multipartite_layout(G, subset_key='layer')
        for node in G.nodes:
            # print(owt[node]["turbine"]) # --> For debugging, feel free to uncomment
            if node in effects:
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=effect_colors[owt[node]["turbine"]], node_size=750, edgecolors=mode_colors[owt[node]["turbine"]], node_shape="s")
            else:
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=effect_colors[owt[node]["turbine"]], node_size=750, edgecolors=mode_colors[owt[node]["turbine"]])
        # nx.draw_networkx_nodes(G, pos, nodelist=effects, node_color="#98c5ed", node_size=750, edgecolors="#799dbd") # --> For debugging, feel free to uncomment
        # nx.draw_networkx_nodes(G, pos, nodelist=modes, node_color="#fabc98", node_size=750, edgecolors="#c89679") # --> For debugging, feel free to uncomment
        nx.draw_networkx_labels(G, pos, font_size=5, verticalalignment='center_baseline')
        nx.draw_networkx_edges(G, pos, arrowsize=20)
        plt.box(False)
        plt.show()
    return G, nx.to_numpy_array(G), gens, effects, modes, names_of_nodes # Return the graph



'''turbine_array_parent Documentation
 -----------------------------------
 This method inputs an adjacency matrix, array of node names (strings), a starting node, the number of turbines we
 have, and the number of the node that is the last effect in our matrix (named the "effects_mark"). We then trace
 the backward propagation of failures through and between turbines. We return the new graph.'''

def turbine_array_parent(arr, nodeNames, start_arr, num_turbines, turbine_info, start_turbine, effects_mark, plot = True):
    adjacency_matrix = copy.deepcopy(arr) # Create a duplicate of the adjacency matrix tp know which failures affect other turbines
    G = nx.DiGraph() # Initialize graph

    effects = [] # Initialize effects and modes arrays for plotting purposes
    modes = []
    names_of_nodes = []

    adj = make_binary(arr).astype(int) # Array to know each node's parents
    nodeList = np.reshape(np.repeat(True, arr.shape[0] * num_turbines), (arr.shape[0], num_turbines)) # List to know where we've visited
    nodes = diagonal_nodes(adj) # Array to know each node's parents
    queue = []
    gens = {}
    owt = {}

    for start in start_arr:
        G.add_node(str(start_turbine) + ": " + str(nodeNames[start-1]))
        if start < effects_mark: effects.append(str(start_turbine) + ": " + str(nodeNames[start-1]))
        else: modes.append(str(start_turbine) + ": " + str(nodeNames[start-1]))
        nodeList[start-1][start_turbine] = False
        queue.append([start, 100, start_turbine])
        gens.update({str(start_turbine) + ": " + str(nodeNames[start-1]): {"layer": 100}})
        owt.update({str(start_turbine) + ": " + str(nodeNames[start-1]): {"turbine": start_turbine}})

    while len(queue) > 0: # While there are still nodes left to visit
        current = queue[0] # Get the current node
        # print("current", current) # --> For debugging, feel free to uncomment

        parent_bool = nodes @ adj[:, current[0]-1] # vector of zeros and parent names (numerical names)
        folks = parent_bool[np.nonzero(parent_bool)] #list of just the parent names (numerical names)
        
        layers = np.reshape(np.repeat(current[1] - 1, len(folks)), (len(folks), 1)) # List of layer the parents are in
        turbine_nums = np.reshape(np.repeat(current[2], len(folks)), (len(folks), 1)) # List of turbine the children are in
        parents = np.hstack((np.reshape(folks, (len(folks),1)), layers, turbine_nums)) # Array of parents, layers, turbines

        # print("Current", current)
        # print("Parent bool", parent_bool)
        # print("Folks", folks)
        # print("Parents", parents)
        if len(parents) > 0:
            x = 14
            # print("Parent", parents[0])
        else:
            queue = queue[1:]
            # print("no parents!", current)
            continue

        for parent in parents: # For each parent ...
            if nodeList[parent[0] - 1][parent[2]] == True: # If the parent has not been visited...
                G.add_node(str(parent[2]) + ": " + str(nodeNames[parent[0]-1])) # Add parent and its edge to graph
                G.add_edge(str(parent[2]) + ": " + str(nodeNames[parent[0]-1]), str(current[2]) + ": " + str(nodeNames[current[0]-1]))

                # Determine if parent is failure effect or failure mode
                if parent[0] < effects_mark: effects.append(str(parent[2]) + ": " + str(nodeNames[parent[0]-1]))
                else: modes.append(str(parent[2]) + ": " + str(nodeNames[parent[0]-1]))

                queue.append(parent) # Add parent node to graph
                nodeList[parent[0] - 1][parent[2]] = False # Change the status of the parent to say we have visited it
                gens.update({str(parent[2]) + ": " + str(nodeNames[parent[0]-1]): {"layer": parent[1]}}) # Add parent to dictionary of nodes
                owt.update({str(parent[2]) + ": " + str(nodeNames[parent[0]-1]): {"turbine": parent[2]}}) # Add parent to dictionary of nodes

                # print("mid-check", len(gens)) # --> For debugging, feel free to uncomment
                # print("mid-check", len(G.nodes)) # --> For debugging, feel free to uncomment
                # print("-------------------") # --> For debugging, feel free to uncomment

            if adjacency_matrix[parent[0] - 1][current[0] - 1] > 1: # If current could be cause by a failure in another turbine...
                if parent[0] == 3:
                    turbine_index_val = 1
                elif parent[0] == 7:
                    turbine_index_val = 2
                elif parent[0] == 19:
                    turbine_index_val = 3
                elif parent[0] == 23:
                    turbine_index_val = 4
                elif parent[0] == 40:
                    turbine_index_val = 5
                else:
                    print("Error current -", parent[0])
                    break
                # print("collision turbines", turbines_affected)
                for i in range(len(turbine_info)):
                    if current[2] in turbine_info[i][turbine_index_val]:
                        # print("current", current[0], "parent", parent[0]) # --> For debugging, feel free to uncomment
                        if (i >= 0 and i<=9) and nodeList[parent[0] - 1][i]: # Check if visited any(nodeList[parent[0] - 1])

                            # Determine if parent node is effect or mode
                            if parent[0] < effects_mark: effects.append(str(i) + ": " + str(nodeNames[parent[0]-1]))
                            else: modes.append(str(i) + ": " + str(nodeNames[parent[0]-1]))

                            G.add_node(str(i) + ": " + str(nodeNames[parent[0]-1])) # Add parent and its edge to graph
                            G.add_edge(str(i) + ": " + str(nodeNames[parent[0]-1]), str(current[2]) + ": " + str(nodeNames[current[0]-1]))
                            
                            # print("INFO", parent[2]+1, np.where(nodeList[parent[0] - 1] == True)[0][0]) # --> For debugging, feel free to uncomment

                            queue.append([parent[0], parent[1], i]) # Add parent to the queue

                            # print(nodeList[parent[0] - 1], parent[2]) # --> For debugging, feel free to uncomment

                            nodeList[parent[0] - 1][i] = False # Change the status of the parent to say we have visited it
                            gens.update({str(i) + ": " + nodeNames[parent[0]-1]: {"layer": parent[1]}}) # Add parent to dictionary of nodes
                            owt.update({str(i) + ": " + nodeNames[parent[0]-1]: {"turbine": i}}) # Add parent to dictionary of nodes
                            
                            # print("2nd turbine", len(gens)) # --> For debugging, feel free to uncomment
                            # print("2nd turbine", len(G.nodes)) # --> For debugging, feel free to uncomment
                            # print("-------------------") # --> For debugging, feel free to uncomment

        queue = queue[1:] # Remove current node from queue

    # print("end", len(gens)) # --> For debugging, feel free to uncomment
    # print("end", len(G.nodes)) # --> For debugging, feel free to uncomment

    nx.set_node_attributes(G, gens) # Set layer attributes
    if plot:
        # Colors for up to 10 turbines
        effect_colors = ["#ffd6ed", "#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff", "#b1adff", "#e4adff", "#e5e5e5", "#e8d9c5"]
        mode_colors = ["#e5c0d5", "#e5a1a7", "#e5c8a7", "#e5e5a7", "#a7e5b4", "#a7cae5", "#9f9be5", "#cd9be5", "#cecece", "#d0c3b1"]
        mode_edges = ["#CCABBD", "#cc8f94", "#ccb294", "#cccc94", "#94cca0", "#94b4cc", "#8d8acc", "#b68acc", "#b7b7b7", "#b9ad9d"]

        # Plot the graph
        pos = nx.multipartite_layout(G, subset_key='layer')
        for node in G.nodes:
            # print(owt[node]["turbine"]) # --> For debugging, feel free to uncomment
            if node in effects:
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=effect_colors[owt[node]["turbine"]], node_size=750, edgecolors=mode_colors[owt[node]["turbine"]], node_shape="s")
            else:
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=effect_colors[owt[node]["turbine"]], node_size=750, edgecolors=mode_colors[owt[node]["turbine"]])
        # nx.draw_networkx_nodes(G, pos, nodelist=effects, node_color="#98c5ed", node_size=750, edgecolors="#799dbd") # --> For debugging, feel free to uncomment
        # nx.draw_networkx_nodes(G, pos, nodelist=modes, node_color="#fabc98", node_size=750, edgecolors="#c89679") # --> For debugging, feel free to uncomment
        nx.draw_networkx_labels(G, pos, font_size=5, verticalalignment='center_baseline')
        nx.draw_networkx_edges(G, pos, arrowsize=20)
        plt.box(False)
        plt.show()
    return G # Return the graph



'''turbine_array_child_prob Documentation
 -----------------------------------------
 This method inputs an adjacency matrix, array of node names (strings), a starting node, the number of turbines we
 have, the number of the node that is the last effect in our matrix (named the "effects_mark"), and a boolean called update
 that determines if we update our conditional probabilities as failures happend. We then trace the forward propagation 
 of failures through and between turbines given a set of probabilities. We return the new graph.'''

def turbine_array_child_prob(arr, nodeNames, start_turbine, turbine_info, start_arr, num_turbines, effects_mark, update = False, midpoint = True, randseed = True, plot = True):
     # Set random seed and create copy of adjacency matrix so that we know which nodes cause failures in other turbines
    # if randseed:
        # random.seed(20)
    adjacency_matrix = copy.deepcopy(arr)
    return_array = np.zeros((adjacency_matrix.shape[0] * num_turbines, adjacency_matrix.shape[1]*num_turbines))
    probs = np.zeros((adjacency_matrix.shape[0]*num_turbines, 1))

    #turbine_array_child(arr, nodeNames, start, num_turbines, effects_mark) # --> For debugging, feel free to uncomment

    # Array of probabilities, as calculated from the COREWIND data
    probabilities = np.array([0.0195, 0.0195, 0.013625, 0.0055, 0.0175, 0.2075, 0.001, 0.001, 0.001, 0.093185, 0.001, 0.001,
                        0.027310938, 0.033968125, 0.033968125, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375,
                        0.0205, 0.0205, 0.02, 0.01, 0.01, 0.233, 0.288, 0.543374, 0.1285, 0.01, 0.01, 0.01, 0.015, 0.0155,
                        0.015, 0.0155, 0.015, 0.0155, 0.015, 0.33, 0.025, 0.025, 0.025, 0.025, 0.025, 0.105]) #0.01375, 
    probabilities = np.reshape(probabilities, (arr.shape[0], 1)) # Reshape these probabilities into a vector
    
    # If you want to update the probabilities, update the probabilities given the indicated starting node
    if update: 
        probabilities = conditional_probabilities_update(start, probabilities)
    
    # Compute the probabilities for all linked nodes
    transitions, tm = transition_matrix(arr, probabilities, midpoint) 
    G = nx.DiGraph()
    
    
    # Initialize arrays and add start node to graph, as described in turbine_array_child() method
    effects = []
    modes = []
    adj = make_binary(arr).astype(int)
    nodeList = np.reshape(np.repeat(True, arr.shape[0] * num_turbines), (arr.shape[0], num_turbines))
    nodes = diagonal_nodes(adj)
    queue = []
    gens = {}
    owt = {}

    for start in start_arr:
        G.add_node(str(start_turbine) + ": " + str(nodeNames[start-1]))
        nodeList[start-1][start_turbine] = False
        queue.append([start, 0, start_turbine])
        gens.update({str(start_turbine) + ": " + str(nodeNames[start-1]): {"layer": 0}})
        owt.update({str(start_turbine) + ": " + str(nodeNames[start-1]): {"turbine": start_turbine}})

    if randseed:
        random.seed(16)

    while len(queue) > 0:

        # Find child info as described in turbine_array_child() method
        current = queue[0]
        children_bool = adj[current[0]-1] @ nodes
        kids = children_bool[np.nonzero(children_bool)]
        layers = np.reshape(np.repeat(current[1] + 1, len(kids)), (len(kids), 1))
        turbine_nums = np.reshape(np.repeat(current[2], len(kids)), (len(kids), 1))
        children = np.hstack((np.reshape(kids, (len(kids),1)), layers, turbine_nums))
        turbine_index_val = current[2]

        for child in children:
            if nodeList[child[0] - 1][current[2]] == True:
                # print(transitions[current[0] - 1][child[0] - 1])
                random_num = np.random.rand()
                # print(random_num)

                # If a random number is less than the probability in transition matrix, add the node to the graph
                if  random_num < probability_over_time(transitions[current[0] - 1][child[0] - 1], current[1]+1):
                    G.add_node(str(child[2]) + ": " + str(nodeNames[child[0]-1]))
                    return_array[current[2]*arr.shape[0] + current[0] - 1][child[2]*arr.shape[0] + child[0] - 1] += 1
                    probs[child[-1]*arr.shape[0] + child[0] - 1][0] += 1

                    # If updateing is desired, update the probabilities with the addition of the child node
                    if update:
                        probabilities = conditional_probabilities_update(current[0]-1, probabilities)
                        transitions, tm = transition_matrix(arr, probabilities, midpoint)
                    
                    # Update graph as described in turbine_array_child() method
                    G.add_edge(str(current[2]) + ": " + str(nodeNames[current[0]-1]), str(child[2]) + ": " + str(nodeNames[child[0]-1]))
                    if child[0] < effects_mark: effects.append(str(child[2]) + ": " + str(nodeNames[child[0]-1]))
                    else: modes.append(str(child[2]) + ": " + str(nodeNames[child[0]-1]))

                    queue.append(child)
                    nodeList[child[0] - 1][child[2]] = False
                    gens.update({str(child[2]) + ": " + str(nodeNames[child[0]-1]): {"layer": child[1]}})
                    owt.update({str(child[2]) + ": " + str(nodeNames[child[0]-1]): {"turbine": child[2]}})

                    # print("mid-check", len(gens)) # --> For debugging, feel free to uncomment
                    # print("mid-check", len(G.nodes)) # --> For debugging, feel free to uncomment
                    # print("-------------------") # --> For debugging, feel free to uncomment

            # Find if current node affects other turbines
            if adjacency_matrix[current[0] - 1][child[0] - 1] > 1:
                num_turbines_affected = int(adjacency_matrix[current[0] - 1][child[0] - 1])
                num_turbines_affected = 2
                if current[0] == 3:
                    turbine_index_val = 1
                elif current[0] == 7:
                    turbine_index_val = 2
                elif current[0] == 19:
                    turbine_index_val = 3
                elif current[0] == 23:
                    turbine_index_val = 4
                elif current[0] == 40:
                    turbine_index_val = 5
                else:
                    print("Error")
                    break
                turbines_affected = turbine_info[current[2]][turbine_index_val]
                # print("collision turbines", turbines_affected)
                for i in turbines_affected:
                    if nodeList[child[0] - 1][i] == True:
                        # print("i", i)
                        
                        # If a random number is less than the probability in transition matrix, add the node to the graph
                        if np.random.rand() < 0.7 * probability_over_time(transitions[current[0] - 1][child[0] - 1], current[1]+1):
                            G.add_node(str(i) + ": " + str(nodeNames[child[0]-1]))
                            return_array[current[2]*arr.shape[0] + current[0] - 1][(i)*arr.shape[0] + child[0] - 1] += 1
                            probs[(i)*arr.shape[0] + child[0] - 1][0] += 1
                            
                            # If updateing is desired, update the probabilities with the addition of the child node
                            if update:
                                probabilities = conditional_probabilities_update(current[0]-1, probabilities)
                                transitions, tm = transition_matrix(arr, probabilities, midpoint)

                            # Update graph as described in turbine_array_child() method
                            if child[0] < effects_mark: effects.append(str(i) + ": " + str(nodeNames[child[0]-1]))
                            else: modes.append(str(i) + ": " + str(nodeNames[child[0]-1]))
                            
                            G.add_edge(str(current[2]) + ": " + str(nodeNames[current[0]-1]), str(i) + ": " + str(nodeNames[child[0]-1]))
                            queue.append([child[0], child[1], i])
                            nodeList[child[0] - 1][i] = False 
                            gens.update({str(i) + ": " + nodeNames[child[0]-1]: {"layer": child[1]}})
                            owt.update({str(i) + ": " + nodeNames[child[0]-1]: {"turbine": i}})
                            # print("edge:", current, child, i)
                            
                            # print(queue[-1]) # --> For debugging, feel free to uncomment
                            # print("2nd turbine", len(gens)) # --> For debugging, feel free to uncomment
                            # print("2nd turbine", len(G.nodes)) # --> For debugging, feel free to uncomment
                            # print("-------------------") # --> For debugging, feel free to uncomment

        queue = queue[1:] # Remove current node from queue

    # Plot the graph
    nx.set_node_attributes(G, gens)
    if plot:
        effect_colors = ["#ffd6ed", "#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff", "#b1adff", "#e4adff", "#e5e5e5", "#e8d9c5"]
        mode_colors = ["#e5c0d5", "#e5a1a7", "#e5c8a7", "#e5e5a7", "#a7e5b4", "#a7cae5", "#9f9be5", "#cd9be5", "#cecece", "#d0c3b1"]
        pos = nx.multipartite_layout(G, subset_key='layer')
        for node in G.nodes:
            if node in effects: nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=effect_colors[owt[node]["turbine"]], node_size=750, edgecolors=mode_colors[owt[node]["turbine"]], node_shape="s")
            else: nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=effect_colors[owt[node]["turbine"]], node_size=750, edgecolors=mode_colors[owt[node]["turbine"]])
        nx.draw_networkx_labels(G, pos, font_size=5, verticalalignment='center_baseline')
        nx.draw_networkx_edges(G, pos, arrowsize=20)
        plt.box(False)
        plt.show()

    return G, return_array, probs # Return graph



'''turbine_array_parent_prob Documentation
 -----------------------------------
 This method inputs an adjacency matrix, array of node names (strings), a starting node, the number of turbines we
 have, the number of the node that is the last effect in our matrix (named the "effects_mark"), and a boolean called update
 that determines if we update our conditional probabilities as failures happen. We then trace the backward propagation of 
 failures through and between turbines given a set of probabilities. We return the new graph.'''

def turbine_array_parent_prob(arr, nodeNames, start_arr, num_turbines, effects_mark, update = False, midpoint=True, randseed=True, plot = False):
     # Set random seed and create copy of adjacency matrix so that we know which nodes cause failures in other turbines
    adjacency_matrix = copy.deepcopy(arr)
    return_array = np.zeros((adjacency_matrix.shape[0] * num_turbines, adjacency_matrix.shape[1]*num_turbines))
    probs = np.zeros((adjacency_matrix.shape[0]*num_turbines, 1))

    #turbine_array_child(arr, nodeNames, start, num_turbines, effects_mark) # --> For debugging, feel free to uncomment

    # Array of probabilities, as calculated from the COREWIND data
    probabilities = np.array([0.0195, 0.0195, 0.013625, 0.0055, 0.0175, 0.2075, 0.001, 0.001, 0.001, 0.093185, 0.001, 0.001,
                        0.027310938, 0.033968125, 0.033968125, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375,
                        0.0205, 0.0205, 0.02, 0.01, 0.01, 0.233, 0.288, 0.543374, 0.1285, 0.01, 0.01, 0.01, 0.015, 0.0155,
                        0.015, 0.0155, 0.015, 0.0155, 0.015, 0.33, 0.025, 0.025, 0.025, 0.025, 0.025, 0.105]) #0.01375, 
    probabilities = np.reshape(probabilities, (arr.shape[0], 1)) # Reshape these probabilities into a vector
    
    # If you want to update the probabilities, update the probabilities given the indicated starting node
    if update: 
        probabilities = conditional_probabilities_update(start, probabilities)
    
    # Compute the probabilities for all linked nodes
    transitions, tm = transition_matrix(arr.T, probabilities, midpoint) 
    
    # Initialize arrays and starting node as in turbine_array_parent() method
    G = nx.DiGraph()
    effects = []
    modes = []
    adj = make_binary(arr).astype(int)
    nodeList = np.reshape(np.repeat(True, arr.shape[0] * num_turbines), (arr.shape[0], num_turbines))
    nodes = diagonal_nodes(adj)
    queue = []
    gens = {}
    owt = {}

    for start in start_arr:
        G.add_node(str(0) + ": " + str(nodeNames[start-1])) 
        nodeList[start-1][0] = False
        queue. append([start, 100, 0])
        gens.update({str(0) + ": " + str(nodeNames[start-1]): {"layer": 100}})
        owt.update({str(0) + ": " + str(nodeNames[start-1]): {"turbine": 0}})

    if randseed:
        random.seed(16)

    while len(queue) > 0:

        # Find child info as described in turbine_array_parent() method
        current = queue[0]
        parent_bool = nodes @ adj[:, current[0]-1]
        kids = parent_bool[np.nonzero(parent_bool)]
        layers = np.reshape(np.repeat(current[1] - 1, len(kids)), (len(kids), 1))
        turbine_nums = np.reshape(np.repeat(current[2], len(kids)), (len(kids), 1))
        parent = np.hstack((np.reshape(kids, (len(kids),1)), layers, turbine_nums))

        for folk in parent:
            if nodeList[folk[0] - 1][current[2]] == True:
                # print("transitions", transitions[current[0] - 1][folk[0] - 1])
                random_num = np.random.rand()
                # print("random", random_num)

                # If a random number is less than the probability in transition matrix, add the node to the graph
                if random_num < transitions[current[0] - 1][folk[0] - 1]: #probability_over_time(transitions[current[0] - 1][folk[0] - 1], current[1]+1):
                    G.add_node(str(folk[2]) + ": " + str(nodeNames[folk[0]-1]))
                    probs[folk[-1]*arr.shape[0] + folk[0] - 1][0] += 1

                    # If updateing is desired, update the probabilities with the addition of the folk node
                    if update:
                        probabilities = conditional_probabilities_update(current[0]-1, probabilities)
                        transitions, tm = transition_matrix(arr.T, probabilities)

                    # Update graph as described in turbine_array_folk() method
                    G.add_edge(str(folk[2]) + ": " + str(nodeNames[folk[0]-1]), str(current[2]) + ": " + str(nodeNames[current[0]-1]))
                    if folk[0] < effects_mark: effects.append(str(folk[2]) + ": " + str(nodeNames[folk[0]-1]))
                    else: modes.append(str(folk[2]) + ": " + str(nodeNames[folk[0]-1]))
                    queue.append(folk)
                    nodeList[folk[0] - 1][folk[2]] = False
                    gens.update({str(folk[2]) + ": " + str(nodeNames[folk[0]-1]): {"layer": folk[1]}})
                    owt.update({str(folk[2]) + ": " + str(nodeNames[folk[0]-1]): {"turbine": folk[2]}})


            if adjacency_matrix[folk[0] - 1][current[0] - 1] > 1:
                if any(nodeList[folk[0] - 1]) and folk[2]+1 <= num_turbines - 1: #np.where(nodeList[folk[0] - 1] == True)[0][-1]:
                    # If a random number is less than the probability in transition matrix, add the node to the graph
                    if np.random.rand() < transitions[current[0] - 1][folk[0] - 1]: # If the random value is less than the probability...
                        if folk[0] < effects_mark: effects.append(str(folk[2] + 1) + ": " + str(nodeNames[folk[0]-1]))
                        else: modes.append(str(folk[2] + 1) + ": " + str(nodeNames[folk[0]-1]))

                        G.add_node(str(folk[2] + 1) + ": " + str(nodeNames[folk[0]-1]))
                        probs[(folk[-1]+1)*arr.shape[0] + folk[0] - 1][0] += 1

                        # If updateing is desired, update the probabilities with the addition of the folk node
                        if update:
                            probabilities = conditional_probabilities_update(current[0]-1, probabilities)
                            transitions, tm = transition_matrix(arr.T, probabilities)

                        # Update graph as described in turbine_array_folk() method
                        G.add_edge(str(folk[2] + 1) + ": " + str(nodeNames[folk[0]-1]), str(current[2]) + ": " + str(nodeNames[current[0]-1]))
                        queue.append([folk[0], folk[1], folk[2] + 1])
                        nodeList[folk[0] - 1][folk[2] + 1] = False
                        gens.update({str(folk[2]+1) + ": " + nodeNames[folk[0]-1]: {"layer": folk[1]}})
                        owt.update({str(folk[2]+1) + ": " + nodeNames[folk[0]-1]: {"turbine": folk[2] + 1}})

        queue = queue[1:] # Remove current node from queue

    # Plot the graph
    nx.set_node_attributes(G, gens)
    effect_colors = ["#ffd6ed", "#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff", "#b1adff", "#e4adff", "#e5e5e5", "#e8d9c5"]
    mode_colors = ["#e5c0d5", "#e5a1a7", "#e5c8a7", "#e5e5a7", "#a7e5b4", "#a7cae5", "#9f9be5", "#cd9be5", "#cecece", "#d0c3b1"]
    pos = nx.multipartite_layout(G, subset_key='layer')
    for node in G.nodes:
        if node in effects:
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=effect_colors[owt[node]["turbine"]], node_size=750, edgecolors=mode_colors[owt[node]["turbine"]], node_shape="s")
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=effect_colors[owt[node]["turbine"]], node_size=750, edgecolors=mode_colors[owt[node]["turbine"]])
    nx.draw_networkx_labels(G, pos, font_size=5, verticalalignment='center_baseline')
    nx.draw_networkx_edges(G, pos, arrowsize=20)
    plt.box(False)
    plt.show()

    return G, return_array, probs # Return the graph



'''monte_carlo_sim_array Documentation
 -----------------------------------
 This method inputs number of iterations, number of turbines, boolean for plotting, starting node, an adjacency matrix, 
 array of node names (strings), boolean for using random seed, and boolean for using midpoint calculation. We then generate
 a graph with failure probabilities for the number of iterations and find the average graph and the probability that each node
 is in the graph (the number of times the node shows up divided by the number of iterations). Lastly, we calculate the similarity
 between the probability of the nodes in the graph (that we just calculated) compared to the estimated probability calculated via
 conditional probabilities. We return the first list of probabilities and cosine similarity between the two lists of probabilities.'''

def monte_carlo_sim_array(num_iterations, num_turbines, plot, start, adjacency_matrix, nodeNames, rand_seed, mid_point):
    t = time.process_time() # For calculating how long the simulations took
    adj_matrices = np.zeros((adjacency_matrix.shape[0]*num_turbines, adjacency_matrix.shape[1]*num_turbines)) # Initialize adjacency matrix for average graph
    probs = np.zeros((adjacency_matrix.shape[0] * num_turbines, 1)) # Iniitialize array for average probabilities

    # Initialize individual event probabilities
    probabilities = np.array([0.0195, 0.0195, 0.013625, 0.0055, 0.0175, 0.2075, 0.001, 0.001, 0.001, 0.093185, 0.001, 0.001,
                                0.027310938, 0.033968125, 0.033968125, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375,
                                0.0205, 0.0205, 0.02, 0.01, 0.01, 0.233, 0.288, 0.543374, 0.1285, 0.01, 0.01, 0.01, 0.015, 0.0155,
                                0.015, 0.0155, 0.015, 0.0155, 0.015, 0.33, 0.025, 0.025, 0.025, 0.025, 0.025, 0.105]) #0.01375, 
    probabilities = np.reshape(probabilities, (adjacency_matrix.shape[0], 1))

    # Run simulations
    for i in range(num_iterations):
        arr = copy.deepcopy(adjacency_matrix)
        G, adj_mat, prob = turbine_array_child_prob(arr, nodeNames, start, num_turbines, 27, update = False, midpoint=mid_point, randseed=rand_seed, plot=False)
        adj_matrices += adj_mat.astype(float64) # Update adjacency matrix
        probs += prob # Update probabilities
        print(i+1) # Print progress

    # Calculate average graph and average probabilities
    adj_matrices = adj_matrices/num_iterations
    probs = probs/num_iterations

    # Calculate conditional probabilities
    num_turbine_probs = np.tile(probabilities, (10,1))
    v1 = conditional_probabilities_update(start, num_turbine_probs)
    v2 = probs
    
    if plot: # Plot the average graph
        nodeNamesArray = []
        for k in range(num_turbines): # Update the node names to accomodate for the number of turbines
            for node in nodeNames:
                nodeNamesArray.append(str(k) + ": " + node)
        nodeNames = np.array(nodeNamesArray)
        draw_bfs_multipartite(adj_matrices, nodeNamesArray, start, "multi-child", multi_turbine=True)

    elapsed_time = time.process_time() - t# For calculating how long the simulations took
    print("elapsed time:", elapsed_time)# For calculating how long the simulations took

    return v2, cosine_similarity(v1, v2) # Return the average probabilities and the similarity of the average and conditional probabilities



'''one_to_one_array_inference Documentation
 -------------------------------------------
 This method inputs an adjacency matrix, list of node names, list of starting components, integer of turbine we want evidence from,
 integer of turbine we want hypothesis from, string indicating parent or child, array of farm layout, list of probabilities, integer
 number of turbines, integer starting turbine (for tree), and integer showing where the effects end. We then use the failureProbabilities.py
 methods to calculate Bayesian inference for one node in evidence and one node in hypothesis. We write a document with these probabilities,
 but nothing is returned.'''

def one_to_one_array_inference(arr, nodeNames, start_components, evidence_num, hypothesis_num, parent_or_child, array_layout, probabilities, num_turbines, start_turbine, effects_mark):
    # Generate tree for Bayesian network
    print("Generating tree...")
    if parent_or_child == "parent":
        G = turbine_array_parent(arr, nodeNames, start_components, num_turbines, array_layout, start_turbine, effects_mark, plot = False)
    elif parent_or_child == "child":
        G, adj_array, gens, effects, modes, names_of_nodes = turbine_array_child(arr, nodeNames, start_components, num_turbines, array_layout, start_turbine, effects_mark, plot = False)
    else:
        print("ERROR! -- not child or parent")
        return
    
    # Create array of names of nodes and array of probabilities
    nodeNamesArray = np.array([])
    new_probabilities = []
    for node in G.nodes:
        name = node[3:]
        index = np.where(nodeNames == name)
        new_probabilities.append(probabilities[index])
        nodeNamesArray = np.append(nodeNamesArray, node)
    nodeNamesArray = np.reshape(nodeNamesArray, (len(nodeNamesArray),))
    new_probabilities = np.array(new_probabilities)
    new_probabilities = np.reshape(new_probabilities, (len(new_probabilities), 1))

    # Create list of nodes that we want in the evidence and hypothesis
    evidence = []
    hypothesis = []
    for node in range(len(nodeNamesArray)):
        if nodeNamesArray[node][0] == str(hypothesis_num):
            hypothesis.append(node)
        if nodeNamesArray[node][0] == str(evidence_num):
            evidence.append(node)

    # Create adjacency matrix
    this_array = nx.to_numpy_array(G)

    # Make file name
    filename = "turbineInference_" + str(start_turbine) + "_" + str(evidence_num) + "_" + str(hypothesis_num)+ "_" + parent_or_child + ".xlsx"

    print("Calculating probabilities...")
    with pd.ExcelWriter(filename) as writer:
        # Initialize array of all (summed) probabilities, and array of probabilities for just one evidence node
        all_probabilities = np.zeros((len(evidence),3))
        array_of_probabilities = np.zeros((len(evidence), len(hypothesis)))
        array_of_probabilities2 = np.zeros((len(evidence), len(hypothesis)))

        for first_node in evidence:
            index1 = np.where(np.array(evidence) == first_node)[0][0] # Index of evidence node
            summed_probabilities1 =np.array([0.0, 0.0])
            summed_probabilities2 =np.array([0.0, 0.0])

            # Calculate Bayesian inference and add to sum of probabilities for each node in hypothesis turbine
            for this_node in hypothesis:
                A, B = backward_bayesian_inference(this_array, nodeNamesArray, [0], [first_node], [this_node], new_probabilities, start_bool = True, multi = True, poc = parent_or_child)
                summed_probabilities1 += A
                summed_probabilities2 += B
                print(this_node, A)
                index2 = np.where(np.array(hypothesis) == this_node)[0][0] # Index of hypothesis node
                array_of_probabilities[index1][index2] = B[0]
                array_of_probabilities2[index1][index2] = B[1]

            # Print sum of probabilities
            print("------------", nodeNamesArray[first_node].replace("\n", " "), "------------")
            print("Summed Probabilities 1", summed_probabilities1, np.array(summed_probabilities1)/(summed_probabilities1[0]+summed_probabilities1[1]))
            print("Summed Probabilities 2", summed_probabilities2, np.array(summed_probabilities2)/(summed_probabilities2[0]+summed_probabilities2[1]))
            
            # Add summed probabilities to array
            all_probabilities[index1][0] = first_node
            all_probabilities[index1][1] = summed_probabilities1[0]
            all_probabilities[index1][2] = summed_probabilities1[1]

        # Write to excel file
        df = pd.DataFrame(array_of_probabilities, columns = hypothesis)
        df.to_excel(writer, sheet_name=str(first_node)+"."+str(this_node)+".0")
        df2 = pd.DataFrame(array_of_probabilities2, columns = hypothesis)
        df2.to_excel(writer, sheet_name=str(first_node)+"."+str(this_node)+".1")
        df3 = pd.DataFrame(all_probabilities)
        df3.to_excel(writer, sheet_name="allProbs")
    return# Nothing returned



'''altered_one_to_one_array_inference Documentation
 ---------------------------------------------------
 This method inputs an adjacency matrix, list of node names, string indicating parent or child, array of farm layout, list of probabilities, 
 number of turbines, and integer showing where the effects end. We then use the failureProbabilities.py methods to calculate Bayesian 
 inference for one node in evidence and one node in hypothesis. The difference from the previous method is that our starting tree starts 
 from a different node (so that its the same as the evidence node). We write a document with these probabilities, but nothing is returned.'''

def altered_one_to_one_array_inference(arr, nodeNames, parent_or_child, array_layout, probabilities, num_turbines, effects_mark):
    # Create file name
    filename = "turbineInference_" + "_" + parent_or_child + "blah.xlsx"

    with pd.ExcelWriter(filename) as writer:
        all_probabilities = np.zeros((arr.shape[0]*num_turbines,arr.shape[1]*num_turbines)) # Array of all probabilities
        for start_turbine in range(num_turbines):
            for start_component in range(1,arr.shape[0]+1):

                # Generate tree for Bayesian Network
                print("Generating tree...")
                adj_mat = arr.copy()
                if parent_or_child == "parent":
                    G = turbine_array_parent(adj_mat, nodeNames, [start_component], num_turbines, array_layout, start_turbine, effects_mark, plot = False)
                elif parent_or_child == "child":
                    G, adj_array, gens, effects, modes, names_of_nodes = turbine_array_child(adj_mat, nodeNames, [start_component], num_turbines, array_layout, start_turbine, effects_mark, plot = False)
                else:
                    print("ERROR! -- not child or parent")
                    return
                
                # Create array of node names and array of probabilities
                nodeNamesArray = np.array([])
                new_probabilities = []
                for node in G.nodes:
                    name = node[3:]
                    index = np.where(nodeNames == name)
                    new_probabilities.append(probabilities[index])
                    nodeNamesArray = np.append(nodeNamesArray, node)
                nodeNamesArray = np.reshape(nodeNamesArray, (len(nodeNamesArray),))
                new_probabilities = np.array(new_probabilities)
                new_probabilities = np.reshape(new_probabilities, (len(new_probabilities), 1))

                # Iterate through each turbine
                print("------------", nodeNamesArray[0].replace("\n", " "), "------------")
                for hypothesis_num in range(num_turbines):

                    # Create array of nodes that we want as our hypothesis nodes
                    hypothesis = []
                    for node in range(len(nodeNamesArray)):
                        if nodeNamesArray[node][0] == str(hypothesis_num):
                            hypothesis.append(node)

                    # Create adjacency matrix
                    this_array = nx.to_numpy_array(G)

                    # Calculate Bayesian inference and add to array of probabilities
                    for this_node in hypothesis:
                        A, B = backward_bayesian_inference(this_array, nodeNamesArray, [0], [0], [this_node+1], new_probabilities, start_bool = True, multi = True, poc = parent_or_child)
                        index2 = np.where(nodeNames == nodeNamesArray[this_node][3:])[0][0]
                        all_probabilities[start_turbine * arr.shape[0] + start_component - 1][hypothesis_num * arr.shape[0] + index2] = A[0]
        
        # Write array to Excel file
        df3 = pd.DataFrame(all_probabilities)
        df3.to_excel(writer, sheet_name="allProbs")
    return # Nothing returned



'''turbine_to_one_array_inference Documentation
 ---------------------------------------------------
 This method inputs an adjacency matrix, list of node names, integer of turbine we want hypothesis from, string indicating parent or child,
 array of farm layout, list of probabilities, number of turbines, integer starting turbine (for tree), and integer showing where the effects end.
 We then use the failureProbabilities.py methods to calculate Bayesian inference for entire turbine of nodes in evidence and one node in hypothesis.
 We write a document with these probabilities, but nothing is returned.'''

def turbine_to_one_array_inference(arr, nodeNames, hypothesis_num, parent_or_child, array_layout, probabilities, num_turbines, start_turbine, effects_mark):
    filename = "turbine21Inference_" + str(start_turbine) + str(hypothesis_num) + "_" + parent_or_child + ".xlsx"
    with pd.ExcelWriter(filename) as writer:

        # Initialize arrays of probabilities and array of evidence (for tree generation)
        all_probabilities = np.zeros((num_turbines, 4))
        array_of_probabilities = np.zeros((num_turbines, arr.shape[0]))
        array_of_probabilities2 = np.zeros((num_turbines, arr.shape[0]))
        evidence = [i for i in range(arr.shape[0])]

        # Create forward propagation tree
        print("Generating tree...")
        adj_mat = arr.copy()
        if parent_or_child == "parent":
            G = turbine_array_parent(adj_mat, nodeNames, evidence, num_turbines, array_layout, start_turbine, effects_mark, plot = False)
        elif parent_or_child == "child":
            G, adj_array, gens, effects, modes, names_of_nodes = turbine_array_child(adj_mat, nodeNames, evidence, num_turbines, array_layout, start_turbine, effects_mark, plot = False)
        else:
            print("ERROR! -- not child or parent")
            return

        # Create array of node names and array of probabilities
        nodeNamesArray = np.array([])
        new_probabilities = []
        for node in G.nodes:
            name = node[3:]
            index = np.where(nodeNames == name)
            new_probabilities.append(probabilities[index])
            nodeNamesArray = np.append(nodeNamesArray, node)
        nodeNamesArray = np.reshape(nodeNamesArray, (len(nodeNamesArray),))
        new_probabilities = np.array(new_probabilities)
        new_probabilities = np.reshape(new_probabilities, (len(new_probabilities), 1))

        # Create lists of nodes for hypothesis and evidence
        hypothesis = []
        evidence = []
        for node in range(len(nodeNamesArray)):
            if nodeNamesArray[node][0] == str(hypothesis_num):
                hypothesis.append(node)
            if nodeNamesArray[node][0] == str(start_turbine):
                evidence.append(node)
        this_array = nx.to_numpy_array(G)
        
        # Initialize sum of probabilities
        print("Calculating probabilities...")
        summed_probabilities1 =np.array([0.0, 0.0])
        summed_probabilities2 =np.array([0.0, 0.0])

        # Calculate Bayesian inference, add to sum of probabilities, and add to array of probabilities
        for this_node in hypothesis:
            this_array2 = this_array.copy()
            new_probabilities2 = new_probabilities.copy()
            A, B = backward_bayesian_inference(this_array2, nodeNamesArray, [0], evidence, [this_node], new_probabilities2, start_bool = True, multi = True, poc = parent_or_child)
            summed_probabilities1 += A
            summed_probabilities2 += B
            index2 = np.where(nodeNames == nodeNamesArray[this_node][3:])[0][0]
            array_of_probabilities[hypothesis_num][index2] = B[0]
            array_of_probabilities2[hypothesis_num][index2] = B[1]

        # Print summed probabilities
        print("------------", hypothesis_num, "------------")
        if len(hypothesis) < 1:
            print("Summed Probabilities 1", summed_probabilities1)
            print("Summed Probabilities 2", summed_probabilities2)
        else:
            print("Summed Probabilities 1", summed_probabilities1, np.array(summed_probabilities1)/(summed_probabilities1[0]+summed_probabilities1[1]))
            print("Summed Probabilities 2", summed_probabilities2, np.array(summed_probabilities2)/(summed_probabilities2[0]+summed_probabilities2[1]))
        
        # Add summed probabilities to array of probabilities
        all_probabilities[hypothesis_num][0] = summed_probabilities1[0]/(len(hypothesis))
        all_probabilities[hypothesis_num][1] = summed_probabilities1[1]/(len(hypothesis))
        all_probabilities[hypothesis_num][2] = summed_probabilities2[0]/(summed_probabilities2[0]+summed_probabilities2[1])
        all_probabilities[hypothesis_num][3] = summed_probabilities2[1]/(summed_probabilities2[0]+summed_probabilities2[1])

        # Write to Excel file
        df = pd.DataFrame(array_of_probabilities)
        df.to_excel(writer, sheet_name="Probability of Failure")
        df2 = pd.DataFrame(array_of_probabilities2)
        df2.to_excel(writer, sheet_name="Probability of Success")
        df3 = pd.DataFrame(all_probabilities)
        df3.to_excel(writer, sheet_name="allProbs")
    return



'''one_to_turbine_array_inference Documentation
 -----------------------------------------------
 This method inputs an adjacency matrix, list of node names, integer of turbine we want hypothesis from, string indicating parent or child,
 array of farm layout, list of probabilities, number of turbines, integer starting turbine (for tree), and integer showing where the effects end.
 We then use the failureProbabilities.py methods to calculate Bayesian inference for one nodes in evidence and entire turbine of node in hypothesis.
 We write a document with these probabilities, but nothing is returned.'''

def one_to_turbine_array_inference(arr, nodeNames, hypothesis_num, parent_or_child, array_layout, probabilities, num_turbines, start_turbine, effects_mark):
    filename = "12turbineInference_" + str(start_turbine) + str(hypothesis_num) + "_" + parent_or_child + ".xlsx"
    with pd.ExcelWriter(filename) as writer:

        # Initialize arrays of probabilities and array of evidence (for tree generation)
        all_probabilities = np.zeros((num_turbines, 4))
        array_of_probabilities = np.zeros((num_turbines, arr.shape[0]))
        array_of_probabilities2 = np.zeros((num_turbines, arr.shape[0]))
        evidence = [i for i in range(arr.shape[0])]

        # Create forward propagation tree
        print("Generating tree...")
        adj_mat = arr.copy()
        if parent_or_child == "parent":
            G = turbine_array_parent(adj_mat, nodeNames, evidence, num_turbines, array_layout, start_turbine, effects_mark, plot = False)
        elif parent_or_child == "child":
            G, adj_array, gens, effects, modes, names_of_nodes = turbine_array_child(adj_mat, nodeNames, evidence, num_turbines, array_layout, start_turbine, effects_mark, plot = False)
        else:
            print("ERROR! -- not child or parent")
            return
        this_array = nx.to_numpy_array(G)
        this_array2 = this_array.copy()
        
        # Create array of node names and array of probabilities
        nodeNamesArray = np.array([])
        new_probabilities = []
        for node in G.nodes:
            name = node[3:]
            index = np.where(nodeNames == name)
            new_probabilities.append(probabilities[index])
            nodeNamesArray = np.append(nodeNamesArray, node)
        nodeNamesArray = np.reshape(nodeNamesArray, (len(nodeNamesArray),))
        new_probabilities = np.array(new_probabilities)
        new_probabilities = np.reshape(new_probabilities, (len(new_probabilities), 1))
        new_probabilities2 = new_probabilities.copy()

        # Create lists of nodes for hypothesis and evidence
        hypothesis = []
        evidence = []
        for node in range(len(nodeNamesArray)):
            if nodeNamesArray[node][0] == str(hypothesis_num):
                hypothesis.append(node)
            if nodeNamesArray[node][0] == str(start_turbine):
                evidence.append(node)

        # Initialize sum of probabilities
        print("Calculating probabilities...")
        summed_probabilities1 =np.array([0.0, 0.0])
        summed_probabilities2 =np.array([0.0, 0.0])
        
        # Calculate Bayesian inference, add to sum of probabilities, and add to array of probabilities
        A, B = backward_bayesian_inference(this_array2, nodeNamesArray, [0], [0], hypothesis, new_probabilities2, start_bool = True, multi = True, poc = parent_or_child)
        summed_probabilities1 += A
        summed_probabilities2 += B
        array_of_probabilities[hypothesis_num][0] = B[0]
        array_of_probabilities2[hypothesis_num][0] = B[1]

        # Print summed probabilities
        print("------------", hypothesis_num, "------------")
        if len(hypothesis) < 1:
            print("Summed Probabilities 1", summed_probabilities1)
            print("Summed Probabilities 2", summed_probabilities2)
        else:
            print("Summed Probabilities 1", summed_probabilities1, np.array(summed_probabilities1)/(summed_probabilities1[0]+summed_probabilities1[1]))
            print("Summed Probabilities 2", summed_probabilities2, np.array(summed_probabilities2)/(summed_probabilities2[0]+summed_probabilities2[1]))
        
        # Add summed probabilities to array of probabilities
        all_probabilities[hypothesis_num][0] = summed_probabilities1[0]/(len(hypothesis))
        all_probabilities[hypothesis_num][1] = summed_probabilities1[1]/(len(hypothesis))
        all_probabilities[hypothesis_num][2] = summed_probabilities2[0]/(summed_probabilities2[0]+summed_probabilities2[1])
        all_probabilities[hypothesis_num][3] = summed_probabilities2[1]/(summed_probabilities2[0]+summed_probabilities2[1])

        # Write to Excel file
        df = pd.DataFrame(array_of_probabilities)
        df.to_excel(writer, sheet_name="Probability of Failure")
        df2 = pd.DataFrame(array_of_probabilities2)
        df2.to_excel(writer, sheet_name="Probability of Success")
        df3 = pd.DataFrame(all_probabilities)
        df3.to_excel(writer, sheet_name="allProbs")
    return



'''# Run for Inference Calculations

arr, nodeNames = excel2Matrix("ExcelFiles/failureData.xlsx", "bigMatrix")
start_components = [0]
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
evidence_num = 0
hypothesis_num = 8
parent_or_child = "child"
start_turbine = evidence_num
adj_mat = arr.copy()
evidence = [i for i in range(arr.shape[0])]

altered_one_to_one_array_inference(arr, nodeNames, hypothesis_num, parent_or_child, array_layout, probabilities, num_turbines, start_turbine, effects_mark)'''