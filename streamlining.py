import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graphBuilder import *
from spectralAnalysis import *
from searchMethods import *
from failureProbabilities import *

''' streamlining.py -----------------------------------------------------------------------------------------------

            ****************************** Last Updated: 18 April 2024 ******************************

 Methods:
 1) get_position: input graph, string --> output dictionary of node positions

 2) plot_graph_probabilities: input graph, string, effects_mark (integer), list of node names, list of probabilities,
 threshold (automatically 0.01), inclusion boolean --> plot graph (no return)

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------


   get_position Documentation
 -----------------------------------
 This method inputs a Networkx graph and a string indicating the type of position we want in our plot. We then create
 a dictionary with the position of the nodes. We return this dictionary.'''

def get_position(G, type):
    # This type attempts to have all the edges the same length
    if type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)  # positions for all nodes
    
    # This type plots the graph in a random configuration
    elif type == "random":
        pos = nx.random_layout(G)  # positions for all nodes
    # This type treats nodes as repelling objects and edges as springs (causing them to moving in simulation)
    elif type == "spring":
        pos = nx.spring_layout(G)  # positions for all nodes
    else:
        print("Error!") # Print that an error has occurred
        return
    return pos # Return the position dictionary



'''plot_graph_probabilities Documentation
 -----------------------------------------
 This method inputs a graph, string indicating the type of layout we want, effects_mark (integer), list of node names, 
 list of probabilities, threshold for including/excluding nodes (automatically 0.01), and inclusion boolean that keeps
 improbable nodes when True and excludes them when False. We then find the nodes with probabilities that 
 are below the threshold and plot the graph. Nothing is returned.'''

def plot_graph_probabilities(G, type, effects_mark, nodeNames, probabilities, threshold = 0.01, include = True):

    # Find the nodes below the threshold probability
    light_effects = nodeNames[np.where(probabilities[:effects_mark] < threshold)[0]]
    light_nodes = np.where(probabilities < threshold)[0]
    light_modes = nodeNames[light_nodes[np.where(light_nodes >= effects_mark)[0]]]

    # Get position dictionary
    pos = get_position(G, type)

    # Draw nodes that are below threshold with lighter color if include=True
    if include:
        nx.draw_networkx_nodes(G, pos, nodelist=nodeNames[:effects_mark], node_color="#57a0e2")
        nx.draw_networkx_nodes(G, pos, nodelist=nodeNames[effects_mark:], node_color="#f89c67")
        nx.draw_networkx_nodes(G, pos, nodelist=light_effects, node_color="#d5e7f7")
        nx.draw_networkx_nodes(G, pos, nodelist=light_modes, node_color="#fde4d5")
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos)

    # Otherwise, do not draw nodes below threshold probability
    else:
        new_effects = nodeNames[:effects_mark]
        new_modes = nodeNames[effects_mark:]
        for node in nodeNames:
            if node in nodeNames[light_nodes]:
                G.remove_node(node)
                if node in new_effects:
                    new_effects = np.delete(new_effects, np.where(new_effects == node)[0][0])
                if node in new_modes:
                    new_modes = np.delete(new_modes, np.where(new_effects == node)[0][0])
        pos = get_position(G, type)
        nx.draw_networkx_nodes(G, pos, nodelist=new_effects, node_color="#57a0e2")
        nx.draw_networkx_nodes(G, pos, nodelist=new_modes, node_color="#f89c67")
    
    # Plot rest of the graph
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.box(False)
    plt.show()
    return # Nothing returned
    

'''
Uncomment the code below to plot the graph while distinguishing which pairs of nodes have higher transitional probabilities.

arr, nodeNames = excel2Matrix("ExcelFiles/failureData.xlsx", "bigMatrix")
G, arr = matrix2Graph(arr, nodeNames, nodeLabels=True)
probabilities = np.array([0.0195, 0.0195, 0.013625, 0.0055, 0.0175, 0.2075, 0.001, 0.001, 0.001, 0.093185, 0.001, 0.001,
                                        0.027310938, 0.033968125, 0.033968125, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375,
                                        0.0205, 0.0205, 0.02, 0.01, 0.01, 0.233, 0.288, 0.543374, 0.1285, 0.01, 0.01, 0.01, 0.015, 0.0155,
                                        0.015, 0.0155, 0.015, 0.0155, 0.015, 0.33, 0.025, 0.025, 0.025, 0.025, 0.025, 0.105]) #0.01375, 
threshold = 0.01
plot_graph_probabilities(G, "random", 26, nodeNames, probabilities, threshold=0.01, include=True)'''