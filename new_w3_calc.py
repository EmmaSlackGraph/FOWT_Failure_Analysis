import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from node import FMEAMode
from graphBuilder import *


class circularBayNet():
    def __init__(self):
        self.G = nx.DiGraph()
        self.turbines = []
        self.causes = []
        self.effects = []
        self.modes = []
        self.systems = []
        self.fowts = []
        self.farm = []

    def create_circular_graph(self, num_turbines, prob_type, calc_multi = False):
        # Get FMECA Table
        fileName = "new_w3_bay_net.xlsx"
        df = pd.read_excel(fileName)
        df = pd.DataFrame(df)

        # Set up array layout based on number of turbines
        if num_turbines == 1:
            array_layout = [[0, [0], [0], [0], [0], [0]]]
        elif num_turbines == 10:
            array_layout = [[0, [1, 2], [1], [1], [1], [2]], 
                    [1, [0, 2, 3], [0, 2], [0, 2], [2], [3]],
                    [2, [0,1,3,4], [1, 3], [1, 3], [3], [0, 4]], 
                    [3, [1,2,4], [2, 4], [2, 4], [4], [1]],
                    [4, [2,3,], [3], [3], [3], [2]],
                    [5, [3,4,6,7], [4, 6], [4, 6], [6], [3, 7]],
                    [6, [4,5,7,8], [5, 7], [5, 7], [7], [4, 8]],
                    [7, [5,6,8,9], [6, 8], [6, 8], [8], [5, 9]],
                    [8, [6, 7, 9], [7, 9], [7, 9], [9], [6]],
                    [9, [7,8], [8], [8], [8], [7]]]

        # Initialize lists
        names = []
        nodes = []

        # Repeat fpllowing process for total number of turbines
        for turbine in range(num_turbines):
            turbine_arr = []

            # Read each row of combined FMECA table
            for row in range(len(df)):

                # Get the failure cause, mode, effect, system, occurrence, and RPN from the row
                name = str(turbine) + ": "+ str(df.at[row, 'Failure Mode2'])
                effect = str(turbine) + ": "+ str(df.at[row, 'Potential End Effects'])
                cause = str(df.at[row, 'Failure Cause'])
                system = str(turbine) + ": "+ str(df.at[row, 'System'])
                occurrence = float(df.at[row, 'Occurrence'])
                rpn = float(df.at[row, 'RPN'])

                # Determine if we are using occurrence or RPN and append value to node attribute
                if 'o' in prob_type: probability = occurrence
                elif 'r' in prob_type: probability = rpn/100

                # Determine if failure mode can impact multiple turbines
                multi = False
                if df.at[row, 'Mult-Turbine'] >= 0.5:
                    multi = True
                
                # Add nodes for failure cause, mode, effect, system, turbine, and farm
                self.add_another_node(name, 1, probability)
                self.add_another_node(effect, 2, probability)
                self.add_another_node(cause, 0, probability)
                self.add_another_node(system, 3, probability)
                self.add_another_node(str(turbine)+": "+"FOWT Failure", 4, probability)
                self.add_another_node("Farm Failure", 5, probability)

                # Append failure mode, effect, and system to list of failures in turbine (for plotting)
                turbine_arr.append(name)
                turbine_arr.append(effect)
                turbine_arr.append(system)

                # Append each failure to associated list of failure type (for plotting)
                self.effects.append(effect)
                self.modes.append(name)
                self.causes.append(cause)
                self.systems.append(system)
                self.fowts.append(str(turbine)+": "+"FOWT Failure")
                self.farm.append("Farm Failure")

                # Add edges between newly created nodes
                self.G.add_edge(cause, name)
                self.G.add_edge(name, effect)
                self.G.add_edge(effect, system)
                self.G.add_edge(system, str(turbine)+": "+"FOWT Failure")
                self.G.add_edge(str(turbine)+": "+"FOWT Failure", "Farm Failure")

                # Create additional edges for those nodes that could impact another turbine
                if multi and calc_multi:
                    for second_turbine in array_layout[turbine][1]:
                        second_effect = str(second_turbine) + ": "+ str(df.at[row, 'Potential End Effects'])
                        self.add_another_node(second_effect, 2, probability)
                        self.G.add_edge(name, second_effect)

                # Append failure mode to list if not already there
                if name not in names:
                    node = FMEAMode(len(names), name, [effect], [cause], multi, system)
                    names.append(name)
                    nodes.append(node)
                    continue
            
            # Append the list of turbine failures to master list of all failures
            self.turbines.append(turbine_arr)

        # Calculate each node's probability and add to node attribute
        for node in list(self.G.nodes):
            instances = self.G.nodes[node]['instance']
            sum_probs = self.G.nodes[node]['probability']
            occurrence_intervals = {1:[0, 1*10**(-6)], 2:[1*10**(-6),50*10**(-6)], 3:[50*10**(-6),100*10**(-6)], 4:[100*10**(-6),1*10**(-3)],
                        5:[1*10**(-3),2*10**(-3)], 6:[2*10**(-3),5*10**(-3)], 7:[5*10**(-3),10*10**(-3)], 
                        8:[10*10**(-3),20*10**(-3)], 9:[20*10**(-3),50*10**(-3)], 10:[50*10**(-3),1]}
            if 'o' in prob_type:
                index = round(sum_probs/instances * 2)
                prob_range = occurrence_intervals[index]
                self.G.nodes[node]['probability'] = (prob_range[1]-prob_range[0])*np.random.random() + prob_range[0]
            else: self.G.nodes[node]['probability'] = sum_probs/instances


    def plot_graph(self):
        # Plot the graph using the multipartite layout in networkx
        pos = nx.multipartite_layout(self.G, subset_key='layer')
        nx.draw_networkx(self.G, pos)
        plt.show()

    def export2graphML(self):
        # Export the graph to a graphML file (for plotting in yEd)
        export2graphML(self.G, "new_w3_graphml2")

    def add_another_node(self, node_name, layer, probability):
        # If node already exists, update the instance and probability attributes
        if node_name in list(self.G.nodes):
            self.G.nodes[node_name]['instance'] += 1
            self.G.nodes[node_name]['probability'] += probability
            
        # Else, create a new node with these attributes
        else: self.G.add_node(node_name, layer=layer, probability=probability, instance=1)

