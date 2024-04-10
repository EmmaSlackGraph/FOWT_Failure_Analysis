import numpy as np
import matplotlib.pyplot as plt 
from graphBuilder import *
from graphDiagnostics import *
from spectralAnalysis import *
from searchMethods import *
from fiedlerVoronoi import *
from plotClusters import *

''' clusterAnalysis-----------------------------------------------------------------------------------------------

            ****************************** Last Updated: 28 February 2024 ******************************

 Methods:
 1) get_sections: inputs type of sections (modes, effects, both) --> outputs dictionary of node names sorted into
 sections

 2) compare_clusters: inputs adjacency martrix, names of the nodes, number of clusters desired, array of clustered nodes, 
 a method indicating the type of cluster type desired, the number of iterations for k-means algorithm --> outputs cosine 
 similarity between the distributions of the cluster and the graph

 3) average_distribution_similarity: inputs adjacency matrix, list of node names, number of iterations, number of clusters
 --> outputs the array of similarities

 4) measure_cluster --> inputs adjacency matrix, list of node names, array of clusters --> outputs the calculated Q-value

 5) get_qs: inputs adjacency matrix, list of node names, number of clusters --> eturns an array of Q-values

 6) average_qs: inputs adjacency matrix, list of node names, number of iterations, number of clusters --> outputs average 
 Q-values across all the iterations
 
------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------


   get_sections Documentation
 -----------------------------------
 This method inputs type of section desired (modes, effects, or both) and outputs a dictionary of node names
 partitioned into sections based on their physical location/what system they are a part of.'''

def get_sections(type):
    if type == "modes":
        return {"Wind Turbine": ["RNA\nstructural", "generator\n& gearbox", "turbine\ncontrols", "tower\nstructural"],
                    "Platform": ["platform\n(structural)", "platform\n(watertightness)","ballast\nsystem\nfailure"],
                    "Mooring": ["chain", "synthetic\nrope", "wire\nrope", "connectors", "clump\nweights\nor floats", 
                                "single\nanchor", "shared\nanchor", ],
                    "Array Cables": ["buoyancy\nmodules", "tether &\nanchor\nsystems", "cable\nprotection\nsystem", 
                                    "dynamic\ncable", "static\ncable", "terminations", "offshore\njoints"]}
    elif type == "effects":
        return {"Turbine": ["Incorrect\nturbine\noperation","Increased\nturbine\nloads","Falling\ntopside\ncomponents",
                               "Turbine\nparked","Reduced\npower\noutput"],
                   "Platform": ["Drift off\nstation","Compromised\nstability","Large\nhydrostatic\noffset","Excess\ndynamics", 
                                "Capsize", "Sink", "Vessel or\nAircraftCollision"],
                   "Mooring Lines": ["Mooring\n-mooring\nclashing", "Mooring\n-cable\nclashing", "Anchor\n-cable\nclashing", 
                                     "Change in\nmooring\nprofile", "Excess\nmooring\nloads", "Mooring\nline\nnonfunctional", 
                                     "Shared\nline\nnonfunctional"],
                   "Mooring Anchors": ["Excess\nanchor\nload", "Anchor\ndragging"],
                   "Array Cable": ["Change\nin cable\nprofile", "Excessive\nload on\ncable", "Array cable\ndisconnect"],
                   "Offshore Substation": ["Substation\n/grid\ninterruption", "Reduced\nAEP"]}

    return {"Wind Turbine": ["RNA\nstructural", "generator\n& gearbox", "turbine\ncontrols", "tower\nstructural",
                                      "Incorrect\nturbine\noperation","Increased\nturbine\nloads","Falling\ntopside\ncomponents",
                                      "Turbine\nparked","Reduced\npower\noutput"],
                 "Platform": ["platform\n(structural)", "platform\n(watertightness)","ballast\nsystem\nfailure", "Drift off\nstation",
                              "Compromised\nstability","Large\nhydrostatic\noffset","Excess\ndynamics", "Capsize", "Sink", 
                              "Vessel or\nAircraft\nCollision"],
                 "Mooring": ["chain", "synthetic\nrope", "wire\nrope", "connectors", "clump\nweights\nor floats", 
                             "single\nanchor", "shared\nanchor", "Mooring\n-mooring\nclashing", "Mooring\n-cable\nclashing", 
                             "Anchor\n-cable\nclashing", "Change in\nmooring\nprofile", "Excess\nmooring\nloads", 
                             "Mooring\nline\nnonfunctional", "Shared\nline\nnonfunctional"],
                 "Array Cables": ["buoyancy\nmodules", "tether &\nanchor\nsystems", "cable\nprotection\nsystem", 
                                  "dynamic\ncable", "static\ncable", "terminations", "offshore\njoints", 
                                  "Change\nin cable\nprofile", "Excessive\nload on\ncable", "Array cable\ndisconnect"],
                 "Mooring Anchors": ["Excess\nanchor\nload", "Anchor\ndragging"],
                 "Offshore Substation": ["Substation\n/grid\ninterruption", "Reduced\nAEP"]}


''' compare_clusters Documentation
 ------------------------------------
 This method inputs an adjacency martrix, names of the nodes, number of clusters desired, an array of clustered nodes, 
 a method indicating the type of cluster type desired, and the number of iterations for k-means algorithm. This method
 counts the number of nodes in each physical system of the floating offshore wind farm (i.e. mooring line, achor, etc.).
 It then compares this distribution with the entire graph's distribution of nodes. It plots a histogram and pie chart
 showing the break-down of physical systems in the cluster. This method returns the cosine similarity between the 
 vectors of distribution for the cluster and the graph.'''

def compare_clusters(arr, nodeNames, k, cluster_type, method = None, n_ints = 100):
    # Get the clusters and the dictionary of physical systems. Initialize a counter for the graph that will
    # find the number of nodes in each system. Initialize an array that will hold the cosine similarities
    # between such counters.
    clusters = get_clusters(arr, nodeNames, k, cluster_type, plot = False, method = method, n_ints = n_ints)
    sections = get_sections("both")
    graph_counter = np.zeros((6,))
    similarities = []

    # Count the number of nodes in each physical system in the graph
    for node in nodeNames:
        if node in sections["Wind Turbine"]:
            graph_counter[0] += 1
        elif node in sections["Platform"]:
            graph_counter[1] += 1
        elif node in sections["Mooring"]:
            graph_counter[2] += 1
        elif node in sections["Array Cables"]:
            graph_counter[3] += 1
        elif node in sections["Mooring Anchors"]:
            graph_counter[4] += 1
        elif node in sections["Offshore Substation"]:
            graph_counter[5] += 1
        else:
            print(node + " --> Error!")

    # Go through each cluster, initialize a new counter for each one, and describe the name of physical systems in an array
    for cluster in clusters:
        counter = np.zeros((6,))
        areas = ["Wind Turbine", "Platform", "Mooring", "Array Cables", "Mooring Anchors", "Offshore Substation"]

        # Count the number of nodes in each physical system in the cluster
        for node in cluster:
            if node in sections["Wind Turbine"]:
                counter[0] += 1
                graph_counter[0] += 1
            elif node in sections["Platform"]:
                counter[1] += 1
                graph_counter[1] += 1
            elif node in sections["Mooring"]:
                counter[2] += 1
                graph_counter[2] += 1
            elif node in sections["Array Cables"]:
                counter[3] += 1
                graph_counter[3] += 1
            elif node in sections["Mooring Anchors"]:
                counter[4] += 1
                graph_counter[4] += 1
            elif node in sections["Offshore Substation"]:
                counter[5] += 1
                graph_counter[5] += 1
            else:
                print(node + " --> Error!")

        # Function to find the percent of nodes in each system
        def func(pct, allvals):
            absolute = int(np.round(pct/100.*np.sum(allvals)))
            return f"{pct:.1f}%"

        # Plot a histogram of the physical systems
        plt.bar(areas,counter, width=0.4)
        plt.xlabel("Areas of Failures")
        plt.ylabel("Number of Failures in Cluster")
        plt.title("Distribution of Nodes in Cluster")
        plt.show()

        # Plot a pie chart of the physical systems
        plt.pie(counter, labels = areas)#autopct=lambda pct: func(pct, counter), textprops=dict(color='w'))
        plt.title("Distribution of Nodes in Cluster")
        plt.legend(areas)
        plt.show()

        # Calculate the similarity of distributions between the cluster and the graph
        similarities.append(cosine_similarity(graph_counter, counter))
    #print(similarities)
        
    # Return the array of similarities
    return similarities


''' average_distribution_similarity Documentation
 --------------------------------------------------
 This method inputs an adjacency matrix, list of node names, number of iterations, and number of clusters. 
 The method then calls the compare_clusters function for each type of cluster and method available for the
 given number of iterations. The average similarity of distributions of the clusters is arranged in array and
 then averaged for each type/method. We output the array of similarities.'''

def average_distribution_similarity(arr, nodeNames, iterations, k):
    # Identify all the cluster types and methods in lists
    cluster_types = ["spectral", "similarity", "extended fiedler"]
    spect_methods = ["normalized", "u2n", "nrw", "unnormalized"]
    sim_methods = ["neighbors", "disasters", "parents", "children"]

    # Initialize an array that will hold each similarity calculation. The first axis is the number of iterations.
    # The second is the type/method of cluster, and the third is the number of clusters (recall that similarity
    # is measured for each cluster, not the clusters as a whole).
    sims = np.zeros((iterations, 9, k))

    # For a given number of iterations, calculate the similarities of node distribution for each cluster type
    # and place into the array.
    for i in range(iterations):
        for j in range(len(cluster_types)):
            if cluster_types[j] == "spectral":
                for m in range(len(spect_methods)):
                    sim = compare_clusters(arr, nodeNames, k, cluster_types[j], spect_methods[m], n_ints = 100)
                    sims[i][m] = sim
            elif cluster_types[j] == "similarity":
                for m in range(len(sim_methods)):
                    sim = compare_clusters(arr, nodeNames, k, cluster_types[j], sim_methods[m], n_ints = 100)
                    sims[i][4 + m] = sim
            else:
                sim = compare_clusters(arr, nodeNames, k, cluster_types[j], n_ints = 100)
                if len(sim) < k:
                    diff = k - len(sim)
                    for n in range(diff):
                        sim.append(0)
                sims[i][-1] = sim

    # Replace the nan values with zeros, print the average similarity for each cluster across the iterations, and then
    # print the average similarity for clusters across all iterations and within cluster types/methods
    sims2 = np.nan_to_num(sims, nan = 0)
    print(np.mean(sims2, axis = 0))
    print(np.mean(np.mean(sims2, axis = 0), axis = 1))

    # Return the array of similarities
    return sims


''' measure_cluster Documentation
 ------------------------------------
 This method inputs an adjacency matrix, list of node names, and array of clusters. Then it finds the
 average normalized unifiability and isolability (Q-value) of the clustering. A high Q-value indicates
 a better clustering of nodes (based on how unified the nodes within each cluster are and how isolated
 each cluster is from the other clusters). This method outputs the calculated Q-value.'''

def measure_cluster(arr, nodeNames, clusters):
    # Initialize arrays to place the measures of unifiability for each cluster. Unifiability compares
    # pairs of clusters (hence it's array has dimension two) and isolability measures each cluster
    # individually (hence it's array has dimension one).
    unifiability = np.zeros((len(clusters),len(clusters)))
    isolability = []

    # For each pair of clusters,
    for i in range(len(clusters)):
        for j in range(len(clusters)):

            # Get the clusters
            ci = clusters[i]
            cj = clusters[j]

            # Initialize the sums included in the calculation of unifiability and isolability
            numerator = 0
            denomenatorU = 0
            denomenatorV = 0
            num = 0
            den = 0

            # Iterate through pairs of nodes
            for m in range(arr.shape[0]):
                for n in range(arr.shape[1]):
                    
                    # Idenitify which clusters the nodes are/are not in and add the weight of the edge between the nodes
                    # to the appropriate sum.
                    if nodeNames[m] in ci:
                        if nodeNames[n] not in ci:
                            den += arr[m,n]
                        if nodeNames[n] in cj:
                            numerator += arr[m,n]
                        else:
                            denomenatorU += arr[m,n]
                        num += arr[m,n]
                    else:
                        if nodeNames[n] in cj:
                            denomenatorV += arr[m,n]
            
            # Calculate te unifiability and isolability
            # print(numerator, denomenatorU, denomenatorV)
            if (denomenatorU == 0 and denomenatorV == 0) and numerator == 0:
                print("error - divide by zero")
                continue
            unifiability[i, j] = (numerator)/(denomenatorU + denomenatorV - numerator)
        if den == 0  and num == 0:
            print("error - divide by zero")
            continue
        isolability.append((num)/(num + den))

    # Find the average unifiability and isolability
    qavu = 1/(len(clusters)) * np.sum(unifiability)
    qavi = 1/(len(clusters)) * sum(isolability)

    # Calculate and return the average normalized unifiability and isolability
    qanui = (qavi)/(1 + qavu * qavi)
    # print(qanui)
    return qanui


''' get_qs Documentation
 ------------------------------------
 This method inputs an adjacency matrix, list of node names, and number of clusters. It calculates the Q-values
 for each cluster type/method and returns an array with these values.'''

def get_qs(arr, nodeNames, k):
    # Identify all the cluster types and methods in lists
    cluster_types = ["spectral", "similarity", "extended fiedler"]
    spect_methods = ["normalized", "u2n", "nrw", "unnormalized"]
    sim_methods = ["neighbors", "disasters", "parents", "children"]

    # Itialize array for all the Q-values
    qs = []

    # Go through each cluster type and method, and calculated the Q-value for each clustering. Append Q-value to the
    # array already initialized.
    for cluster_type in cluster_types:
        # print(cluster_type)
        if cluster_type == "spectral":
            for method in spect_methods:
                clusters = get_clusters(arr, nodeNames, k, cluster_type, plot = False, method = method, n_ints = 100)
                if len(clusters) > 1:
                    qs.append(measure_cluster(arr, nodeNames, clusters))
        elif cluster_type == "similarity":
            for method in sim_methods:
                # print("method - ", method)
                clusters = get_clusters(arr, nodeNames, k, cluster_type, plot = False, method = method, n_ints = 100)
                if len(clusters) > 1:
                    qs.append(measure_cluster(arr, nodeNames, clusters))
        else:
            clusters = get_clusters(arr, nodeNames, k, cluster_type, plot = False, method = method, n_ints = 100)
            if len(clusters) > 1:
                    qs.append(measure_cluster(arr, nodeNames, clusters))

    # Return the array of Q-values
    return qs


''' average_qs Documentation
 ------------------------------------
 This method inputs an adjacency matrix, list of node names, number of iterations, and number of clusters. For the
 number of iterations specified, this method calculates the Q-values for each cluster type/method and returns the
 average Q-values across all the iterations.'''

def average_qs(arr, nodeNames, iterations, k):
    # Initialize array to store Q-value arrays in
    iteration = []

    # For each iteration, use the get_qs method to calculate the Q-values of each cluster type and method
    for i in range(iterations):
        iteration.append(get_qs(arr, nodeNames, k))

    # Turn the iteration array into a numpy array
    iteration = np.array(iteration)

    # Print and return the average Q-value for each cluster across the number of iterations
    print(np.mean(iteration, axis = 0))
    return np.mean(iteration, axis = 0)