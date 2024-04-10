import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
from graphBuilder import *
from graphDiagnostics import *
from graphml2nwx import *
from smallWorldMeasures import *
from spectralAnalysis import *
from searchMethods import *
from physicsPositions import *
from fiedlerVoronoi import *
from allNeighbors import *
from allPaths import *

''' plotClusters--------------------------------------------------------------------------------------------------

            ****************************** Last Updated: 27 February 2024 ******************************

 Methods:
 1) get_palette: inputs k (number of colors) --> outputs array of k hex colors

 2) plot_clusters: inputs graph, names of the nodes, array of clusters, palette --> plots the graph (no output)

 3) get_clusters: inputs adjacency matrix, names of the nodes, number of clusters, type of clustering algorithm,
 plotting boolean, specifications for clustering, number of iterations for k-means --> outputs clustered nodes.

 4) get_one_cluster: inputs adjacency matrix, names of the nodes, number of clusters, type of clustering algorithm,
 plotting boolean, specifications for clustering, number of iterations for k-means, source node, target node, 
 threshold for path length, cluster we want --> outputs a graph and an array of the clustered nodes
 
------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------


   get_palette Documentation
 -----------------------------------
 This method inputs k (number of colors) and returns an array of k hex colors.'''

def get_palette(k):
    palette = ["#f6f390", "#ffaeca", "#d799ff", "#84dfff", "#90f6b8", 
            "#ff0a9c", "#ae1857", "#18da8d", "#2033c1", "#40e331",
            "#f9583c", "#dd355b", "#143d55", "#1a1b52", "#312b78", 
            "#ff2323", "#feff5b", "#2a83ff", "#16254d", "#bcbcbc",
            "#f0e3ca", "#40a1c1", "#faeb57", "#647483", "#db000d",
            "#9f0733", "#720428", "#161515", "#04533a", "#05507c",
            "#0084ff", "#44bec7", "#ffc300", "#fa4f5d", "#d696bb",
            "#ff2323", "#feff5b", "#2a83ff", "#16254d", "#a17f7a",
            "#ceaca1", "#ece7e8", "#aab09d", "#909878", "#65737e", "#c99789", "#dde6d5"]
    # Return k colors
    return palette[:k+1]


''' plot_clusters Documentation
 ------------------------------------
 This method inputs a Networkx graph G, names of the nodes, an array of clustered nodes, and a palette. This method 
 plots the graph and has no output.'''

def plot_clusters(G, nodeNames, clusters, palette, cluster_type = ""):
    # Define which nodes are effects and which are modes, and initialize a position dictionary
    effects = nodeNames[:26]
    pos = {}

    # In each cluster, obtain the graph of the cluster. If it is planar, print in a planar layout.
    for i in range(len(clusters)):
        K = G.subgraph(set(clusters[i]))
        if nx.is_planar(K):
            posi = nx.planar_layout(K)

        # If the subgraph is not planar, give the cluster a random layout and rescale it so that it is one kth of
        # the distance of the canvas (k being the number of clusters).
        else:
            posi = nx.random_layout(K)
        addition = nx.rescale_layout_dict(posi, 2/len(clusters))

        # For each position of nodes in the rescaled layout, move the nodes so the clusters form a circle.
        for key in addition:
            [x,y] = addition[key]
            x = 2.0 * np.cos(2 * i * math.pi / len(clusters)) + x
            y = 2.0 * np.sin(2 * i * math.pi / len(clusters)) + y
            addition[key] = [x,y]
        
        # Add the positions of the cluster to the overall position dictionary
        pos.update(addition)

        # For each node in the cluster, draw the node. The colors indicate which cluster each node belongs to, and the shape
        # indicates if the node is an effect or mode. Effects are pentagons and modes are hexagons.
        for node in clusters[i]:
            if node in effects:
                nx.draw_networkx_nodes(G, addition, nodelist=[node], node_color=palette[i], node_size=750, edgecolors="#999999", node_shape="p")
            else:
                nx.draw_networkx_nodes(G, addition, nodelist=[node], node_color=palette[i], node_size=750, edgecolors="#999999", node_shape="H")
    
    # Plot and save the figure
    nx.draw_networkx_labels(G, pos, font_size=5, verticalalignment='center_baseline')
    nx.draw_networkx_edges(G, pos, arrowsize=20)
    plt.box(False)
    plt.savefig("Figures/Clusters/" + cluster_type + "Cluster_allClusters")
    plt.show()


''' get_clusters Documentation
 ------------------------------------
 This method inputs an adjacency matrix, names of the nodes, how many clusters we want, the type of clustering algorithm
 we want to use, a boolean for plotting, the specifications for clustering, and the number of iterations for k-means.
 This method outputs an array of the clustered nodes.'''

def get_clusters(arr, nodeNames, k, cluster_type, plot = False, method = None, n_ints = 100):
    # Create the associated graph and initialize an array of clusters
    G, arr = matrix2Graph(arr, nodeNames, True)
    clusters = []

    # Depending on the type of clustering alorithm requested, compute the appropriate clustering method.
    if cluster_type == "spectral" or cluster_type == "similarity":

        # For spectral and similarity methods, obtain an array that lists the cluster of each node. Then convert
        # this array into an array of cluster arrays (where each cluster array contains the names of the nodes in
        # the cluster as a string).
        if cluster_type == "spectral":
            nums = spectralClustering(arr, k, method)
        elif cluster_type == "similarity":
            sim_mat = similarity_matrix(arr, method)
            vects = eigenval_eigenvects(sim_mat, w_or_sim=True)
            centers, J, nums = kmeans(vects, k, n_ints)
        for i in range(k):
            clusters.append(nodeNames[np.where(nums == i)])

    # Obtain the numerical array of clusters from "fiedler_graph_partition(arr)". Then convert
    # the array of numbers to an array of node names (strings).
    elif cluster_type == "fiedler":
        c1, c2 = fiedler_graph_partition(arr)
        clusters = [nodeNames[c1], nodeNames[c2]]

    # Obtain the numerical array of clusters from "extendedFiedlerPartitions(arr, int(math.log2(k)))". Then convert
    # the array of numbers to an array of node names (strings). The int(math.log2(k)) value roughly will give us k
    # clusters for a given k.
    elif cluster_type == "extended fiedler":
        nums = extendedFiedlerPartitions(arr, int(math.log2(k)))
        for cluster in nums:
            clusters.append(nodeNames[cluster])

    # Use "method" as the centers of the Voronoi split and return the clusters
    elif cluster_type == "voronoi":
        clusters = voronoi_split(G, method)

    # Print an error if an incorrect cluster type is inputted
    else:
        print("Incorrect input type!\nPlease choose 'spectral', 'similarity', 'fiedler', 'extended fiedler', or 'voronoi'.")
        return
    
    # If the user wishes to plot the graph, use the plotting function to plot
    if plot:
        palette = get_palette()
        plot_clusters(G, arr, nodeNames, k, clusters, palette)

    # Return the array of cluster arrays
    return clusters


''' get_one_cluster Documentation
 ------------------------------------
 This method inputs an adjacency matrix, names of the nodes, how many clusters we want, the type of clustering
 algorithm we want to use, a boolean for plotting, the specifications for clustering, the number of iterations for k-means,
 source node, target node, threshold for path length, and which cluster we want. This method outputs a graph and an array
 of the clustered nodes. The difference between get_clusters and get_one_cluster is that get_clusters plots and returns
 a parition of the original graph, whereas get_one_cluster plots and returns a subgraph of the original that consists 
 exclusively of the indicated cluster.'''

def get_one_cluster(arr, nodeNames, k, cluster_type, plot = False, method = None, n_ints = 100, start = 1, target = 2, threshold = 6, cluster = None):
    # Identify which nodes are effects and which are modes
    effects = nodeNames[:26]

    # If the cluster type desired is the graph of all paths under a certain length from a start to end node,
    # plot the subgraph and obtain the paths. Then build the subgraph and array of nodes included in the subgraph.
    if cluster_type == "all paths":
        if plot:
            paths = draw_all_paths(arr, nodeNames, start, target, threshold)
        else:
            adj = make_binary(arr)
            paths = dfs(adj, diagonal_nodes(arr), start, target, threshold = threshold)
        cluster = []
        G = nx.DiGraph()
        print("all paths")
        for path in paths:
            for i in range(len(path)):
                G.add_node(nodeNames[path[i] - 1])
                cluster.append(nodeNames[path[i] - 1])
                if path[i] == target: continue
                G.add_edge(nodeNames[path[i]-1], nodeNames[path[i+1]-1])

    # If the desired clustering algorithm is to show all the parents and children of nodes along the path from a
    # start node to an end node, use the short_paths_neighbors to obtain the subgraph and plot it with draw_short_neighbors
    elif cluster_type == "all neighbors":
        print("all neighbors")
        G, adj, significant_nodes, npath, epath, effs, mods = short_paths_neighbors(arr, nodeNames, target, start)
        cluster = significant_nodes
        if plot:
            draw_short_neighbors(arr, nodeNames, target, start)

    # If the desired clustering algorithm is to show all the children of nodes along the path from a start node to an end node,
    # use the short_paths_child to obtain the subgraph and plot it with draw_short_child
    elif cluster_type == "all children":
        print("all children")
        G, adj, significant_nodes, npath, epath, effs, mods = short_path_child(arr, nodeNames, target, start)
        cluster = significant_nodes
        if plot:
            draw_short_child(arr, nodeNames, target, start)

    # If the desired clustering algorithm is to show all the parents of nodes along the path from a start node to an end node,
    # use the short_paths_parent to obtain the subgraph and plot it with draw_short_parent
    elif cluster_type == "all parents":
        print("all parents")
        G, adj, significant_nodes, npath, epath, effs, mods = short_path_parent(arr, nodeNames, target, start)
        cluster = significant_nodes
        if plot:
            draw_short_parent(arr, nodeNames, target, start)

    # Otherwise, assume the clustering algorithm desired is spectral, similarity, fiedler or voronoi. Obtain the clusters
    # using get_clusters. Select one cluster and plot just this one cluster.
    else:
        print("spectral or similarity or fiedler or voronoi")
        clusters = get_clusters(arr, nodeNames, k, cluster_type, False, method = method, n_ints = n_ints)
        print(clusters)
        gamma, adjacency = matrix2Graph(arr, nodeNames, True)

        # If no specific cluster is indicated, plot and save them all (but in seperate figures)
        if cluster == None:
            if plot:
                for i in range(len(clusters)):
                    G = gamma.subgraph(set(clusters[i]))
                    if nx.is_planar(G):
                        pos = nx.planar_layout(G)
                    else:
                        pos = nx.spring_layout(G)
                    for node in clusters[i]:
                        if node in effects:
                            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="#98c5ed", node_size=750, edgecolors="#c89679")
                        else:
                            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="#fabc98", node_size=750, edgecolors="#799dbd")
                    nx.draw_networkx_labels(G, pos, font_size=5, verticalalignment='center_baseline')
                    nx.draw_networkx_edges(G, pos, arrowsize=20)
                    plt.box(False)
                    filename = "Figures/Clusters/" + cluster_type  + "_" + method + "_cluster" + str(i) + ".png"
                    plt.savefig(filename)
                    plt.show()
                palette = get_palette(k)
                plot_clusters(gamma, arr, nodeNames, k, clusters, palette, cluster_type + "_" + method)
            cluster = 0

        
        # Otherwise, plot and save just one cluster
        else:
            G = gamma.subgraph(set(clusters[cluster]))
            if plot:
                if nx.is_planar(G):
                    pos = nx.planar_layout(G)
                else:
                    pos = nx.spring_layout(G)
                for node in clusters[cluster]:
                    if node in effects:
                        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="#98c5ed", node_size=750, edgecolors="#799dbd")
                    else:
                        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="#fabc98", node_size=750, edgecolors="#c89679")
                nx.draw_networkx_labels(G, pos, font_size=5, verticalalignment='center_baseline')
                nx.draw_networkx_edges(G, pos, arrowsize=20)
                plt.box(False)
                filename = "Figures/Clusters/" + cluster_type + "_" + method + "_cluster" + str(cluster) + ".png"
                plt.savefig(filename)
                plt.show()
    
    # Return the subgraph of the original that is the cluster indicated and return the array of nodes in the cluster
    return G, clusters[cluster]