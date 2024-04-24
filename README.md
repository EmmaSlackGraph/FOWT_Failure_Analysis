# Floating Offshore Wind Turbine Failure Analysis

## Project Description

Fundamentally, this code reads an adjacency matrix from Excel, computes a corresponding graph in Networkx, and runs calculations on this graph. These calculations include plotting data, degree, type of graph (Eulerian, directed acyclic, etc), partitioning the graph, and finding paths between nodes. This code has been designed specifically to analyze failure mode and effects of floating offshore wind turbines, but our code can be expanded to accomodate other fields.

We used the Networkx package in Python since it had one of the largest libraries of algorithms available. To read more about Networkx, navigate to https://networkx.org.

Although we have found clusters in the graph, we have not gleaned any information from these groupings. We hope to implement methods to measure the quality of the clustering and to analyze individual cluster behaviour.

## How to Install and Run the Project
  ### 1) Download and Install Required Packages
  > There are several packages used in these files that you will need to have installed. The libraries needed are:
  > 1) `numpy`
  > 2) `pandas`
  > 3) `matplotlib`
  > 4) `networkx`
> 
 >  If you wish to use other graph theory libraries in Python (such as Dash Cytoscape, Python -igraph, or Easy Graph), then make sure to install those as well.
  
  ### 2) Download Python Files
> Download the files listed above and put into the same folder. It may be helpful to have a subfolder named "Figures," but this is optional.
  
  ### 3) Construct Graph
>  Place your Excel file in the same folder as the code you just downloaded. Make sure that the only labels are the names of the nodes. Everything else in the Excel file should be a number or empty. Also make sure that the information in your Excel file is an adjacency matrix (meaning the matrix should have the same labels on the side and on the top, and they should be in the same order). You are now ready to import into Python!
>  
>  Use the `excel2Matrix()` method found in `graphBuilder.py` to construct your adjacency matrix and the `matrix2Graph()` method also from `graphBuilder.py` to build your graph. Your are now ready to run calculations with your graph!

## How to Use the Project
There is no 'right' or 'wrong' way to use this code (necessarily). There are a lot of different avenues to explore, so we encourage you to consult the summary of the files below and decide which methods you want to call based on that.

  #### 1) allNeighbors.py
   This code starts with an altered breadth-first search to find a shorest path between a start and target node. This path is added to a subgraph of the  inputted nework. We then find all the parents, children, or parents and children of every node on the path and adds these "neighbors" to the subgraph.

  #### 2) allPaths.py
   This code uses an altered breadth-first search to find all the paths (without repeating nodes) between a start and target node. Since this computation  could be computationally large for some graphs, we placed a threshold parameter into the method. It restricts the list of paths generated to be on length less than or equal to the threshold. If not stated, the theshold is assumed to be 47.

  #### 3) arrayPropagation.py
   This code uses the breadth-first search from the searchMethods.py code to track failure propagation within and between floating offshore wind turbines. This file also includes methods for monte-carlo simulations and plotting turbine-wide failure cascades with given probabilities.
   
  #### 4) clusterAnalysis.py
   This code analyzes the distribution of nodes in clusters versus in the graph at large, as well as mesaures how good the clustering is via unifiability and isolability (refer to Biswas and Biswas' "Defining quality metrics for graph clustering evaluation"  at https://www.sciencedirect.com/science/article/pii/S0957417416306339#sec0004 for more information about these metrics).

  #### 5) connectivity.py
   These methods measure the connectiviy between any pair of nodes (via max flow, hitting time, page rank, and shortest path algorithms).
   
  #### 6) failureProbabilities.py
   This file estimates conditional probabilities for pairs of failures and plots a graph of failure propagation given these probabilities. This method also includes methods for generating and calculating inference over Bayesian networks.

  #### 7) fiedlerVoronoi.py
   This file contains methods which partition the graph via the Fiedler or Voronoi partition algorithms. The methods for computing Fiedler partitions are calculated directly, while the Voronoi cell method relies on functions from Networkx.
  
  #### 8) graphBuilder.py
   This file reads in files from Excel (make sure the Excel file(s) are in the same folder as the code you are working with!). The `graphBuilder.py` file  also contains methods for constructing the graph from the matrices collected in Excel, plotting the graph, and calculating the maximum degree (in, out, and overall degree) of the graph.

  #### 9) graphDiagnositics.py
   The methods in this file calculates several measures for the graph, including if the graph is directed acyclic, if it is Eulerian, its efficiency, its flow hierarchy, its reciprocity, a triadic census (how many of each type of "triangle" is in the graph), and the average degree of each node's neighbors (in, out, and combined). This code also calculates the shortest distance between two nodes and gets the degrees for all nodes in the graph.
  
  #### 10) graphml2nwx.py
   This file reads a `graphML` file (presumably from yEd) and constructs a corresponding graph in Networkx.

  #### 11) holding_block.py
  This file is a document with in-progress code.

  #### 12) main.py
  Use this file to run code in other files.
  
  #### 13) otherLibrariesPlot.py
   This file uses several other Python libraries (including Dash Cytoscape, Python `-igraph`, and Easy Graph) to plot inputted graphs.
  
  #### 14) physicsPositions.py
   The code in this file implements the Barycentric method and Fruchterman & Reingold methods. The Barycentric method is a plotting algorithm that attempts to place each 'free' node in the center of all other nodes in the graph. But, this means that the Barycentric method requires an initial set of 'fixed' nodes that give the free nodes a starting point. The Fruchterman & Reingold method is another plotting algorithm, but uses attractive and repulsive forces to place the nodes (akin to spring systems such that each edge is a spring and each node a steel ring).
  
  #### 15) plotClusters.py
  This file finds subgraphs of the inputted failure modes/effects graph. These subgraphs are either computed through clustering algorithms or built from finding paths throughout the graph. The plotClusters.py also has plotting methods to visualize the subgraphs and clusters.

  #### 16) poster_calculation.py
  Run this file to calculate all pairwise probabilities in an array and write to an Excel file.
  
  #### 17) searchMethods.py
   The code in this file uses an altered depth-first search to simulate a cascading failure. For a given failure, the method finds all the immediate consequences (aka the node's children), then all the secondary consequences (aka the node's grandchildren), and so on until there are no more nodes to visit. This produces a subgraph in which all edges are pointing in the same direction (the graph is multipartite).
  
  #### 18) smallWorldAnalysis.py
   This code evaluates if the inputted graph is 'small-world.' Small world graphs are a unique type of network in which information/disease/failures travel quickly and reliably. By reliable travel, we mean that there are multiple routes to each node. So if one route is blocked, there is another way to reach the destination.
  
  #### 19) spectralAnalysis.py
  This file finds clusters in the graph using spectral methods. In other words, this file uses the eigenvalues and eigenvectors of the graph Laplacian (a matrix which encodes information about the nodes' degrees and adjacency) to determine how to cluster the network.

## Credits
Developed by Emma Slack and Matt Hall, with support from the National Renewable Energy Laboratory (NREL) and the Department of Energy's (DOE) Science Undergradute Laboratory Internships (SULI) program.

***
Last Updated: 10 April 2024
](https://github.nrel.gov/eslack/FOWT_failure_analysis.git/)