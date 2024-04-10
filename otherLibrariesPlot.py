from graphBuilder import *
from dash import Dash, html
import dash_cytoscape as cyto
import easygraph as eg
import matplotlib.pyplot as plt
import igraph as ig

''' graphDiagnostics--------------------------------------------------------------------------------------------

            ****************************** Last Updated: 19 February 2024 ******************************

 Methods:
 1) plot_in_library: inputs adjacency matrix, nodeNames, library desired, layout desired --> outputs plot (no return)

 2) find_plotting_layouts: inputs (optional) library --> prints available plotting layouts (no return)

 3) plot_in_dc: inputs adjacency matrix, nodeNames, and layout desired --> plots graph (no return)

 4) plot_in_eg: inputs adjacency matrix, nodeNames, and layout desired --> plots graph (no return)

 5) plot_in_ig: inputs adjacency matrix, nodeNames, and layout desired --> plots graph (no return)

 6) plot_ig_trees: inputs Networkx graph --> plots graph using Python -igraph (no return)
 
------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------


   plot_in_library Documentation
 ----------------------------------
 This method inputs the adjacency matrix, nodeNames, library desired (string), and layout desired (string). This
 method prints the graph requested by calling other functions in this document. Nothing is returned.'''

def plot_in_library(arr, nodeNames, library, layout):
    # If the input is "cyto" (referring to Dash Cytoscape), then we call the dash cytoscape plotting function
    if library == "cyto":
        plot_in_dc(arr, nodeNames, str(layout))

    # If the input is "eg" (referring to Easy Graph), then we call the easy graph plotting function
    elif library == "eg":
        plot_in_eg(arr, nodeNames, layout)
    
    # If the input is "ig" (referring to igraph), then we call the igraph plotting function
    elif library == "ig":
        plot_in_ig(arr, nodeNames, layout)

    # If none of the above entries are inputted, then we plot a networkx graph using methods from graphBuilder.py
    else:
        print("Unknown library! Plotting in Networkx instead...")
        G, arr = matrix2Graph(arr, nodeNames, True)
        plot_graph(G, layout, nodeNames)

    # Noting is returned
    return



''' find_plotting_layouts Documentation
 ----------------------------
 This method has an optional input declaring the library of interest (as string( and prints out the different plotting
 layouts supported by each of the three libraries we are interested in: Dash Cytoscape, Easy Graph, and Python -igraph.
 If no library is specified, then all the layouts for all three libraries are printed. Nothing is returned.'''

def find_plotting_layouts(library = None):
    # The following are the layouts we can use with Dash Cytoscape (spelling must be accurate when specifying the
    # layout when trying to print).
    if library == "cyto":
        print("Cytoscape Layouts: breadthfirst, circle, concentric, cose, grid, random")
        print("External Layouts Supported by Cytoscape: cose-bilkent, cola, euler, spread, dagre, klay")
        print("For more information, visit: https://dash.plotly.com/cytoscape/layout")

    # The following are the layouts we can use with Easy Graph (spelling must be accurate when specifying the
    # layout when trying to print).
    elif library == "eg":
        print("Easy Graph Printing Layouts: circular, kamada_kawai, shell, random")
        print("For more information, visit: https://easy-graph.github.io/docs/reference/easygraph.functions.drawing.html")
    elif library == "ig":
        print("igraph Regular Layouts: circle, grid, grid_3d, random, star")
        print("igraph Physics Layouts: davison_harel, drl, fruchterman_reingold, graphopt, kamada_kawai, lgl, mds, umap")
        print("igraph Tree Layouts: reingold_tilford, reingold_tilford_circular, sugiyama")
        print("igraph Bipartite Layouts: bipartite")
        print("For more information, visit: https://python.igraph.org/en/stable/visualisation.html")

    # The following are the layouts we can use with Python -igraph (spelling must be accurate when specifying the
    # layout when trying to print).
    else:
        print("Cytoscape Layouts: breadthfirst, circle, concentric, cose, grid, random")
        print("External Layouts Supported by Cytoscape: cose-bilkent, cola, euler, spread, dagre, klay")
        print("-----")
        print("Easy Graph Printing Layouts: circular, kamada_kawai, shell, random")
        print("-----")
        print("igraph Regular Layouts: circle, grid, grid_3d, random, star")
        print("igraph Physics Layouts: davison_harel, drl, fruchterman_reingold, graphopt, kamada_kawai, lgl, mds, umap")
        print("igraph Tree Layouts: reingold_tilford, reingold_tilford_circular, sugiyama")
        print("igraph Bipartite Layouts: bipartite")



''' plot_in_dc Documentation
 ----------------------------
 This method inputs the adjacency matrix, names of the nodes, and the layout desired for plotting. This method plots the 
 corresponding Dash Cytoscape plot. In order to see plot, you must click on the link outputted in the terminal by the 
 cytoscape package. Nothing is returned.'''

def plot_in_dc(arr, nodeNames, layout):
    # Load the extra plotting layouts for Dash Cytoscape so that all are available.
    cyto.load_extra_layouts()
    
    # We are going to build the graph using the Dash Cytoscape package. In order to do this, we need to obtain a
    # dictionary of nodes and dictionary of edges. We do this by calling the get_node_dict and get_graph_edges
    # methods from graphBuilder.py.
    nodes = get_node_dict(arr, nodeNames)
    edges = get_graph_edges(arr, 'dict', 0.5)

    # Combine the node dictionary and edge dictionary into one large dictionary
    elems = nodes + edges

    # Initialize the Dash workspace
    app = Dash(__name__)

    # Plot the graph using the Dash Cytoscape format
    app.layout = html.Div([

        # Title the graph (CHANGE THIS)
        html.P("Failure Modes and Effects Interactive Graph"),
        cyto.Cytoscape(

            # Specify that we want to use the cytoscape package from C
            id='cytoscape',

            # Specify the nodes and edges using the dictionary we created
            elements=elems,

            # Indicate which type of layout we want
            layout={'name': layout},

            # Speficy the size of the plot
            style={'width': '100%', 'height': '700px'},

            # Indicate hte style that we want for our graph
            stylesheet=[
                # Specify that we want to label the nodes with their names (strings)
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(label)'
                    }
                },

                # Specify that we want the failure mode nodes to be colored orange
                {
                    'selector': '.mode',
                    'style': {
                        'background-color': 'orange'
                    }
                },

                # Specify that we want the failure effect nodes to be colored blue
                {
                    'selector': '.effect',
                    'style': {
                        'background-color': 'blue'
                    }
                },

                # Specify that we want our edges to be directed, straight, and gray
                {'selector': 'edge',
                    'style': {
                        'width': 3,
                        'line-color': '#ccc',
                        'target-arrow-color': '#ccc',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'straight'
                    }}]
        )
    ])

    # Run the program to access the plot
    app.run_server(debug=True)

    # Nothing is returned
    return



''' plot_in_eg Documentation
 ----------------------------
 This method inputs the adjacency matrix, names of the nodes, and the layout desired for plotting. This method plots the 
 corresponding Easy Graph plot. Nothing is returned.'''

def plot_in_eg(arr, nodeNames, layout):
    # Initialize a blank Easy Graph graph
    G = eg.DiGraph()

    # Format the node data by calling the get_node_array method from graphBuilder.py
    labels = get_node_array(arr, nodeNames)

    # Format the edge data by calling the get_graph_edges method from graphBuilder.py (inputting 'tuple')
    edges = get_graph_edges(arr, 'tuple', 0.5)

    # Add the nodes and edges using the gathered formats above and use them to build our graph
    G.add_nodes([i for i in range(arr.shape[0])], nodes_attr= labels)
    G.add_edges(edges)

    # Depending on the layout declared, we use different methods from Python -igraph to plot our graph.
    # If 'kamaada_kawai' is inputted, then we plot our grpah in a kamada_kawai layout (physics based layout)
    if layout == "kamada_kawai":
        eg.draw_kamada_kawai(G)
    
    # If 'circular' is inputted, then we plot our grpah in a circle layout
    elif layout == "circular":
        pos = eg.circular_position(G)
        eg.draw_easygraph_edges(G, pos)
        eg.draw_easygraph_nodes(G, pos)
    
    # If 'shell' is inputted, then we plot our grpah in concentric circles
    elif layout == "shell":
        pos = eg.shell_position(G)
        eg.draw_easygraph_edges(G, pos)
        eg.draw_easygraph_nodes(G, pos)
    
    # If none of the above layouts are inputted, then we plot our graph in a random layout
    else:
        pos = eg.random_position(G)
        eg.draw_easygraph_edges(G, pos)
        eg.draw_easygraph_nodes(G, pos)
    
    # Nothing is returned
    return



''' plot_in_ig Documentation
 ----------------------------
 This method inputs the adjacency matrix, names of the nodes, and the layout desired for plotting. This method plots the 
 corresponding Python -igraph plot. Nothing is returned.'''

def plot_in_ig(arr, nodeNames, layouts):
    # Create our igraph graph using adjacency matrix inputted
    g = ig.Graph.Adjacency(arr)

    # Specify if the nodes are failure effects or failure modes
    g.vs["mode"] = [False for i in range(arr[:26].shape[0])] + [True for i in range(arr[26:].shape[0])]

    # Depending on if the nodes are effects or modes, we will color them differently (effects blue, modes orange)
    color_dict = {True: "#fabc98", False:"#98c5ed"}

    # We need to specify that we are plotting by way of matplotlib
    ig.config['plotting.backend'] = 'matplotlib'

    # Plot the graph. 'g' specifies which graph we are plotting.
    ig.plot(g,
            
            # Name the type of layout we want to use
            layout=layouts,

            # Indicate the size of the nodes
            vertex_size=20,

            # List the color of each node using the array and dictionary we created above
            vertex_color=[color_dict[mode] for mode in g.vs["mode"]],

            # Label each vertex. In this case, we are labeling with words rather than numbers
            vertex_label=[nodeNames[i] for i in range(arr.shape[0])],

            # Indicate the thickness of the nodes
            edge_width=[1])
    
    # Plot the graph using matplotlib
    plt.show()



''' plot_ig_trees Documentation
 ----------------------------
 This method inputs a networkx graph and plots a Python -igraph tree. Nothing is returned'''

def plot_ig_trees(G):
    # Create a Python -igraph graph from a Networkx graph
    g = ig.from_networkx(G)

    # We need to specify that we are plotting by way of matplotlib
    ig.config['plotting.backend'] = 'matplotlib'

    # Plot the graph. 'g' specifies which graph we are plotting.
    ig.plot(g,
            
        # Name the type of layout we want to use
        layout="random",

        # Indicate the size of the nodes
        vertex_size=20,

        # Indicate the thickness of the nodes
        edge_width=[1])