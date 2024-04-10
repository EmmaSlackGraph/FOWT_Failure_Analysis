import networkx as nx
from graphBuilder import *
from graphDiagnostics import *
import xml.etree.ElementTree as ET

''' graphml2nwx--------------------------------------------------------------------------------------------------------

            ****************************** Last Updated: 19 February 2024 ******************************

 Methods:
 1) graphml2graph: inputs graphML file --> outputs G (Networkx graph), array of node information, and adjacency matrix.
 
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------


   graphml2graph Documentation
 -------------------------------
 This method takes in a graphML filename (string) and outputs the created Networkx graph, a list of the nodes with
 information about their color, number, and name, and the corresponding adjacency matrix for the graph.'''

def graphml2graph(filename):

    # G = nx.read_graphml("bigMatrix_array.graphml")
    G = nx.read_graphml(filename)

    # tree = ET.parse('bigMatrix_array.graphml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Initialize the node name and number to "error" so that we know what went wrong in the reading process
    node_num = "error"
    node_text = "error"

    # Initialize the array of all the nodes
    nodes = []

    # Iterate through the graphML file (for those of you who are familiar, graphML files work the same as XML files)
    # to find all the nodes in the graph.
    for node in root[-1].findall("{http://graphml.graphdrawing.org/xmlns}node"):

        # Access the data for each node
        for data in node.findall("{http://graphml.graphdrawing.org/xmlns}data"):

            # For the specific data of the node's number, the data tag must have length zero.
            if len(data) == 0:
                node_num = data.text # Set the node number to the number from the graphML file

            # For all other data that we are interested in, 
            elif len(data) > 0:

                # Find all the instances of the "list" tag in the file. From this tag, we will find the name of
                # each failure effect/mode (i.e. the name of each node) and create array with this information.
                for x_list in data.findall("{http://www.yworks.com/xml/yfiles-common/markup/3.0}List"):
                    node_text = x_list[0][0].text
                    nodes.append([int(node_num), "n" + node_num, node_text])

                # We will now find which turbine each node belongs to by accessing the color attribute in
                # each "ShapeNodeStyle" tag. Depending on the color, we will add a value between 1 and 10
                # to the array for each specific node that indicates if the node belongs to turbine 1, 2, etc.
                for color in data.findall("{http://www.yworks.com/xml/yfiles-for-html/2.0/xaml}ShapeNodeStyle"):
                    if color.attrib['fill'] == '#FFDBB0FF':
                        nodes[-1].append(3)
                    elif color.attrib['fill'] == '#FF41ECFF':
                        nodes[-1].append(1)
                    elif color.attrib['fill'] == '#FFFBB259':
                        nodes[-1].append(9)
                    elif color.attrib['fill'] == '#FFCCCACA':
                        nodes[-1].append(7)
                    elif color.attrib['fill'] == '#FFDB21F9':
                        nodes[-1].append(5)
                    elif color.attrib['fill'] == '#FFFF564F':
                        nodes[-1].append(4)
                    elif color.attrib['fill'] == '#FF88FF88':
                        nodes[-1].append(2)
                    elif color.attrib['fill'] == '#FFFFD2FC':
                        nodes[-1].append(10)
                    elif color.attrib['fill'] == '#FF99FBEA':
                        nodes[-1].append(8)
                    elif color.attrib['fill'] == '#FFFAF498':
                        nodes[-1].append(6)

                    # If the color is not one of the colors above, then the nodes are (most likely)
                    # shared mooring lines or shared anchors. Depending on the color, either append
                    # 'm' for mooring line or 'a' for anchor to the array for that specific node.
                    elif color.attrib['fill'] == '{y:GraphMLReference 48}':
                        nodes[-1].append('m')
                    elif color.attrib['fill'] == '{y:GraphMLReference 49}':
                        nodes[-1].append('a')

        # Create an adjacency matrix for the graph
        arr = nx.to_numpy_array(G)

    # Return the graph, node information, and adjacency matrix
    return G, nodes, arr

