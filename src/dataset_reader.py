from os import path

import networkx as nx


def read_graph(file_name: str, weighted: bool = False)-> nx.Graph:

    """Read edges from file
    Args:
        file_name: File name (Here, 'example1.dat')
    Returns:
        List of edges that are described as a list of two nodes
    """

    # Define the relative path to the zipped data
    file_dir = path.dirname(__file__)   
    rel_path = "../data/" + file_name
    dataset_path = path.join(file_dir, rel_path)

    # Read the data set and save them in a list 

    if weighted:
        graph = nx.read_edgelist(path = dataset_path, nodetype=int, delimiter=',')
    else:
        graph = nx.read_edgelist(path = dataset_path, nodetype=int, delimiter=',')
    

    return graph

