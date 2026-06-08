import networkx as nx
import torch

NUM_NODES = 512
NODE_DIM = 16


def build_topology(device):
    """Builds the 9D hypercube graph, its adjacency matrix, and node shells.

    Shells group nodes by Hamming weight (popcount of the node index),
    matching the hierarchy used to weight inputs in encode_to_brain.
    """
    graph = nx.hypercube_graph(9)
    graph = nx.convert_node_labels_to_integers(graph)
    adjacency = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)
    shells = {i: bin(i).count("1") for i in range(NUM_NODES)}
    return graph, adjacency, shells
