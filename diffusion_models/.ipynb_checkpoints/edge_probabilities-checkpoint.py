import numpy as np
import networkx as nx
import torch

def generate_edge_probabilities(G, method='uniform', seed=None, **kwargs):
    """
    Generate edge probabilities for the graph G based on the specified method.

    Parameters:
    - G (networkx.Graph): The input graph.
    - method (str): The method to use for generating probabilities. Options:
        - 'uniform': Assign a uniform probability to all edges.
        - 'random': Assign random probabilities to edges.
        - 'weighted': Assign probabilities based on edge weights.
    - kwargs: Additional parameters specific to the chosen method.

    Returns:
    - edge_probs (dict): A dictionary where keys are edges and values are probabilities.
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    edge_probs = {}

    if method == 'uniform':
        p = kwargs.get('p', 0.01)
        for edge in G.edges():
            edge_probs[edge] = p
            if G.is_directed()==False:
                edge_probs[edge] = p

    elif method == 'random':
        low = kwargs.get('low', 0.01)
        high = kwargs.get('high', 1)
        for (u, v) in G.edges():
            edge_probs[(u, v)] = np.random.uniform(low, high)
            #edge_probs[(u,v)] = np.random.normal(loc=0.5, scale=0.25)/3
            if edge_probs[(u,v)] < 0:
                edge_probs[(u,v)] = 0
            elif edge_probs[(u,v)] > 1:
                edge_probs[(u,v)] = 1
            #edge_probs[(v, u)] = edge_probs[(u, v)]
            #if G.is_directed()==False:
                #edge_probs[(v, u)] = edge_probs[(u, v)]

    elif method == 'weighted_sum':
        # Retrieve or set default weights for node and edge features
        node_weights = kwargs.get('node_weights', None)
        edge_weights = kwargs.get('edge_weights', None)
        max_edge = kwargs.get('max_edge', 0.333)

        for v, u, data in G.edges(data=True):
            # Retrieve node features; assume they are stored under the 'features' key
            v_features = np.array(G.nodes[v].get('features', np.zeros(1)))
            u_features = np.array(G.nodes[u].get('features', np.zeros(1)))
            # Retrieve edge features; assume they are stored under the 'features' key
            e_features = np.array(data.get('features', np.zeros(1)))

            # Validate or initialize weights
            if node_weights is None:
                node_weights = np.ones_like(v_features)
            if edge_weights is None:
                edge_weights = np.ones_like(e_features)

            # Compute weighted sums
            node_feature_sum = np.dot(node_weights, v_features) + np.dot(node_weights, u_features)
            edge_feature_sum = np.dot(edge_weights, e_features)
            edge_score = (node_feature_sum + edge_feature_sum)

            # Make sigmoid input more negative (optional)
            # edge_score = -abs(edge_score)  # stronger push toward 0
        
            #edge_probs[(v, u)] = 1 / (1 + np.exp(-edge_score))
            # Step 1: Get all the values
            degree = max(G.out_degree(u), 1)
            edge_probs[(v, u)] = edge_score/degree
            if G.is_directed()==False:
                edge_probs[(u, v)] = edge_score
        # Step 1: Get all the values
        values = list(edge_probs.values())
        
        # Step 2: Compute min and max
        min_val = min(values)
        max_val = max(values)
        
        # Step 3: Normalize using the formula
        edge_probs = {
            edge: max_edge*(prob - min_val) / (max_val - min_val)
            #edge: max_edge / (1 + np.exp(prob))
            for edge, prob in edge_probs.items()
        }
    elif method == 'degree_based':
        c = kwargs.get('c', 1)  # overall scaling factor
        for v, u in G.edges():
            if G.is_directed():
                degree = max(G.out_degree(u), 1)  # Avoid division by 0
                edge_probs[(v, u)] = c / degree
            else:
                degree = max(G.degree(v), 1)  # Avoid division by 0
                edge_probs[(v, u)] = c / degree
                edge_probs[(u, v)] = edge_probs[(v, u)]
    else:
        raise ValueError("Invalid method. Choose from 'uniform', 'random', 'degree_based' or 'weighted'.")

    return edge_probs
