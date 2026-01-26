import numpy as np
import torch
import networkx as nx
import math
from collections import defaultdict
import itertools
from scipy.sparse import csr_matrix

def optimized_independent_cascade(G, prior_probs, edge_probs, k, seed=None):
    """
    Optimized version of the Independent Cascade model using NumPy.

    Parameters:
    - G (networkx.Graph): The input graph.
    - prior_probs (dict): Initial infection probabilities for each node.
    - edge_probs (dict): Activation probabilities for each edge.
    - k (int): Number of simulation steps.

    Returns:
    - posterior_probs (dict): Infection probabilities after k steps.
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
    
    nodes = list(G.nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_indices.items()}
    n = len(nodes)

    # Precompute adjacency list with edge probabilities
    adjacency = {node: [] for node in nodes}
    for u, v in G.edges():
        adjacency[u].append((v, edge_probs[(u, v)]))
        #if (u, v) in edge_probs:
            #adjacency[u].append((v, edge_probs[(u, v)]))
        #if (v, u) in edge_probs:
            #adjacency[v].append((u, edge_probs[(v, u)]))

    infection_counts = np.zeros(n, dtype=np.int32)
    prior_array=np.array(prior_probs.squeeze())

    for _ in range(k):
        rand_vals = np.random.rand(n)
        active = rand_vals < prior_array
        visited = active.copy()
        infection_counts += active.astype(np.int32)

        newly_active = set(np.where(active)[0])

        while newly_active:
            next_active = set()
            for idx in newly_active:
                node = idx_to_node[idx]
                for neighbor, p in adjacency[node]:
                    neighbor_idx = node_indices[neighbor]
                    if not visited[neighbor_idx] and np.random.rand() < p:
                        visited[neighbor_idx] = True
                        next_active.add(neighbor_idx)
                        infection_counts[neighbor_idx] += 1
            newly_active = next_active

    posterior_probs = {idx_to_node[i]: infection_counts[i] / k for i in range(n)}
    return posterior_probs

def Edge_Simulation(G, prior_probs, edge_probs, k, seed=None):
    """
    Edge Simulation (Algorithm 2 in https://doi.org/10.14232/actacyb.21.1.2013.4):
    For each sample:
      - sample live-edge graph G' by activating edges with probability p_e
      - for each node v:
          f_v += 1 - prod_{u reaches v in G'} (1 - p_u)

    Parameters
    ----------
    G : networkx graph (directed expected, but works for undirected too)
    prior_probs : array-like (n,) or dict {node: p_u}
        Initial infection probabilities p_u.
    edge_probs : dict {(u,v): p_uv}
        Transmission probabilities per edge.
    k : int
        Number of samples.
    seed : int or None
        RNG seed.

    Returns
    -------
    posterior_probs : dict {node: f_v}
        Estimated infection probabilities per node.
    """
    # Reproducible RNG
    rng = np.random.default_rng(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    # Prior probs -> array aligned with nodes
    if isinstance(prior_probs, dict):
        p_u = np.array([prior_probs[idx_to_node[i]] for i in range(n)], dtype=float)
    else:
        p_u = np.asarray(prior_probs, dtype=float).reshape(-1)
        if p_u.shape[0] != n:
            raise ValueError(f"prior_probs has length {p_u.shape[0]}, but G has {n} nodes.")

    # log(1 - p_u) (handle p_u==1 -> -inf cleanly)
    with np.errstate(divide="ignore", invalid="ignore"):
        log1m_p = np.log1p(-p_u)  # log(1 - p_u)

    # Flatten edges into arrays for fast sampling
    edges = list(G.edges())
    m = len(edges)
    src = np.empty(m, dtype=np.int32)
    dst = np.empty(m, dtype=np.int32)
    pe  = np.empty(m, dtype=float)
    for i, (u, v) in enumerate(edges):
        src[i] = node_to_idx[u]
        dst[i] = node_to_idx[v]
        try:
            pe[i] = edge_probs[(u, v)]
        except KeyError:
            if (v, u) in edge_probs:
                pe[i] = edge_probs[(v, u)]
            else:
                raise KeyError(f"Missing edge probability for edge ({u}, {v}).")

    f = np.zeros(n, dtype=float)

    for _ in range(k):
        # 1) Sample live edges
        active_mask = rng.random(m) < pe

        # Build adjacency list for sampled G'
        adj = [[] for _ in range(n)]
        active_src = src[active_mask]
        active_dst = dst[active_mask]
        for a, b in zip(active_src, active_dst):
            adj[a].append(b)

        # 2) Accumulate log s_v = sum_{u reaches v} log(1 - p_u)
        log_s = np.zeros(n, dtype=float)

        # For each u, DFS/BFS to find nodes reachable from u in G'
        # and add log1m_p[u] to those nodes.
        for u in range(n):
            w = log1m_p[u]
            # If p_u == 0 then log(1-p_u)=0, adds nothing, skip
            if w == 0.0:
                continue

            seen = np.zeros(n, dtype=bool)
            stack = [u]
            seen[u] = True
            while stack:
                x = stack.pop()
                # u reaches x, so x gets multiplied by (1-p_u) in s_x
                log_s[x] += w
                for y in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        stack.append(y)

        # 3) f_v += 1 - s_v
        # s_v = exp(log_s[v]) (if log_s[v] = -inf => s_v=0 => contributes 1)
        s = np.exp(log_s)
        f += (1.0 - s)

    f /= float(k)
    return {idx_to_node[i]: f[i] for i in range(n)}

def dmp_um_python(G, prior_probs, edge_probs, T=10):
    # INITIALIZATION:
    p_without = {}
    p_without_stepback = {}
    for (u,v) in G.edges():
        p_without[(u,v)] = prior_probs[u]
        p_without_stepback[(u,v)] = 0
    p = np.zeros(G.number_of_nodes())
        
    # MAIN LOOP:
    for _ in range(T):
        p_without_new = p_without.copy()
        for (u,v) in G.edges():
            influence = 1
            for w in G.neighbors(u):
                if w != v:
                    influence *= (1-edge_probs[(w,u)]*(p_without[(w,u)]-p_without_stepback[(w,u)]))
            p_without_new[(u,v)] = 1 - (1-p_without[(u,v)])*influence
        p_without_stepback = p_without.copy()
        p_without = p_without_new.copy()

    # FINAL VALUES:
    for u in G.nodes():
        influence = 1
        for v in G.neighbors(u):
            influence *= (1-edge_probs[(v,u)]*p_without[(v,u)])
        p[u] = 1 - (1-prior_probs[u])*influence

    return p

def dmp_um_fast(G, prior_probs, edge_probs, T=10, eps=1e-12):
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}

    edges = list(G.edges())
    m = len(edges)

    src = np.array([idx[u] for u, v in edges], dtype=np.int64)
    dst = np.array([idx[v] for u, v in edges], dtype=np.int64)
    p_edge = np.array([edge_probs[(u, v)] for u, v in edges], dtype=float)

    # map reverse edges
    edge_index = {(u, v): i for i, (u, v) in enumerate(edges)}
    rev = np.array(
        [edge_index.get((v, u), -1) for (u, v) in edges],
        dtype=np.int64
    )

    prior = np.array([prior_probs[u] for u in nodes], dtype=float)

    p_wo = prior[src].copy()
    p_wo_prev = np.zeros_like(p_wo)

    E = csr_matrix(
        (np.ones(m), (dst, np.arange(m))),
        shape=(n, m)
    )

    for _ in range(T):
        delta = p_wo - p_wo_prev

        q = 1.0 - p_edge * delta
        q = np.clip(q, eps, 1.0)

        log_q = np.log(q)
        log_prod = E.dot(log_q)
        prod_full = np.exp(log_prod)

        # exclude the correct reverse edge
        prod_cavity = prod_full[src].copy()
        mask = rev >= 0
        prod_cavity[mask] /= q[rev[mask]]

        prod_cavity = np.clip(prod_cavity, eps, 1.0)

        #p_wo_new = 1.0 - (1.0 - p_wo) * prod_cavity
        p_wo_new = p_wo + (1 - p_wo) * (1 - prod_cavity)


        p_wo_prev = p_wo
        p_wo = p_wo_new

    q_final = 1.0 - p_edge * p_wo
    q_final = np.clip(q_final, eps, 1.0)

    log_q = np.log(q_final)
    log_prod = E.dot(log_q)
    prod_final = np.exp(log_prod)

    p = 1.0 - (1.0 - prior) * prod_final
    return np.clip(p, 0.0, 1.0)

def dmp_python(G, prior_probs, edge_probs, T=10):
    # INITIALIZATION:
    p_without = {}
    for (u,v) in G.edges():
        p_without[(u,v)] = prior_probs[u]
    p = np.zeros(G.number_of_nodes())
        
    # MAIN LOOP:
    for _ in range(T):
        p_without_new = p_without.copy()
        for (u,v) in G.edges():
            influence = 1
            for w in G.neighbors(u):
                if w != v:
                    influence *= (1-edge_probs[(w,u)]*p_without[(w,u)])
            p_without_new[(u,v)] = 1 - (1-prior_probs[u])*influence
        p_without = p_without_new.copy()
        
    # FINAL VALUES:
    for u in G.nodes():
        influence = 1
        for v in G.neighbors(u):
            influence *= (1-edge_probs[(v,u)]*p_without[(v,u)])
        p[u] = 1 - (1-prior_probs[u])*influence

    return p


def dmp_inf(G, prior_probs, edge_probs, eps=1e-20, max_iter=100):
    """
    Dynamic Message Passing (DMP) for influence estimation in Independent Cascade model.
    
    Based on Algorithm 2 from: https://arxiv.org/pdf/1912.12749.pdf

    Parameters:
    - G (networkx.DiGraph): Directed graph
    - prior_probs (dict): Initial infection probabilities for each node
    - edge_probs (dict): Transmission probabilities for each edge (i, j)
    - T (int): Number of DMP iterations

    Returns:
    - pi (dict): Estimated infection probability of each node
    """
    # --- indexing
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {u: i for i, u in enumerate(nodes)}

    # --- build edge arrays
    edge_list = []
    for (u, v), prob in edge_probs.items():
        if u not in node_idx or v not in node_idx:
            continue
        edge_list.append((node_idx[u], node_idx[v], float(prob)))

    if len(edge_list) == 0:
        # no edges -> trivial result
        pi = {u: float(prior_probs.get(u, 0.0)) for u in nodes}
        return pi, float(sum(pi.values()))

    dtype = np.float64
    edge_arr = np.array(edge_list, dtype=[('src', int), ('dst', int), ('p', float)])
    src = edge_arr['src'].astype(np.int32)
    dst = edge_arr['dst'].astype(np.int32)
    p_e = edge_arr['p'].astype(dtype)
    m = len(edge_arr)

    # mapping (src,dst) -> edge index for reverse lookup
    pair_to_idx = {(int(src[i]), int(dst[i])): i for i in range(m)}
    rev = np.array([pair_to_idx.get((int(dst[i]), int(src[i])), -1) for i in range(m)], dtype=np.int32)

    # CSR matrix mapping node <- incoming edges
    # rows = node (dst), cols = edge index. Aq.dot(vec_edges) gives per-node sum/product in log-space
    Aq = csr_matrix((np.ones(m, dtype=dtype), (dst, np.arange(m, dtype=np.int32))), shape=(n, m), dtype=dtype)

    # prior p0 as array aligned with nodes
    p0 = np.zeros(n, dtype=float)

    # If it's a dict, direct lookup; otherwise assume array/tensor-like and index by node_idx
    if isinstance(prior_probs, dict):
        for u in nodes:
            p0[node_idx[u]] = float(prior_probs[u])
    else:
        # prior_probs[node_idx[u]] must work (e.g. numpy array, torch tensor)
        for u in nodes:
            p0[node_idx[u]] = float(prior_probs[node_idx[u]])


    # initialize messages: p_{i->j} (per edge). Common init: p0[src]
    m_e = p0[src].astype(dtype).copy()

    # main iteration
    for it in range(max_iter):
        # q_e = 1 - p_e * m_e  (edge-wise survival contribution)
        q_e = 1.0 - p_e * m_e
        # clip to avoid log(0) and negative due to rounding
        q_e = np.clip(q_e, eps, 1.0)

        # compute product over incoming edges per node via logs
        # prod_incoming[node] = prod_{e: dst==node} q_e[e]
        log_q = np.log(q_e)              # safe because clipped
        sum_log_per_node = Aq.dot(log_q) # shape (n,)
        prod_incoming = np.exp(sum_log_per_node)  # numeric-safe product

        # update messages for each edge e = (i -> j):
        # new_m_e = 1 - (1 - p0[i]) * prod_{l in N(i) \ {j}} (1 - p_{l i} * m_{l i})
        # We compute prod_all_incoming_at_i = prod_incoming[i], then exclude reverse edge (j->i) by dividing by q_rev
        # if reverse edge exists. If it doesn't, denom = 1.
        # q_rev_index = rev[e]
        denom = np.ones(m, dtype=dtype)
        mask_rev = rev != -1
        denom[mask_rev] = q_e[rev[mask_rev]]   # q at reverse edge (j->i); already clipped

        # safe division (denom clipped to >= eps)
        denom = np.clip(denom, eps, 1.0)

        prod_excl = prod_incoming[src] / denom

        new_m_e = 1.0 - (1.0 - p0[src]) * prod_excl

        # ensure messages in [0,1]
        new_m_e = np.clip(new_m_e, 0.0, 1.0)

        # check convergence
        delta = float(np.max(np.abs(new_m_e - m_e)))
        m_e = new_m_e
        if delta <= eps:
            break

    # final node probabilities: pi[i] = 1 - (1 - p0[i]) * prod_incoming[i]
    # recompute q_e and prod_incoming with final m_e to be consistent
    q_e = 1.0 - p_e * m_e
    q_e = np.clip(q_e, eps, 1.0)
    sum_log_per_node = Aq.dot(np.log(q_e))
    prod_incoming = np.exp(sum_log_per_node)
    pi_arr = 1.0 - (1.0 - p0) * prod_incoming
    pi_arr = np.clip(pi_arr, 0.0, 1.0)

    return pi_arr

def dmp_ic_fast(G, prior_probs, edge_probs, T=10, eps=1e-20):
    node_list = list(G.nodes())
    n = len(node_list)
    node_index = {u: i for i, u in enumerate(node_list)}

    # Convert edges to vector form
    edge_list = []
    for (u, v), prob in edge_probs.items():
        edge_list.append((node_index[u], node_index[v], prob))

    edge_list = np.array(edge_list, dtype=[('src', int), ('dst', int), ('p', float)])
    src = edge_list['src']
    dst = edge_list['dst']
    p = edge_list['p']
    m = len(edge_list)

    # Build reverse-edge index
    pair_to_idx = {(s, d): idx for idx, (s, d) in enumerate(zip(src, dst))}
    rev = np.array([pair_to_idx.get((dst[i], src[i]), -1) for i in range(m)], dtype=np.int32)

    # Prior infection probability
    p0 = np.zeros(n, dtype=np.float64)
    for u in node_list:
        p0[node_index[u]] = prior_probs[u]

    # Messages p_{i->j}
    p_ij = np.clip(p0[src].copy(), 0.0, 1.0)

    # CSR used to compute per-node inbound log-products
    Aq = csr_matrix((np.ones(m), (dst, np.arange(m))), shape=(n, m))

    for t in range(1,1+T):
        # q_e = 1 - p * p_ij, clipped into (0,1]
        q_e = 1.0 - p * p_ij
        q_e = np.clip(q_e, eps, 1.0)

        # incoming_q_prod[i] = product_{e incoming to i} q_e
        incoming_q_prod = np.exp(Aq.dot(np.log(q_e)))

        # qq[i] = (1 - p0[i]) * product incoming q_e
        qq = (1.0 - p0) * incoming_q_prod

        # Reverse q for edges, default is 1
        q_rev = np.ones(m, dtype=np.float64)
        mask = rev != -1
        q_rev[mask] = q_e[rev[mask]]
        q_rev = np.clip(q_rev, eps, 1.0)

        # update messages
        new_p_ij = 1.0 - qq[src] / q_rev

        # clamp to [0,1]
        p_ij = np.clip(new_p_ij, 0.0, 1.0)

    # Final probabilities
    q_e = 1.0 - p * p_ij
    q_e = np.clip(q_e, eps, 1.0)

    incoming_q_prod = np.exp(Aq.dot(np.log(q_e)))
    pi = 1.0 - (1.0 - p0) * incoming_q_prod

    return np.clip(pi, 0.0, 1.0)


def ALE_heuristic(G, prior_probs, edge_probs, num_steps):
    """
    Function for estimnating the independent cascade based on ALE model from:
    https://doi.org/10.14232/actacyb.21.1.2013.4

    
    G: networkx.DiGraph
    prior_probs: (num_nodes,) array of initial infection probabilities
    edge_probs: dict mapping (u, v) -> infection probability
    num_steps: int, number of diffusion steps
    """

    # ---- setup ----------------------------------------------------
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    src = []
    dst = []
    p = []

    for u, v in G.edges():
        src.append(idx[u])
        dst.append(idx[v])
        p.append(edge_probs.get((u, v), 0.0))

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    # state vectors
    x = np.asarray(prior_probs, dtype=np.float64).copy()
    current_x = x.copy()
    result = x.copy()

    # reusable buffer
    msg = np.zeros(n, dtype=np.float64)

    # ---- diffusion ------------------------------------------------
    for _ in range(1, num_steps):
        msg.fill(0.0)

        # propagate along edges
        np.add.at(msg, dst, current_x[src] * p)

        # clamp like the torch version
        np.clip(msg, 0.0, 1.0, out=msg)

        current_x = msg.copy()
        result += current_x

    return result

def modified_ALE(G, prior_probs, edge_probs, num_steps):
    """
    Improved function for estimnating the independent cascade based on ALE model from:
    https://doi.org/10.14232/actacyb.21.1.2013.4

    G: networkx.DiGraph
    prior_probs: (num_nodes,) array of initial infection probabilities
    edge_probs: dict mapping (u, v) -> infection probability
    num_steps: int, number of propagation steps
    """

    # ---- setup ----------------------------------------------------
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    # edge lists
    src = []
    dst = []
    p = []

    for u, v in G.edges():
        src.append(idx[u])
        dst.append(idx[v])
        p.append(edge_probs.get((u, v), 0.0))

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    # state vectors
    x = np.asarray(prior_probs, dtype=np.float64).copy()
    survival = 1.0 - x

    # reusable buffer
    msg = np.zeros(n, dtype=np.float64)

    # ---- propagation ----------------------------------------------
    for _ in range(num_steps):
        msg.fill(0.0)

        # accumulate incoming influence
        np.add.at(msg, dst, x[src] * p)

        # clamp to valid probabilities
        np.clip(msg, 0.0, 1.0, out=msg)

        # update survival and current activation
        survival *= (1.0 - msg)
        x = msg.copy()  # explicit copy avoids aliasing bugs

    return 1.0 - survival

def Naive(G, edge_probs, prior_probs, T, eps=1e-12):
    '''
    INITIALIZATION:
    p[i] = prior_probs[i] for i in G.nodes

    MAIN LOOP:
    for t=1...T:
        p_new = p
        for i in G.nodes:
            p_new[i] = 1 - (1-prior_probs[i])*\prod_{j->i}(1-edge_probs[j,i]*p[j])
        p = p_new

    RETURN:
    p
    '''
    # Node indexing
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {u: i for i, u in enumerate(nodes)}

    # Convert edges to array form
    edge_list = []
    for (u, v), p in edge_probs.items():
        edge_list.append((node_idx[u], node_idx[v], p))

    edge_arr = np.array(edge_list, dtype=[('src', int), ('dst', int), ('p', float)])
    src = edge_arr['src']
    dst = edge_arr['dst']
    p_e = edge_arr['p']
    m = len(src)

    # Build inbound-edge CSR: for node i, A_in[i, k] = 1 if edge k ends at i
    A_in = csr_matrix((np.ones(m), (dst, np.arange(m))), shape=(n, m))

    # Initialize p
    p = np.zeros(n, dtype=float)
    prior = np.zeros(n, dtype=float)

    for u in nodes:
        prior[node_idx[u]] = prior_probs[u]
        p[node_idx[u]] = prior_probs[u]

    # Time loop
    for _ in range(T):
        # Edge contributions: edge_probs[j->i] * p[j]
        contrib = p_e * p[src]
        contrib = np.clip(contrib, 0.0, 1.0 - eps)

        # q_e = 1 - contrib  (probability target NOT infected from src)
        q_e = np.clip(1.0 - contrib, eps, 1.0)

        # incoming product: ∏_{j→i} q_e
        incoming_log = A_in.dot(np.log(q_e))
        incoming_prod = np.exp(incoming_log)

        # Update p_new[i] = 1 - (1 - prior[i]) * incoming_prod
        p_new = 1.0 - (1.0 - prior) * incoming_prod

        p = p_new

    return np.clip(p, 0.0, 1.0)

def get_best_parameters(
    G,
    edge_probs,
    prior_probs,
    true_probs,
    method="SWE",
    eps=1e-12,
    max_t=10,
    min_t=1,
    max_layers=10,
):
    true_probs = np.asarray(true_probs, dtype=float)

    def rmse_fn(probs):
        return np.sqrt(np.mean((probs - true_probs) ** 2))
        
    best_t = min_t
    if method == "SWE":
        # ---- greedy over T
        best_rmse = 1
        best_t = 0
        best_l = 0
        for t in range(1,max_t+1):
            for l in range(max_layers+1):
                probs = SWE(G, edge_probs, prior_probs, t, a=1.0, layers=l, eps=eps)
                new_rmse = rmse_fn(probs)
                if new_rmse < best_rmse:
                    best_rmse = new_rmse
                    best_t = t
                    best_l = l
        return best_t, best_l
    elif method == "um_IC":
        # ---- greedy over T
        best_t = 0
        best_rmse = 1
        for t in range(1,max_t+1):
            new_rmse = rmse_fn(um_IC(G, edge_probs, prior_probs, t)[t])
            if new_rmse < best_rmse:
                best_t=t
                best_rmse = new_rmse
        return best_t
    elif method == "Naive":
        best_t = 0
        best_rmse = 1
        for t in range(1,max_t+1):
            new_rmse = rmse_fn(Naive(G, edge_probs, prior_probs, t))
            if new_rmse < best_rmse:
                best_t=t
                best_rmse = new_rmse
        return best_t
    elif method == "dmp_ic_fast":
        best_t = 0
        best_rmse = 1
        for t in range(1,max_t+1):
            new_rmse = rmse_fn(dmp_ic_fast(G, prior_probs, edge_probs, t))
            if new_rmse < best_rmse:
                best_t=t
                best_rmse = new_rmse
        return best_t
    elif method == "modified_ALE":
        best_t = 0
        best_rmse = 1
        for t in range(1,max_t+1):
            new_rmse = rmse_fn(modified_ALE(G, prior_probs, edge_probs, t))
            if new_rmse < best_rmse:
                best_t=t
                best_rmse = new_rmse
        return best_t
    elif method == "ALE_heuristic": 
        best_t = 0
        best_rmse = 1
        for t in range(1,max_t+1):
            new_rmse = rmse_fn(ALE_heuristic(G, prior_probs, edge_probs, t))
            if new_rmse < best_rmse:
                best_t=t
                best_rmse = new_rmse
        return best_t
    elif method == "dmp_um_fast":
        best_t = 0
        best_rmse = 1
        for t in range(1,max_t+1):
            new_rmse = rmse_fn(dmp_um_fast(G, prior_probs, edge_probs, t, eps=1e-20))
            if new_rmse < best_rmse:
                best_t=t
                best_rmse = new_rmse
        return best_t
    else:
        return -1, -1

def SWE(
    G, edge_probs, prior_probs, T, a=1.0, layers=0, eps=1e-12
):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}

    # Sparse adjacency: A[target, source] = p_{source->target}
    row, col, data = [], [], []
    for (u, v), p in edge_probs.items():
        row.append(idx[v])
        col.append(idx[u])
        data.append(p * a)
     
    A = csr_matrix((data, (row, col)), shape=(n, n))

    edge_probabilities = np.array(data)
    # Time-wise activation probabilities
    #P = np.zeros((T + 1, n), dtype=float)
    #P[0] = np.asarray(prior_probs, dtype=float)
    P = np.asarray(prior_probs, dtype=float)

    product_term = np.ones(n, dtype=float)

    for t in range(1, T+1):
        # edge-wise p * P[t-1]
        #vals = A.multiply(P[t - 1]).tocsr()
        vals = A.multiply(P).tocsr()
        vals.data = np.clip(vals.data, 0.0, 1.0 - eps)

        # log(1 - p * P)
        logs = np.log1p(-vals.data)
        logs_sparse = csr_matrix(
            (logs, vals.indices, vals.indptr), shape=vals.shape
        )

        # product over incoming neighbors
        log_prod = logs_sparse.sum(axis=1).A1
        prod_term = np.exp(log_prod)

        # incremental survival term
        #product_term *= (1.0 - P[t - 1])
        product_term *= (1.0 - P)

        # exact IC update
        #P[t] = product_term * (1.0 - prod_term)
        P = product_term * (1.0 - prod_term)

    # cumulative activation
    #probs = 1.0 - product_term*(1-P[T])#np.prod(1.0 - P, axis=0)
    probs = 1.0 - product_term*(1-P)
    
    # optional neighborhood refinement (unchanged)
    for l in range(layers):
        vals = A.multiply(probs).tocsr()
        vals.data = np.clip(vals.data, 0.0, 1.0 - eps)

        logs = np.log1p(-vals.data)
        logs_sparse = csr_matrix(
            (logs, vals.indices, vals.indptr), shape=vals.shape
        )

        log_prod = logs_sparse.sum(axis=1).A1
        neighbor_term = np.exp(log_prod)

        probs = 1.0 - (1.0 - prior_probs) * neighbor_term
    
    return np.clip(probs, 0.0, 1.0)

def um_IC(G, edge_probs, prior_B, T, eps=1e-15):
    '''
    Based on the IC approximation in:
    @inproceedings{inproceedings,
        author = {Srivastava, Ajitesh and Chelmis, Charalampos and Prasanna, V.},
        year = {2014},
        month = {08},
        pages = {451-454},
        title = {Influence in social networks: A unified model?},
        doi = {10.1109/ASONAM.2014.6921624}
    }
    '''
    n = G.number_of_nodes()
    node_list = list(G.nodes())
    idx = {u: i for i, u in enumerate(node_list)}

    # Sparse adjacency matrix: P[target, source] = p_{source->target}
    row, col, data = [], [], []
    for (src, tgt), p in edge_probs.items():
        row.append(idx[tgt])
        col.append(idx[src])
        data.append(p)
    P = csr_matrix((data, (row, col)), shape=(n, n))

    B = np.zeros((T+1, n))
    B[0] = np.asarray(prior_B, dtype=float)

    for t in range(1, T+1):
        B_prev = B[t-1]
        B_prevprev = B[t-2] if t-2 >= 0 else np.zeros(n)
        A_prev = B_prev - B_prevprev
    
        # edgewise infection values
        vals = P.multiply(A_prev).tocsr()        # force CSR
        vals.data = np.clip(vals.data, 0, 1 - eps)
    
        # log(1 - p * A_prev) on edges
        logs_data = np.log1p(-vals.data)
    
        # rebuild sparse with logs
        logs_sparse = csr_matrix((logs_data, vals.indices, vals.indptr), shape=vals.shape)
    
        # row sums = log product
        log_prod = logs_sparse.sum(axis=1).A1
        prod_term = np.exp(log_prod)
    
        B[t] = 1.0 - (1.0 - B_prev) * prod_term


    return B

def fixed_point_probs(G, edge_probs, prior_probs, max_iter=1000, tol=1e-9, eps=1e-15):
    """
    Solve p' = 1 - (1 - p) * ∏_{v in N(u)} (1 - p_{vu} * p'_v)
    by fixed-point iteration.
    
    Parameters
    ----------
    G : networkx.Graph or DiGraph
    edge_probs : dict {(u,v): p_uv}
        Edge transmission probabilities.
    prior_probs : dict {u: p_u(0)}
        Initial infection probs.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance (L∞ norm).
    eps : float
        Numerical safety margin.

    Returns
    -------
    p : dict {u: p_u'}
        Converged probabilities.
    """
    
    node_list = list(G.nodes())
    n = len(node_list)
    idx = {u: i for i,u in enumerate(node_list)}

    src, tgt, probs = [], [], []
    for (u,v), p in edge_probs.items():
        src.append(idx[u])
        tgt.append(idx[v])
        probs.append(p)
    src = np.array(src, dtype=int)
    tgt = np.array(tgt, dtype=int)
    probs = np.array(probs, dtype=float)

    p0 = np.array([prior_probs[u] for u in node_list], dtype=float)

    # initialize with prior
    p = p0.copy()

    for _ in range(max_iter):
        edge_vals = probs * p[src]
        edge_vals = np.clip(edge_vals, 0, 1 - eps)
        log_terms = np.log1p(-edge_vals)

        log_prod = np.zeros(n)
        np.add.at(log_prod, tgt, log_terms)
        prod_term = np.exp(log_prod)

        new_p = 1 - (1 - p0) * prod_term

        if np.max(np.abs(new_p - p)) < tol:
            p = new_p
            break
        p = new_p
    
    return p