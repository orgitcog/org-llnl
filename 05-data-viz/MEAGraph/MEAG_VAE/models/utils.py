import torch
import numpy as np
from MEAG_VAE.config import cfg
import math
import torch.nn.functional as F

def edge_ptr(batched_data):
    """
    Return the partition of edge_index for a mini-batch PyG data.

    Args:
        batched_data (torch_geometric.data.Batch): Batched PyG data.

    Returns:
        list: List of ranges representing the partition of edge_index.
    """
    ptr = []
    s = 0
    for i in range(batched_data.num_graphs):
        e = s + batched_data[i].edge_index.shape[1]
        ptr.append(range(s, e))
        s = e
    return ptr

def edge_reduction(edge_index, ptr_e, attention_list, rate):
    """
    Perform edge reduction based on attention scores.

    Args:
        edge_index (torch.Tensor): Edge indices.
        ptr_e (list): List of ranges representing the partition of edge_index.
        attention_list (torch.Tensor): Attention scores for each edge.
        rate (float): Rate of edges to keep.

    Returns:
        torch.Tensor: Reduced edge indices.
    """
    remaining_edge_index = []
    for i in range(len(ptr_e) - 1):
        edge_index_T = torch.transpose(edge_index, 0, 1)
        edge_index_temp = edge_index_T[ptr_e[i]:ptr_e[i+1], :]
        att_list_temp = attention_list[ptr_e[i]:ptr_e[i+1]]
        att_idx = torch.argsort(att_list_temp)
        num_edges = edge_index_temp.size(0)
        num_low_edges = int(num_edges * (1 - rate)) * 2
        delete_idx = att_idx[:num_low_edges]
        mask = torch.ones(num_edges, dtype=torch.bool)
        mask[delete_idx] = False
        remaining_edge_index_temp = edge_index_temp[mask, :].detach().cpu().tolist()
        remaining_edge_index.extend(remaining_edge_index_temp)
    remaining_edge_index = torch.transpose(torch.tensor(remaining_edge_index, dtype=torch.long, device=cfg.device), 0, 1)
    return remaining_edge_index


def edge_reduction_score_mean(edge_index, att_list, rate):
    """
    Perform edge reduction for edges with attention scores below its average values.

    Args:
        edge_index (torch.Tensor): Edge indices.
        att_list (torch.Tensor): Attention scores for each edge.
        rate (float): Rate of edges to keep.

    Returns:
        torch.Tensor: Reduced edge indices.
    """
    # Remove self-loops from edge_index and att_list
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    att_list = att_list[mask]

    # If no edges remain after removing self-loops, return the original edge_index
    if edge_index.shape[1] == 0:
        return edge_index
    
    # Normalize attention scores
    att_min, att_max = torch.min(att_list), torch.max(att_list)
    att_list_normalized = ((att_list - att_min) + 1e-4) / ((att_max - att_min) + 1e-4)
    # Keep edges with attention scores above the mean
    mask_keep = att_list_normalized > torch.mean(att_list_normalized)
    edge_index_reduced = edge_index[:, mask_keep]
    return edge_index_reduced

def edge_reduction_score_adaptive(edge_index, att_list, fixed_rate=-1.0):
    """
    Perform adaptive edge reduction based on attention scores and an optional fixed rate.

    Args:
        edge_index (torch.Tensor): Edge indices.
        att_list (torch.Tensor): Attention scores for each edge.
        fixed_rate (float, optional): Fixed rate of edges to keep. Defaults to None.

    Returns:
        torch.Tensor: Reduced edge indices.
    """
    # Remove self-loops from edge_index and att_list
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    att_list = att_list[mask]
    # If no edges remain after removing self-loops, return the original edge_index
    if edge_index.shape[1] == 0:
        return edge_index
    
    # Normalize attention scores
    att_min, att_max = torch.min(att_list), torch.max(att_list)
    att_list_normalized = ((att_list - att_min) + 1e-5) / ((att_max - att_min) + 1e-5)
    
    # Determine the rate for edge reduction
    if fixed_rate < 0.0 :
        rate_l = torch.rand(()).item() + 1e-5
    else:
        rate_l = fixed_rate
        
    # Keep edges with attention scores above the rate
    mask_keep = att_list_normalized > rate_l
    edge_index_reduced = edge_index[:, mask_keep]
    return edge_index_reduced

def edge_reduction_recover(edge_index, ptr_e, attention_list, rate):
    """
    Perform edge reduction and recovery based on attention scores.

    Args:
        edge_index (torch.Tensor): Edge indices.
        ptr_e (list): List of ranges representing the partition of edge_index.
        attention_list (torch.Tensor): Attention scores for each edge.
        rate (float): Rate of edges to keep.

    Returns:
        torch.Tensor: Reduced and recovered edge indices.
    """
    remaining_edge_index = []
    edge_index_T = torch.transpose(edge_index, 0, 1)
    for i in range(len(ptr_e) - 1):
        edge_index_temp = edge_index_T[ptr_e[i]:ptr_e[i+1], :].detach().cpu().numpy()
        att_list_temp = attention_list[ptr_e[i]:ptr_e[i+1]].detach().cpu().numpy()
        
        if edge_index_temp.shape[0] == 0:
            continue
        edge_index_sorted, att_list_sorted = zip(*sorted(zip(edge_index_temp, att_list_temp), key=lambda x: x[1]))
        
        num_tot_edges = edge_index_temp.shape[0]
        num_remove_edges = int(num_tot_edges * (1 - rate))
        edge_index_reduced = list(edge_index_sorted[num_remove_edges:])
        edge_index_recover = []

        if num_remove_edges > 0:
            last_att_value = att_list_sorted[num_remove_edges]
            att_sim_removed = [float(i / last_att_value) for i in att_list_sorted[:num_remove_edges]]
            sim_threshold = 0.95
            sim_limit_idx = find_idx_beyond_sim_threshold(att_sim_removed, sim_threshold)
            if sim_limit_idx is not None:
                edge_index_recover = list(edge_index_sorted[sim_limit_idx:num_remove_edges])
        
        remaining_edge_index.extend(edge_index_reduced)
        remaining_edge_index.extend(edge_index_recover)
    remaining_edge_index = np.array(remaining_edge_index)
    remaining_edge_index = torch.transpose(torch.tensor(remaining_edge_index, dtype=torch.long, device=cfg.device), 0, 1)
    return remaining_edge_index

def find_idx_beyond_sim_threshold(att_sim_removed, sim_threshold):
    """
    Find the index beyond a similarity threshold.

    Args:
        att_sim_removed (list): List of attention similarity scores.
        sim_threshold (float): Similarity threshold.

    Returns:
        int or None: Index beyond the similarity threshold, or None if not found.
    """
    if att_sim_removed[-1] < 0.95:
        return None
    left, right = 0, len(att_sim_removed)
    while left < right:
        mid = left + (right - left) // 2
        if att_sim_removed[mid] < sim_threshold:
            left = mid + 1
        else:
            right = mid
    return left

def build_graph(data, fixed_rate_l, fixed_rate_r):
    """
    Build a graph from input data using fixed rates for edge selection.

    Args:
        data (torch_geometric.data.Data): Input data.
        fixed_rate_l (float): Lower rate for edge selection.
        fixed_rate_r (float): Upper rate for edge selection.

    Returns:
        tuple: Tuple containing unique nodes, new feature matrix, and edge indices.
    """
    x = data.x
    x = F.normalize(x, p=2, dim=0)
    
    sim_amat = torch.cdist(x, x, p=2)
    exp_sim = torch.exp(-sim_amat)

    # Create a mask excluding diagonal elements
    diag_mask = ~torch.eye(exp_sim.size(0), dtype=bool)
    exp_sim_noself = exp_sim[diag_mask]

    # Normalize exp_sim_noself
    exp_sim_min, exp_sim_max = torch.min(exp_sim_noself), torch.max(exp_sim_noself)
    exp_sim_normalized = (exp_sim - exp_sim_min + 1e-5) / (exp_sim_max - exp_sim_min + 1e-5)

  
    rate_l, rate_r = fixed_rate_l, fixed_rate_r
 
    # Create mask based on normalized similarity scores
    mask_keep = (exp_sim_normalized > rate_l) & (exp_sim_normalized < rate_r)

    # Get unique nodes and reset edge_index indices
    edge_index = torch.nonzero(mask_keep, as_tuple=False).t()
    unique_nodes, edge_index = torch.unique(edge_index.flatten(), return_inverse=True)
    edge_index = edge_index.view(2, -1)

    # Use the unique nodes to get a new normalized feature matrix
    new_x = F.normalize(x[unique_nodes], p=2, dim=0)

    return unique_nodes, new_x, edge_index