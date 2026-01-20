"""Compressed stats for bincfg CFG objects"""
import numpy as np
import pickle
import sys
import math
from bincfg import AtomicTokenDict, EdgeType


# Info for the get_compressed_stats() method of cfg's
GRAPH_LEVEL_STATS_FUNCS = [
    lambda cfg: cfg.num_blocks,
    lambda cfg: cfg.num_functions,
    lambda cfg: cfg.num_asm_lines,
]
GLS_DTYPE = np.uint32
NODE_SIZE_HIST_BINS: 'list[int]' = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200]
FUNCTION_DEGREE_HIST_BINS: 'list[int]' = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200]
FUNCTION_SIZE_HIST_BINS: 'list[int]' = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200, 300, 400, 500]


def get_compressed_stats(cfg, tokens, ret_pickled=False):
    """Returns some stats about this CFG in a compressed version

    These are meant to be very basic stats useful for simple comparisons (EG: dataset subsampling). These values
    are highly compressed/convoluted as they are used for generating statistics on 100+ million cfg's on HPC, and
    thus output space requirements outweigh single-graph compute time. Will return a single numpy array
    (1-d, dtype=np.uint8) with indices/values:

        - [0:12]: graph-level stats (number of nodes, number of functions, number of assembly lines), each a 4-byte
            unsigned integer of the exact value in the above order. The bytes are always stored as little-endian.

        - [12:20]: node degree histogram. Counts the number of nodes with degrees: 0 incomming, 1 incomming, 2 incomming,
            3+ incomming, 0 outgoing, 1 outgoing, 2 outgoing, 3+ outgoing. See below in things that are not
            in these stats for reasoning. Values will be a list in the above order:

            [0-in, 1-in, 2-in, 3+in, 0-out, 1-out, 2-out, 3+out]

            Reasoning: the vast majority of all nodes will have 0, 1 or 2 incomming normal edges, and 0, 1, or 2 outgoing
            normal edges, so this should be a fine way of storing that data for my purposes. Function call edges will
            be handled by the function degrees.

        - [20:46]: a histogram of node sizes (number of assembly lines per node). Histogram bins (left-inclusive,
            right-exclusive, 26 of them) will be:

            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200+]

            Reasoning: different compiler optimizations (inlining, loop unrolling, AVX instructions, etc.) will likely
            drastically change the sizes of nodes. The histogram bin edges were chosen arbitrarily in a way that tickled
            my non-neurotypical, nice-number-loving brain.

        - [46:72]: a histogram of (undirected) function degrees (in the function call graph). Histogram bins
            (left-inclusive, right-exclusive, 26 of them) will be:

            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200+]

            Reasoning: most functions will only be called a relatively small number of unique times across the
            binary (EG: <10), while those that are called much more are likely

        - [72:93]: a histogram of function sizes (number of nodes in each function). Histogram bins (left-inclusive,
            right-exclusive, 21 of them) will be:

            [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200, 300, 400, 500+]

            Reasoning: different compiler optimizations (especially inlining) will drastically change the size of
            functions. The histogram bins can be more spread out (IE: not as focused on values near 0, and across
            a larger range) since the number of nodes in a function has a noticeably different distribution than,
            say, the histogram of node sizes

        - [93:]: a histogram of assembly tokens. One value per token. You should make sure the normalization method
            you are using is good, and doesn't create too many unique tokens.

            Reasoning: obvious

    The returned array will be of varrying length based on the number of unique tokens in the tokens dictionary.

    Values above (unless otherwise stated) are stored as 1-byte percentages of the number of nodes in the graph that
    are in that bin. EG: 0 would mean there are 0 nodes with that value, 1 would mean between [0, 1/255) of the
    nodes/functions in the graph have that value, 2 would be between [1/255, 2/255), etc., until a 255 which would
    be [245/255, 1.0]

    Things that are NOT in these stats and reasons:

        - Other node degrees: these likely don't change too much between programs (specifically their normalized values)
            as even with different programs/compiler optimizations. Changes between cfg's will likely only change the
            relative proportions of nodes with 1 vs. 2 incoming edges, and those with 1 vs. 2 vs. 3 outgoing edges. Any
            other number of edges are pretty rare, hence why we only keep those edge measures and only those using
            normal edges (since function call edges will be gathered in the func_degree_hist and would mess with this premise)
        - Edge statistics (EG: number of normal vs. function call edges): this information is partially hidden in the
            histograms already present, and exact values do not seem too important
        - Other centrality measures: I belive (but have not proven) that node-based centrality measures would not
            contain enough information to display differences between CFG's to be worth it. Because of the linear
            nature of sequential programs, I believe their centrality measures would be largely similar and/or
            dependant on other graph features already present in the stats above (EG: number of nodes in a function).
            I think any differences between centrality measurements on these graphs will be mostly washed out by the
            linear nature, especially since we would only be looking at normal edges, not function call ones. The only
            differences that would be highlighted would be information about the number of branches/loops in each
            function (which is already partially covered by the assembly line info), and a small amount of information
            on where within functions these branches tend to occur. However, combining these features into graph-level
            statistics would likely dilute these differences even further. It may, however, be useful to include one
            or more of these measures on the function call graph, but I am on the fence about its usefulness vs extra
            computation time/space required. I think for my purposes, the stats above work just fine

    Args:
        tokens (Union[Dict[str, int], AtomicData]): the token dictionary to use and possibly add to. Can also be
            an AtomicData object for atomic token dictionary file.
        ret_pickled (bool): if True, will return the data pickled with pickle.dumps(), otherwise if False, will
            return the normal numpy array

    Returns:
        np.ndarray: the compressed stats, a numpy 1-d uint8 array of shape (97 + len(tokens), )
    """
    # Get the token histogram
    token_counts = cfg.asm_counts

    # Create the token dictionary and update it with the new tokens
    token_dict = tokens if tokens is not None else {}
    if isinstance(token_dict, AtomicTokenDict):
        token_dict.addtokens(*list(token_counts.keys()))
    else:
        for k in token_counts:
            token_dict.setdefault(k, len(token_dict))

    # Make sure this starts off a bit larger than the final length. I think the current val would be 97, but I made
    #   it a little larger for some more wiggle room. The correct size is returned anyways
    ret = np.zeros([100 + len(token_dict)], dtype=np.uint8)
    curr_idx = 0

    # Adding graph statistics as multi-byte unsigned integer values
    nb = GLS_DTYPE().nbytes
    for f in GRAPH_LEVEL_STATS_FUNCS:
        ret[curr_idx: curr_idx + nb] =  _get_np_int_as_little_endian_list(GLS_DTYPE(f(cfg)))
        curr_idx += nb

    # Get the node degree histograms
    in_0, in_1, in_2, in_other, out_0, out_1, out_2, out_other = 0, 0, 0, 0, 0, 0, 0, 0
    for block in cfg.blocks:
        n_in, n_out = block.get_sorted_edges(edge_types=EdgeType.NORMAL)

        # Check edges in
        if len(n_in) == 0:
            in_0 += 1
        elif len(n_in) == 1:
            in_1 += 1
        elif len(n_in) == 2:
            in_2 += 1
        else:
            in_other += 1

        # Check edges out
        if len(n_out) == 0:
            out_0 += 1
        elif len(n_out) == 1:
            out_1 += 1
        elif len(n_out) == 2:
            out_2 += 1
        else:
            out_other += 1

    num_blocks = cfg.num_blocks
    for v in [in_0, in_1, in_2, in_other, out_0, out_1, out_2, out_other]:
        ret[curr_idx] = _get_single_byte_ratio(v, num_blocks)
        curr_idx += 1

    # Get the node size histogram
    curr_idx = _get_single_byte_histogram([len(b.asm_lines) for b in cfg.blocks_dict.values()],
        NODE_SIZE_HIST_BINS, ret, curr_idx)

    # Get the function undirected degree histogram
    curr_idx = _get_single_byte_histogram([f.num_fc_edges for f in cfg.functions_dict.values()],
        FUNCTION_DEGREE_HIST_BINS, ret, curr_idx)

    # Get the function size histogram
    curr_idx = _get_single_byte_histogram([f.num_blocks for f in cfg.functions_dict.values()],
        FUNCTION_SIZE_HIST_BINS, ret, curr_idx)

    # Get the asm line histogram
    # Invert the token dict to get a mapping from index to asm line
    inv_token_dict = {v:k for k, v in token_dict.items()}
    num_asm_lines = sum(v for v in token_counts.values())
    ret[curr_idx: curr_idx + len(inv_token_dict)] = [_get_single_byte_ratio(token_counts[inv_token_dict[i]], num_asm_lines) for i in range(len(inv_token_dict))]
    curr_idx += len(inv_token_dict)

    ret = ret[:curr_idx]

    if ret_pickled:
        return pickle.dumps(ret)
    return ret

def uncompress_stats(stats, dtype=np.uint32):
    """Uncompressed the stats from cfg.get_compressed_stats()

    Will return a numpy array with specified dtype (defaults to np.uint32) of stats in the same order they appreared
    in get_compressed_stats(). The size will decrease by around 12 indices as the initial 4-byte values are converted
    back into a one-index integer.

    Args:
        stats (np.ndarray): either a 1-d or 2-d numpy array of stats. If 2-d, then it is assumed that these are multiple
            stats for multiple cfgs, one cfg per row
        dtype (np.dtype): the numpy dtype to return as. Defaults to np.uint32

    Returns:
        np.ndarray: either a 1-d or 2-d numpy array of uncompressed stats, depending on what was passed to `stats`
    """
    if stats.ndim not in [1, 2]:
        raise ValueError("`stats` array must have dimension 1 or 2, not %d" % stats.ndim)

    # Get the return array, removing the elements for the multi-byte ints. Determine what dimension to be using
    if stats.ndim == 2:
        ret = np.empty([stats.shape[0], stats.shape[1] - (len(GRAPH_LEVEL_STATS_FUNCS) * (GLS_DTYPE().nbytes - 1))], dtype=np.uint32)
        ret_one_dim = False
    else:
        ret = np.empty([1, stats.shape[0] - (len(GRAPH_LEVEL_STATS_FUNCS) * (GLS_DTYPE().nbytes - 1))], dtype=np.uint32)
        stats = [stats]
        ret_one_dim = True

    # Iterate through all rows in stats to uncompress
    for row_idx, stat_arr in enumerate(stats):
        stats_idx = ret_idx = 0

        # Unpack the multi-byte ints
        nb = GLS_DTYPE().nbytes
        for _ in GRAPH_LEVEL_STATS_FUNCS:
            ret[row_idx, ret_idx] = _get_np_int_from_little_endian_list(stat_arr[stats_idx: stats_idx + nb])
            stats_idx += nb
            ret_idx += 1

        num_blocks, num_functions, num_asm_lines = ret[row_idx, 0:3]

        # Unpack the histograms: node degrees, node sizes, function degrees, function sizes, asm lines
        ret_idx, stats_idx = _uncompress_hist(row_idx, ret, stat_arr[stats_idx: stats_idx + 8], ret_idx, stats_idx, num_blocks)
        ret_idx, stats_idx = _uncompress_hist(row_idx, ret, stat_arr[stats_idx: stats_idx + len(NODE_SIZE_HIST_BINS)], ret_idx, stats_idx, num_blocks)
        ret_idx, stats_idx = _uncompress_hist(row_idx, ret, stat_arr[stats_idx: stats_idx + len(FUNCTION_DEGREE_HIST_BINS)], ret_idx, stats_idx, num_functions)
        ret_idx, stats_idx = _uncompress_hist(row_idx, ret, stat_arr[stats_idx: stats_idx + len(FUNCTION_SIZE_HIST_BINS)], ret_idx, stats_idx, num_functions)
        ret_idx, stats_idx = _uncompress_hist(row_idx, ret, stat_arr[stats_idx:], ret_idx, stats_idx, num_asm_lines)

    # Return a 1-d if needed
    ret = ret.reshape([-1]) if ret_one_dim else ret
    return ret.astype(dtype)


def _get_np_int_as_little_endian_list(val):
    """returns a list of bytes for the given numpy integer in little-endian order"""
    ret = list(val.tobytes())

    # Get the byte order and check if we need to swap endianness
    bo = np.dtype(GLS_DTYPE).byteorder
    if (bo == '=' and sys.byteorder == 'big') or bo == '>':
        return reversed(ret)

    return ret


def _get_np_int_from_little_endian_list(l):
    """Returns a numpy integer from the given list of little-endian bytes

    NOTE: `l` MUST be either a python built-in (list/tuple/etc), or a numpy array with dtype np.uint8!
    """
    return int.from_bytes(l, byteorder='little', signed=False)


def _get_single_byte_ratio(val, total):
    """Computes the ratio val/total (assumes total >= val), and converts that resultant value to a byte

    The byte value will be determined based on what 'chunk' in the range [0, 1] the value is, with there being 256
    available chunks for one byte. EG: 0 would mean val == 0, 1 would mean val is between [0, 1/255), 2 would be between
    [1/255, 2/255), etc., until a 255 which would be between [245/255, 1].
    """
    assert total >= val, "Total was < val! Total: %d, val: %d" % (total, val)
    return 0 if total == 0 else math.ceil(val / total * 255)


def _get_single_byte_histogram(vals, bins, ret, curr_idx):
    """Does a full histogram thing

    Args:
        vals (Iterable[int]): the values to bin/histogram
        bins (Iterable[int]): the bins to use. Should start with the lowest value, and have right=False.
            EG: with bins [0, 3, 7, 9] and values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], they would be digitized into
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3]
        ret (np.ndarray): the values to insert into
        curr_idx (int): the index in ret to insert into

    Returns:
        int: the starting index in ret to continue inserting values
    """
    binned = np.digitize(vals, bins, right=False) - 1
    uniques, counts = np.unique(binned, return_counts=True)
    ret[uniques + curr_idx] = [_get_single_byte_ratio(c, len(binned)) for c in counts]
    return curr_idx + len(bins)


def _uncompress_hist(row_idx, ret, stats, ret_idx, stats_idx, val):
    """Uncompresses histogram values, and returns new ret_idx and stats_idx, with `val` being the value that was used
    for the percentages (IE: self.num_blocks). Stores uncompressed values into ret (a 2-d array)"""
    ret[row_idx, ret_idx: ret_idx + len(stats)] = np.ceil(stats * 1/255 * val)
    return ret_idx + len(stats), stats_idx + len(stats)



