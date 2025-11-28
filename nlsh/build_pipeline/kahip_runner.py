import kahip


def run_kahip(vwgt, xadj, adjcwgt, adjncy, nblocks, imbalance, seed, mode):
    """
    Calls the KaHIP library function kaffpa.
    Returns: blocks (list[int]), edgecut (int)
    """
    blocks = [0] * len(vwgt)
    edgecut = 0

    kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, nblocks, imbalance, False, seed, mode, edgecut, blocks)

    return blocks, edgecut
