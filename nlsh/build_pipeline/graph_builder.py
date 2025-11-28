def build_undirected_adj(knn: list[list[int]]) -> list[set[int]]:
    """
    Turn directed KNN (1-indexed) into deduplicated, undirected neighbors.
    """
    node_count = len(knn) - 1
    adj = [set() for _ in range(node_count + 1)]

    for u in range(1, node_count + 1):
        for v in knn[u]:
            if v != u:
                adj[u].add(v)
                adj[v].add(u)

    return adj


def build_csr(adj: list[set[int]]):
    """
    Convert 1-indexed adjacency list (sets) into CSR arrays for KaHIP:
      - 0-based adjncy
      - xadj of length n+1
      - adjcwgt all = 1
      - vwgt all = 1
    """
    node_count = len(adj) - 1

    xadj = [0]
    adjncy = []
    adjcwgt = []
    vwgt = [1] * node_count

    for u in range(1, node_count + 1):
        neighbors = sorted(adj[u])
        for v in neighbors:
            adjncy.append(v - 1)
            adjcwgt.append(1)
        xadj.append(len(adjncy))

    return vwgt, xadj, adjcwgt, adjncy
