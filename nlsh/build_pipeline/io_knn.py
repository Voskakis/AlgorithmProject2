def load_knn_file(path: str) -> list[list[int]]:
    """
    Loads a 1-indexed KNN adjacency file.
    Format: first line ignored, each following line: space-separated neighbor IDs.
    """
    knn = [[]]  # index 0 unused

    with open(path) as f:
        next(f)  # skip header line
        for line in f:
            if line.strip():
                knn.append(list(map(int, line.split())))

    return knn
