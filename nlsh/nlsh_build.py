from build_pipeline import build_undirected_adj, build_csr, invert_partition, write_inverted, load_knn_file, run_kahip
from my_types import BuildInput


def main():
    build_input = BuildInput.parse_args()
    knn = load_knn_file(build_input.input_file)
    adj = build_undirected_adj(knn)
    vwgt, xadj, adjcwgt, adjncy = build_csr(adj)
    # Step 4: Call KaHIP
    blocks, edgecut = run_kahip(vwgt, xadj, adjcwgt, adjncy, build_input.members, build_input.imbalance,
                                build_input.seed, build_input.kahip_mode.value)

    print(f"Edge cut: {edgecut}")

    # Step 5: Inverted file
    inv = invert_partition(blocks, build_input.members)
    write_inverted(inv, "inverted_file.txt")


if __name__ == "__main__":
    main()
