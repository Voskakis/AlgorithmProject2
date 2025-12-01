from torch.utils.data import TensorDataset, DataLoader

from build_pipeline import load_knn_file, run_kahip, build_graph_items, produce_long_tensor, create_inverted_file
from my_types import BuildInput


def main():
    build_input = BuildInput.parse_args()
    knn = load_knn_file(build_input.input_file)

    adj_set, xadj, vwgt, adjcwgt, adjncy = build_graph_items(knn)

    blocks, edgecut = run_kahip(vwgt, xadj, adjcwgt, adjncy, build_input.members, build_input.imbalance,
                                build_input.seed, build_input.kahip_mode.value)

    # X_tensor = torch.from_numpy(knn).float() #TODO: not working
    y_tensor = produce_long_tensor(blocks)

    # dataset = TensorDataset(X_tensor, y_tensor)
    # loader = DataLoader(dataset, batch_size=build_input.batch_size, shuffle=True)

    create_inverted_file(blocks, build_input.members, build_input.index_path+"/inverted_file.txt")
if __name__ == "__main__":
    main()
