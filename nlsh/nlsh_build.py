import subprocess
import uuid

import torch

from build_pipeline import run_kahip, build_graph_items, create_inverted_file
from build_pipeline.neural import MLPClassifier
from my_types import BuildInput


def main():
    build_input = BuildInput.parse_args()

    result = subprocess.check_output(
        ["./lsh", "-d", f"./{build_input.input_file}", "-q", f"./{build_input.input_file}", "-k", f"10", "-L", f"{build_input.batch_size}", "-N",
         f"{build_input.knn_neighbors + 1}", "-o", "output.txt"])
    knn = [list(map(int, line.split())) for line in result.splitlines()]
    counter = 0
    for i in knn:
        if counter in i:
            i.remove(counter)
        counter += 1
    adj_set, xadj, vwgt, adjcwgt, adjncy = build_graph_items(knn)

    blocks, edgecut = run_kahip(vwgt, xadj, adjcwgt, adjncy, build_input.members, build_input.imbalance,
                                build_input.seed, build_input.kahip_mode.value)

    # initialize classifier

    input_dim = 28 * 28  # TODO possibly different for SIFT
    num_classes = build_input.members
    model_wrapper = MLPClassifier(d_in=input_dim, n_out=num_classes, layers=build_input.layers, nodes=build_input.nodes)

    # Train the classifier
    model_wrapper.train_classifier(dataset=build_input.input_file, output=num_classes, epochs=build_input.epochs,
                                   batch_size=build_input.batch_size, lr=build_input.lr)
    # save model weights
    torch.save(model_wrapper.state_dict(), build_input.index_path + "/inverted_file.txt")
    # save inverted index
    create_inverted_file(blocks, build_input.members, build_input.index_path + "/model.pth")


if __name__ == "__main__":
    main()
