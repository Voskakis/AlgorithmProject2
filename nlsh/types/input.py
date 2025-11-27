from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enums import EndianType
from enums.kahip_modes import KahipMode


@dataclass
class BuildInput:
    input_file: Path
    index_path: Path
    type: EndianType
    knn_neighbors: Optional[int] = 10
    members: Optional[int] = 100
    imbalance: Optional[float] = 0.03
    kahip_mode: Optional[KahipMode] = 2
    layers: Optional[int] = 3
    nodes: Optional[int] = 64
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 128
    learn_rate: Optional[float] = 0.001
    seed: Optional[int] = 1

@dataclass
class SearchInput:
    input_file: Path
    query_file: Path
    index_path: Path
    output_file: Path
    type: EndianType
    nearest_neighbors: Optional[int] = 1
    search_radius: Optional[float] = 2800 if type == EndianType.Sift else 2000
    bins_check: Optional[int] = 5
    range: Optional[bool] = True

