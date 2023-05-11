from dataclasses import dataclass

@dataclass
class Paths:
    dataset: str
    experiment: str
    model: str
    images: str

@dataclass
class Train:
    solve_class_imbalance: bool
    type_class_imbalance: str
    cross_val: bool
    test_size: float
    random_state: int
    optim: bool
    grid_scoring: str
    downsample_test: bool
    num_folds: int
    num_feat: int
    type_feat_selection: str

@dataclass
class ProjectConfig:
    paths: Paths
    train: Train