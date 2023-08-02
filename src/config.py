from dataclasses import dataclass

@dataclass
class Paths:
    dataset: str
    ml_dataset: str
    experiment: str
    model: str
    images: str
    artefacts: str

@dataclass
class Train:
    solve_class_imbalance: bool
    type_class_imbalance: str
    thresh_lumA: int
    cross_val: bool
    test_size: float
    random_state: int
    random_state_split: int
    optim: bool
    grid_scoring: str
    downsample_test: bool
    num_folds: int
    num_feat: int
    type_feat_selection: str
    use_multilabel_dataset: bool

@dataclass
class ProjectConfig:
    paths: Paths
    train: Train