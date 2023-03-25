import torch
from torch.utils.data import Dataset
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, minmax_scale, LabelEncoder
from utils import log_transform
import pandas as pd

class GeneExpDataset(Dataset):

    def __init__(self, data_path: str, type: str = 'train'):
        self.path = data_path
        with open(data_path, 'rb') as file:
            dataset = pkl.load(file) 

        X = dataset.drop(columns=['expert_PAM50_subtype', 'tcga_id', 'sample_id', 'cancer_type'], inplace=False)
        y = dataset.expert_PAM50_subtype

        self.X = X

        # LE = LabelEncoder()
        # self.y = LE.fit_transform(y)
        # self.le = LE

        self. y = pd.get_dummies(y).values
        self.length = self.X.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return minmax_scale(self.X.iloc[index, :].values), self.y[index]

        
class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index])

    def __len__(self):
        return len(self.dataset)
    

