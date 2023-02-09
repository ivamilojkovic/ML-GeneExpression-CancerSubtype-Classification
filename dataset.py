import torch
from torch.utils.data import Dataset
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GeneExpDataset(Dataset):

    def __init__(self, data_path: str, type: str = 'train'):
        self.apth = data_path
        self.type = type
        with open(data_path, 'rb') as file:
            dataset = pkl.load(file) 

        X = dataset.drop(columns=['expert_PAM50_subtype', 'tcga_id', 'sample_id', 'cancer_type'], inplace=False)
        y = dataset.expert_PAM50_subtype

        # Split the dataset into train, test and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, 
                                                            shuffle=True, stratify=y)  

        # Data standardization ONLY for train set
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.3, random_state=1, 
                                                        shuffle=True, stratify=y_train)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test_scaled, y_test

    def __len__(self):

        if self.type == 'train':
            return self.X_train.shape[0]
        elif self.type == 'val':
            return self.X_val.shape[0]
        elif self.type == 'test':
            return self.X_test.shape[0]
        else:
            print('Not a valid type!')
    
    def __getitem__(self, index):
        if self.type == 'train':
            return self.X_train[index, :], self.y_train.iloc[index]
        elif self.type == 'val':
            return self.X_val[index, :], self.y_val.iloc[index]
        elif self.type == 'test':
            return self.X_test[index, :], self.y_test.iloc[index]
        else:
            print('Not a valid type!')
        


    

