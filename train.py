import torch
import torch.nn as nn
from torch.optim import Adam
from models import AutoEncoder
from torch.utils.data import DataLoader
import pickle as pkl
from torch.utils.data import DataLoader
from dataset import GeneExpDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error


def test(test_loader, model):

    total_mse = []
    for x, _ in test_loader:

        out = model(x)
        mse = mean_squared_error(out.detach().numpy(), x.detach().numpy())
        total_mse.append(mse)

    avg_mse = sum(total_mse)/len(total_mse)
    return avg_mse, total_mse

if __name__=='__main__':

    # Load the dataset
    DATASET_PATH = "tcga_brca_raw_19036_1053samples.pkl"

    train_set = GeneExpDataset(data_path=DATASET_PATH, type='train')
    val_set = GeneExpDataset(data_path=DATASET_PATH, type='val')
    test_set = GeneExpDataset(data_path=DATASET_PATH, type='test')
    
    # Convert to DataFrames to Tensors
    
    # Define loaders
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_set, shuffle=False)

    # Define a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Available device: ', device)

    model_params = {
        "input_shape": train_set.X_train.shape[1],
        "hidden_shape": 512,
        "latent_shape": 128,
        "output_shape": train_set.X_train.shape[1]
    }

    model = AutoEncoder(**model_params)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()

    min_val_loss = np.inf

    for epoch in tqdm(range(100), desc='Progress Epoch'):

        batch_loss = 0.0
        batch_val_loss = 0.0

        for x, _ in train_loader:

            out = model(x)
            loss = criterion(out.float(), x.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        model.eval()
        for x_val, _ in val_loader:

            val_out = model(x_val)
            loss = criterion(val_out.float(), x_val.float())
            batch_val_loss += loss.item()
        
        print("Training batch loss in epoch: ", batch_loss/len(train_loader), epoch)
        print("Validation batch loss in epoch: ", batch_val_loss/len(val_loader), epoch)

        if batch_val_loss < min_val_loss:
            min_val_loss = batch_loss
            best_model = model

    # Test
    test_results, _ = test(test_loader, best_model)
    print(test_results)
