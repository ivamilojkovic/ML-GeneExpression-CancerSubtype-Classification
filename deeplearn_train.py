import torch
import torch.nn as nn
from torch.optim import Adam
from deeplearn_models import AutoEncoder, CNNAutoEncoder
import pickle as pkl
from torch.utils.data import DataLoader, Subset
from dataset import GeneExpDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import pickle


def test_AE(test_loader, ae_model):

    total_mse = []
    for x, y in test_loader:

        out = ae_model(x)
        mse = mean_squared_error(out.detach().numpy(), x.detach().numpy())
        total_mse.append(mse)

    avg_mse = sum(total_mse)/len(total_mse)
    return avg_mse, total_mse

def test_class(test_loader, new_model):
    total_acc = []
    for x, y in test_loader:
        out = new_model(x.unsqueeze(1).to(torch.float32))[0,:,:]
        gt_class = (y == 1).nonzero(as_tuple=True)[1]
        acc = (out.argmax(1) == gt_class).sum().item()
        total_acc.append(acc)

    acc = sum(total_acc)/len(total_acc)
    return acc, total_acc



if __name__=='__main__':

    AE_TRAIN = False

     # Load the dataset
    DATASET_PATH = "tcga_brca_raw_19036_1053samples.pkl"
    dataset = GeneExpDataset(data_path=DATASET_PATH)

    # Check model dimensions
    x_check = torch.randn(10, 1, 25150)
    model = CNNAutoEncoder()
    if model(x_check).shape==torch.Size([10,1,25150]):
        print('Model is well defined!')

    # Split into train, test and validation sets
    train_idx, test_idx = train_test_split(range(dataset.length), test_size=0.3, 
                                           random_state=42, stratify=dataset.y)
    
    # Standardize the train (and val) sets
    min_train = dataset.X.iloc[train_idx, :].min().min()
    max_train = dataset.X.iloc[train_idx, :].max().max()

    # Minmax standardization
    dataset.X.iloc[train_idx, :] = (dataset.X.iloc[train_idx, :] - min_train) / (max_train - min_train)

    train_idx, val_idx = train_test_split(train_idx, test_size=0.3, 
                                          random_state=42, stratify=dataset.y[train_idx])
    
    # Minmax standardization applied on test set
    dataset.X.iloc[test_idx, :] = (dataset.X.iloc[test_idx, :] - min_train) / (max_train - min_train)
    
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    
    # Define loaders
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_set, shuffle=False)

    # Define a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Available device: ', device)

    model_params = {
        "input_shape": train_set.dataset.X.shape[1],
        "hidden_shape": 1024,
        "latent_shape": 128,
        "output_shape": train_set.dataset.X.shape[1]
    }
    clf_params = {
        "hidden_shape": 64,
        "latent_shape": 128,
        "out_shape": 5 # number of classes
    }

    # TODO: Early stopping

    # Ask if autoencoder should be trained:
    if AE_TRAIN:
        model = CNNAutoEncoder()
        model.to(device)

        optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=0.1)
        criterion = nn.MSELoss()

        min_val_loss = np.inf

        for epoch in tqdm(range(50), desc='Progress Epoch'):

            batch_loss = 0.0
            batch_val_loss = 0.0

            for x, y in train_loader:
                x = x.unsqueeze(1)
                out = model(x)
                loss = criterion(out.float(), x.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            model.eval()
            for x_val, y_val in val_loader:

                x_val = x_val.unsqueeze(1)
                val_out = model(x_val)
                loss = criterion(val_out.float(), x_val.float())
                batch_val_loss += loss.item()
            
            print("Training batch loss in epoch: ", batch_loss/len(train_loader), epoch)
            print("Validation batch loss in epoch: ", batch_val_loss/len(val_loader), epoch)

            if batch_val_loss < min_val_loss:
                min_val_loss = batch_loss
                best_model = model

        # Test
        test_results, _ = test_AE(test_loader, best_model)
        print('Autoencoder test results {}'.format(test_results))

        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

    # Do only classification
    else:
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)

    # Previous 
    features = nn.ModuleList(best_model.children())[:-1]
    model_feat = nn.Sequential(*features) 

    clf_layers = nn.Sequential(
        nn.Linear(clf_params["latent_shape"], clf_params["hidden_shape"]*2),
        nn.ReLU(True), 
        nn.Linear(clf_params["hidden_shape"]*2, clf_params["hidden_shape"]), 
        nn.ReLU(True), 
        nn.Linear(clf_params["hidden_shape"], clf_params["out_shape"]),
        nn.Softmax(dim=2))
    
    new_model = nn.Sequential(model_feat, clf_layers)
    new_model.to(device)
    #clf = EmbeddingClassifier(**clf_params)
    #clf.to(device)

    optimizer = Adam(new_model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(50), desc='Progress of Classification Epoch'):

        batch_loss, acc = 0.0, 0.0

        for x, y in train_loader:

            out = new_model(x.unsqueeze(1).to(torch.float32))
            loss = criterion(out.squeeze().float(), y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            acc += (out.argmax(1) == y).sum().item()

    acc = test_class(test_loader, new_model)
    print(acc)

    

    

    












   





