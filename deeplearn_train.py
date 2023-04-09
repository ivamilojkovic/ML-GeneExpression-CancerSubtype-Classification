import torch
import torch.nn as nn
from torch.optim import Adam
from deeplearn_models import *
import pickle as pkl
from torch.utils.data import DataLoader, Subset
from dataset import GeneExpDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import pickle
from configs import *

def test_AE(test_loader, ae_model):

    total_mse = []
    for x, y in test_loader:

        out = ae_model(x)
        mse = mean_squared_error(out.detach().numpy(), x.detach().numpy())
        total_mse.append(mse)

    avg_mse = sum(total_mse)/len(total_mse)
    return avg_mse, total_mse

def test_class(test_loader, model):
    y_matrix, out_matrix = [], []
    for x, y in test_loader:
        out = model(x.unsqueeze(1).to(torch.float32))
        gt_class = (y == 1).nonzero(as_tuple=True)[1]

        # Create output and y rows e.g. [0, 0, 1, 0, 0]
        out_row = [0]*5
        out_row[out.argmax(1)]=1
        y_row = y.numpy()[0]
        out_matrix.append(out_row)
        y_matrix.append(y_row)

    return np.stack(out_matrix), np.stack(y_matrix)

def create_save_loaders(dataset):

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

    with open('trainloader.pkl', 'wb') as f:
        pickle.dump(train_loader, f)
    with open('testloader.pkl', 'wb') as f:
        pickle.dump(test_loader, f)
    with open('valloader.pkl', 'wb') as f:
        pickle.dump(val_loader, f)

    return train_loader, test_loader, val_loader
  
def read_loaders():

    with open('trainloader.pkl', 'rb') as f:
        train_loader = pickle.load(f)
    with open('testloader.pkl', 'rb') as f:
        test_loader = pickle.load(f)
    with open('valloader.pkl', 'rb') as f:
        val_loader = pickle.load(f)
    
    return train_loader, test_loader, val_loader


if __name__=='__main__':

    CREATE_LOADERS = False

     # Load the dataset
    DATASET_PATH = "tcga_brca_raw_19036_1053samples.pkl"
    dataset = GeneExpDataset(data_path=DATASET_PATH)
    if CREATE_LOADERS:
        train_loader, test_loader, val_loader = create_save_loaders(dataset)
    else:
        train_loader, test_loader, val_loader = read_loaders()

    # Define a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Available device: ', device)

    ########################################################################    

    # TODO: Early stopping

    # Define model, optimizer and criterion
    model_params = model_ae_params
    model = CNNAttentionClassifier(**model_params)
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0.1)
    criterion = nn.MSELoss()

    min_val_loss = np.inf

    for epoch in tqdm(range(5), desc='Progress Epoch'):

        batch_loss = 0.0
        batch_val_loss = 0.0

        for x, y in train_loader:
            x = x.unsqueeze(1)
            out = model(x)
            loss = criterion(out.float(), y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        model.eval()
        for x_val, y_val in val_loader:

            x_val = x_val.unsqueeze(1)
            val_out = model(x_val)
            loss = criterion(val_out.float(), y_val.float())
            batch_val_loss += loss.item()
        
        print("\nTraining batch loss in epoch: ", batch_loss/len(train_loader), epoch)
        print("\nValidation batch loss in epoch: ", batch_val_loss/len(val_loader), epoch)

        if batch_val_loss < min_val_loss:
            min_val_loss = batch_loss
            best_model = model

    # Test scores
    y_pred, y_true = test_class(test_loader, best_model)
    recall = recall_score(y_pred, y_true, average='macro')
    prec = precision_score(y_pred, y_true, average='macro')
    f1 = f1_score(y_pred, y_true, average='macro')
    acc = accuracy_score(y_pred, y_true)

    print('Test results:\n')
    print("Accuracy: {}".format(acc))
    print("Recall unweighted: {}".format(recall))
    print("Precision unweighted: {}".format(prec))
    print("F1 score unweighted: {}".format(f1))

    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)



# # Previous 
# features = nn.ModuleList(best_model.children())[:-1]
# model_feat = nn.Sequential(*features) 

# clf_layers = nn.Sequential(
#     nn.Linear(clf_params["latent_shape"], clf_params["hidden_shape"]*2),
#     nn.ReLU(True), 
#     nn.Linear(clf_params["hidden_shape"]*2, clf_params["hidden_shape"]), 
#     nn.ReLU(True), 
#     nn.Linear(clf_params["hidden_shape"], clf_params["out_shape"]),
#     nn.Softmax(dim=2))

# new_model = nn.Sequential(model_feat, clf_layers)
# new_model.to(device)
# #clf = EmbeddingClassifier(**clf_params)
# #clf.to(device)

# optimizer = Adam(new_model.parameters(), lr=1e-4, weight_decay=0.01)
# criterion = nn.CrossEntropyLoss()

# for epoch in tqdm(range(50), desc='Progress of Classification Epoch'):

#     batch_loss, acc = 0.0, 0.0

#     for x, y in train_loader:
#         x = x.unsqueeze(1).to(torch.float32)
#         out = new_model(x)
#         loss = criterion(out.squeeze().float(), y.float())

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         batch_loss += loss.item()
#         acc += (out.argmax(1) == y).sum().item()

# acc = test_class(test_loader, new_model)
# print(acc)
























