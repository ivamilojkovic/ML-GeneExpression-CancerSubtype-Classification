import torch.nn as nn
import torch
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(kwargs["input_shape"], kwargs["hidden_shape"]*2),
            nn.Dropout(p=0.3),
            nn.ReLU(True), 
            nn.Linear(kwargs["hidden_shape"]*2, kwargs["hidden_shape"]), 
            nn.Dropout(p=0.3),
            nn.ReLU(True), 
            nn.Linear(kwargs["hidden_shape"], kwargs["latent_shape"]),
            nn.Dropout(p=0.3))
        
        self.decoder = nn.Sequential(
            nn.Linear(kwargs["latent_shape"], kwargs["hidden_shape"]),
            nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(kwargs["hidden_shape"], kwargs["hidden_shape"]*2),
            nn.Dropout(p=0.3),
            nn.ReLU(True), 
            nn.Linear(kwargs["hidden_shape"]*2, kwargs["output_shape"]), 
            nn.Dropout(p=0.3)
            ) 
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
class CNNAutoEncoder(nn.Module):
    def __init__(self, attention=False):
        super(CNNAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
            )      
        self.attention = nn.MultiheadAttention(embed_dim=3143, num_heads=1, dropout=0.3, batch_first=True)  

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.encoder(x)
        x, _ = self.attention(x, x, x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, 25150, mode='linear')
        
        return x
    
class CNNAttentionEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(CNNAttentionEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
            )      
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, 
                                               num_heads=4, dropout=0.3,
                                               batch_first=True)  

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.encoder(x)
        x, _ = self.attention(x, x, x)

        return x
    
class Classifier(nn.Module):
    def __init__(self, **clf_params) -> None:
        super(Classifier, self).__init__()

        self.clf = nn.Sequential(
                nn.Linear(100608, clf_params["hidden_shape"]*2),
                nn.ReLU(True), 
                nn.Linear(clf_params["hidden_shape"]*2, clf_params["hidden_shape"]), 
                nn.ReLU(True), 
                nn.Linear(clf_params["hidden_shape"], clf_params["output_clf_shape"])
                )
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = nn.Flatten(start_dim=1)(x)
        x = self.clf(x)
        x = nn.functional.softmax(x, dim=1)
        return x
    
class CNNAttentionClassifier(nn.Module):
    def __init__(self, **model_params) -> None:
        super(CNNAttentionClassifier, self).__init__()

        self.encoder = CNNAttentionEncoder(model_params['latent_shape'])
        self.clf = Classifier(**model_params)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.encoder(x)
        x = self.clf(x)
        return x




    



    

        
