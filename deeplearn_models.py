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
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=4),
           
        )

        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


    

        
