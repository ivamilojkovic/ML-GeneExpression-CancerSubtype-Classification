import torch.nn as nn
import torch
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], 
                                              out_features=kwargs["hidden_shape"])
        self.drop1 = nn.Dropout(p=0.5)
        self.latent_layer = nn.Linear(in_features=kwargs["hidden_shape"], 
                                      out_features=kwargs["latent_shape"])
        self.drop2 = nn.Dropout(p=0.5)
        self.decoder_hidden_layer = nn.Linear(in_features=kwargs["latent_shape"], 
                                              out_features=kwargs["hidden_shape"])
        self.drop3 = nn.Dropout(p=0.5)
        self.decoder_output_layer = nn.Linear(in_features=kwargs["hidden_shape"], 
                                              out_features=kwargs["output_shape"])        

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.encoder_hidden_layer(x)
        x = self.drop1(x)
        x = F.relu(x)
        x = self.latent_layer(x)
        x = self.drop2(x)
        x = F.relu(x)
        x = self.decoder_hidden_layer(x)
        x = self.drop3(x)
        x = F.relu(x)
        x = self.decoder_output_layer(x)

        return x