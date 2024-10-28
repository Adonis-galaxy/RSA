import torch
import torch.nn as nn
import math

FACTOR = 1
HEIGHT = 1
WIDTH = 1

class LanScaleModel(nn.Module):
    '''
    LanScaleModel Network class

    Arg(s):
        text_feat_dim: int
            dimension of input CLIP text feature
    '''
    def __init__(self, text_feat_dim=1024):
        super().__init__()
        self.scene_feat_net = nn.Sequential(
            nn.Linear(text_feat_dim, 512),
            # nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(),
            # nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256)
        )

        self.shift_net = nn.Sequential(
            # nn.Linear(256, 256),
            # nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            # nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, FACTOR)
        )

        self.scale_net = nn.Sequential(
            # nn.Linear(256, 256),
            # nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            # nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, FACTOR)
        )

    def forward(self, text_feat):
        '''
        Forwards the inputs through the network

        Arg(s):
            text_feat: torch.Tensor[float32]
                N x text_feat_dim(1024 by default)
        Returns:
            shift_pred: torch.Tensor[float32]
                N x 1
            scale_pred: torch.Tensor[float32]
                N x 1
        '''
        scene_feat = self.scene_feat_net(text_feat)

        scale_pred = torch.exp(self.scale_net(scene_feat)).reshape(-1, HEIGHT, WIDTH)
        shift_pred = torch.exp(self.shift_net(scene_feat)).reshape(-1, HEIGHT, WIDTH)

        return scale_pred, shift_pred
