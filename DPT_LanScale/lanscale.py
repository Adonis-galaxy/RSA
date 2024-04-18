import torch
import torch.nn as nn

class LanScaleModel(nn.Module):
    '''
    LanScaleModel Network class

    Arg(s):
        text_feat_dim: int
            dimension of input CLIP text feature
    '''
    def __init__(self, text_feat_dim=1024, default_scale=0.000305, default_shift=0.1378, dataset=None):
        super().__init__()
        self.default_scale = default_scale
        self.default_shift = default_shift
        self.scene_feat_net = nn.Sequential(
            nn.Linear(text_feat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256)
        )

        self.shift_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.scale_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        self.dataset = dataset
        # nn.init.zeros_(self.shift_net[-1].weight)
        # nn.init.zeros_(self.shift_net[-1].bias)
        # nn.init.zeros_(self.scale_net[-1].weight)
        # nn.init.zeros_(self.scale_net[-1].bias)

        # self.scale_global = nn.Parameter(torch.tensor(default_scale))
        # self.shift_global = nn.Parameter(torch.tensor(default_shift))

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

        scale_pred = torch.sigmoid(self.scale_net(scene_feat))
        shift_pred = torch.exp(self.shift_net(scene_feat))
        # shift_pred = self.shift_net(scene_feat) + 1

        # if self.dataset == "kitti":
        #     scale_pred = scale_pred * 0.000125
        #     shift_pred = shift_pred * 0.005

        return scale_pred, shift_pred
