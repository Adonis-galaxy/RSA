import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class L1Loss(nn.Module):
    def __init__(self, max_depth=10, min_depth=1e-3, normalize=False):
        super(L1Loss, self).__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.normalize = normalize


    def forward(self, depth_prediction, gts):

        depth_prediction = depth_prediction.unsqueeze(1)

        weight_depth = torch.where(
            (gts > self.min_depth) * (gts < self.max_depth),
            torch.ones_like(gts),
            torch.zeros_like(gts)
        )
        
        # gts = weight_depth * gts  # remove invalid gt

        if self.normalize:
            diff = torch.abs(depth_prediction - gts)
            loss = torch.where(weight_depth == 1, diff / gts, torch.zeros_like(diff))
            loss = torch.sum(loss, dim=(1, 2, 3)) / torch.sum(weight_depth, dim=(1, 2, 3))
            loss = torch.mean(loss)
            return loss
        else:
            diff = torch.abs(depth_prediction - gts)

        num_pixels = (gts > self.min_depth) * (gts < self.max_depth)


        diff = torch.where(
        (gts > self.min_depth) * (gts < self.max_depth),
        diff,
        torch.zeros_like(diff)
        )

        diff = diff.reshape(diff.shape[0], -1)
        num_pixels = num_pixels.reshape(num_pixels.shape[0], -1).sum(dim=-1) + 1e-6

        loss1 = (diff).sum(dim=-1) / num_pixels

        total_pixels = gts.shape[1] * gts.shape[2] * gts.shape[3]

        weight = num_pixels.to(diff.dtype) / total_pixels

        loss = (loss1 * weight).sum()

        return loss
