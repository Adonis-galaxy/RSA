import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
class NormalizedL1Loss(nn.Module):
    def __init__(self):
        super(NormalizedL1Loss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.abs(y_pred - y_true)
        norm_factor = torch.abs(y_true).mean()
        normalized_loss = loss / norm_factor
        return normalized_loss.mean()

class L1Loss(nn.Module):
    def __init__(self, max_depth=10, min_depth=1e-3, loss_type = "L1"):
        super(L1Loss, self).__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth
        if loss_type == "NormalizedL1":
            self.loss = NormalizedL1Loss()
        elif loss_type == "L1":
            self.loss = torch.nn.L1Loss()
        elif loss_type == "Huber":
            self.loss = torch.nn.Huber(reduction='mean', delta=1.0)
        elif loss_type == "L2":
            self.loss = torch.nn.MSELoss()



    def forward(self, depth_prediction, gts):

        depth_prediction = depth_prediction.unsqueeze(1)

        weight_depth = torch.where(
            (gts > self.min_depth) * (gts < self.max_depth),
            torch.ones_like(gts),
            torch.zeros_like(gts)
        )

        # gts = weight_depth * gts  # remove invalid gt



        diff = self.loss(depth_prediction, gts)

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
