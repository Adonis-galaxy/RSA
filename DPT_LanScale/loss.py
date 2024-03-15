import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def pixel_unshuffle(fm, r):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    b, c, h, w = fm.shape
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    fm_view = fm.contiguous().view(b, c, out_h, r, out_w, r)
    fm_prime = fm_view.permute(0,1,3,5,2,4).contiguous().view(b, out_channel, out_h, out_w)

    return fm_prime

# class Depth_Loss(nn.Module):
#     def __init__(self, smooth = False, w_smooth = 1e-2):
#         super(Depth_Loss, self).__init__()
#         self.smooth = smooth
#         self.w_smooth = w_smooth

#     def gradient_yx(self, T):
#         '''
#         Computes gradients in the y and x directions

#         Arg(s):
#             T : torch.Tensor[float32]
#                 N x C x H x W tensor
#         Returns:
#             torch.Tensor[float32] : gradients in y direction
#             torch.Tensor[float32] : gradients in x direction
#         '''

#         dx = T[:, :, :, :-1] - T[:, :, :, 1:]
#         dy = T[:, :, :-1, :] - T[:, :, 1:, :]
#         return dy, dx

#     def smoothness_loss_func(self, weight, predict, image):
#         '''
#         Computes the local smoothness loss

#         Arg(s):
#             weight : torch.Tensor[float32]
#                 N x 1 x H x W binary mask
#             predict : torch.Tensor[float32]
#                 N x 1 x H x W predictions
#             image : torch.Tensor[float32]
#                 N x 3 x H x W RGB image
#         Returns:
#             torch.Tensor[float32] : mean local smooth loss
#         '''


#         predict_dy, predict_dx = self.gradient_yx(predict)
#         image_dy, image_dx = self.gradient_yx(image)

#         # Create edge awareness weights
#         weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
#         weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

#         weight_valid_x=weight[:,:,:,:-1]
#         weight_valid_y=weight[:,:,:-1,:]

#         smoothness_x = torch.sum(weight_valid_x * weights_x * torch.abs(predict_dx)) / torch.sum(weight_valid_x)
#         smoothness_y = torch.sum(weight_valid_y * weights_y * torch.abs(predict_dy)) / torch.sum(weight_valid_y)

#         return smoothness_x + smoothness_y

#     def forward(self, depth_prediction, gts, image=None):

#         loss_depth = torch.abs(depth_prediction-gts) / (torch.abs(gts) + 1e-8)


#         weight_depth = torch.where(
#         (gts > 1e-3) * (gts < 10),
#         torch.ones_like(loss_depth),
#         torch.zeros_like(loss_depth)
#         )

#         loss_depth = torch.sum(weight_depth * loss_depth) / torch.sum(weight_depth)

#         weight_smooth = 1-weight_depth

#         if self.smooth:
#             loss_smooth = self.smoothness_loss_func(weight_smooth,depth_prediction,image)
#             loss = loss_depth + self.w_smooth * loss_smooth
#             return loss, loss_depth, loss_smooth
#         else:
#             return loss_depth


class SILogLoss(nn.Module):
    def __init__(self, SI_loss_lambda=0.85, max_depth=10, min_depth=1e-3):
        super(SILogLoss, self).__init__()

        self.SI_loss_lambda = SI_loss_lambda
        self.max_depth = max_depth
        self.min_depth = min_depth

    def forward(self, depth_prediction, gts):

        depth_prediction=depth_prediction.unsqueeze(1)

        shape = [depth_prediction.shape[2], depth_prediction.shape[3]]
        #scale_factor = self.shape_h // shape[0]
        scale_factor = int(np.sqrt(depth_prediction.shape[1]))

        reshaped_gt = pixel_unshuffle(gts, scale_factor)

        diff = torch.log(depth_prediction) - torch.log(reshaped_gt)

        num_pixels = (reshaped_gt > self.min_depth) * (reshaped_gt < self.max_depth)


        diff = torch.where(
        (reshaped_gt > self.min_depth) * (reshaped_gt < self.max_depth),
        diff,
        torch.zeros_like(diff)
        )
        lamda = self.SI_loss_lambda

        diff = diff.reshape(diff.shape[0], -1)
        num_pixels = num_pixels.reshape(num_pixels.shape[0], -1).sum(dim=-1) + 1e-6

        loss1 = (diff**2).sum(dim=-1) / num_pixels
        loss1 = loss1 - lamda * (diff.sum(dim=-1) / num_pixels) ** 2
        #loss1 = diff.abs().sum(dim=-1) / num_pixels

        total_pixels = reshaped_gt.shape[1] * reshaped_gt.shape[2] * reshaped_gt.shape[3]

        weight = num_pixels.to(diff.dtype) / total_pixels

        loss = (loss1 * weight).sum()

        return loss




# class L1_Loss(nn.Module):
#     def __init__(self, normalize=False, epsilon=1e-8):
#         '''
#         Arg(s):
#             normalize : [bool]
#                 if normalize : normalized l1 loss
#                 else : plain l1 loss
#             epsilon : [float]
#                 avoid normlize devide by 0

#         '''
#         super(L1_Loss, self).__init__()

#         self.normalize = normalize
#         self.loss_func = torch.nn.L1Loss(reduction='none')
#         self.epsilon = epsilon

#     def forward(self, depth_prediction, gts):
#         '''
#         Arg(s):
#             depth_prediction : torch.Tensor[float32]
#                 N x 1 x H x W source image (output depth)
#             gts : torch.Tensor[float32]
#                 N x 1 x H x W target image (depth gt)
#         Returns:
#             float : mean l1 loss across batch
#         '''
#         loss=self.loss_func(depth_prediction,gts)


#         if self.normalize:
#             loss = loss / (torch.abs(gts) + self.epsilon)


#         return torch.mean(loss)

class L1Loss(nn.Module):
    def __init__(self, max_depth=10, min_depth=1e-3, normalize=False):
        super(L1Loss, self).__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.normalize = normalize


    def forward(self, depth_prediction, gts):

        depth_prediction = depth_prediction.unsqueeze(1)

        # shape = [depth_prediction.shape[2], depth_prediction.shape[3]]

        # scale_factor = int(np.sqrt(depth_prediction.shape[1]))

        # reshaped_gt = pixel_unshuffle(gts, scale_factor)


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
