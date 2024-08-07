import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import config
import utils

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]?
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(config.DEVICE)
        pad_no = math.floor(win[0] / 2)
        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # compute CC squares
        conv_fn = getattr(F, 'conv%dd' % ndims)

        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji
        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win) #return product
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)

class CrossEntropyLoss:
    def __init__(self, **kwargs):
        self.loss = nn.BCELoss(**kwargs)

    def loss(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class L1Loss:
    """
    L1 loss
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims #save each dim grad
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif] #start_dim=1 to flatten each dim and mean each dim (dim=-1)
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class SmoothnessLoss(nn.Module):
    def __init__(self, smoothness_order=1, smoothness_const=-150, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.smoothness_order = smoothness_order
        self.smooth_const = smoothness_const
        if self.smoothness_order == 1:
            self.flow_grad = self.flow_grad_1st_order
        elif self.smoothness_order == 2:
            self.flow_grad = self.flow_grad_2nd_order

    def grad(self, img, stride=1):
        gx = img[..., :, :-stride] - img[..., :, stride:]  # NCHW
        gy = img[..., :-stride, :] - img[..., stride:, :]  # NCHW
        return gx, gy

    def grad_img(self, im, stride):
        im_grad_x, im_grad_y = self.grad(im, stride)
        im_grad_x = im_grad_x.abs().mean(-3, keepdim=True)
        im_grad_y = im_grad_y.abs().mean(-3, keepdim=True)
        return im_grad_x, im_grad_y

    def get_smoothness_mask(self, mask, stride=1):
        mask_x = mask[..., :-stride] * mask[..., stride:]
        mask_y = mask[..., :-stride, :] * mask[..., stride:, :]
        return mask_x, mask_y

    def flow_grad_1st_order(self, flows):
        return self.grad(flows)

    def flow_grad_2nd_order(self, flows):
        flows_grad_x, flows_grad_y = self.grad(flows)
        flows_grad_xx, _ = self.grad(flows_grad_x)
        _, flows_grad_yy = self.grad(flows_grad_y)
        return flows_grad_xx, flows_grad_yy

    def charbonnier_loss(self, data):
        return torch.sqrt(data ** 2 + self.eps ** 2)

    def smoothness_loss_flow(self, flows, ims):
        ims_grad_x, ims_grad_y = self.grad_img(ims, self.smoothness_order)

        flows_grad_x, flows_grad_y = self.flow_grad(flows)
        smoothness_loss = (self.charbonnier_loss(torch.exp(self.smooth_const * ims_grad_x) * flows_grad_x.abs()).mean((-1, -2, -3))
                           + self.charbonnier_loss(torch.exp(self.smooth_const * ims_grad_y) * flows_grad_y.abs()).mean((-1, -2, -3))) / 2

        return smoothness_loss.mean()

    def forward(self, flows, ims):
        return self.smoothness_loss_flow(flows, ims)


class PhotometricLoss(nn.Module):
    def __init__(self, eps=0.001):
        super().__init__()
        self.eps = eps
    def charbonnier_loss(self, data):
        return torch.sqrt(data ** 2 + self.eps ** 2)

    def forward(self, img1, img2, mask):
        im1_grad = utils.calc_edge2d(data=img1, grad=False)
        im2_grad = utils.calc_edge2d(data=img2, grad=False)
        loss = (self.charbonnier_loss(img1 - img2)*mask).sum((-1, -2, -3)) / (mask.sum((-1, -2, -3)) + 1e-6)
        # loss += self.charbonnier_loss((im1_grad - im2_grad) * mask) / (mask.sum() + 1e-6)).sum(-1, -2, -3) / 2
        return loss.mean()


def test():
    gradloss = SmoothnessLoss()
    imloss = PhotometricLoss()
    im = torch.ones((100, 1, 5, 5))
    x = torch.randn((5,5))
    y = torch.randn((5,5))
    flow = torch.stack((x,y))
    flow = torch.randn((100, 1, 5, 5))
    # flow = flow.unsqueeze(0)
    print(gradloss(flow, im))

if __name__ == '__main__':
    test()