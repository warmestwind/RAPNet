import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    # input = input.clone().unsqueeze(1).contiguous()  # clone().
    # input[input==255]=0
    # print('unique:', torch.unique(input))
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)
    return result


class onecls_DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        # print(predict.shape, target.shape)
        assert predict.shape == target.shape, "predict & target shape don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        # print('unique:', target.unique())
        # print(predict.max(), target.max())
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(torch.pow(predict, 2) + torch.pow(target, 2), dim=1) + self.smooth
        den = torch.sum(predict,dim=1) + torch.sum(target,dim=1) + self.smooth
        # print(num, den)
        loss = 1 - 2*num / den
        return torch.mean(loss)

class onecls_TverskyLoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=0.6):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = 1 - self.alpha

    def forward(self, predict, target):
        # print(predict.shape, target.shape)
        assert predict.shape == target.shape, "predict & target shape don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        tp = torch.sum(torch.mul(predict, target), dim=1)
        fp = torch.sum(torch.mul(predict, 1-target), dim=1)
        fn = torch.sum(torch.mul(1-predict, target), dim=1)
        tversky = (tp + self.smooth) / (tp + self.beta*fp + self.alpha*fn + self.smooth)
        loss = 1 - tversky
        return loss

class multicls_DiceLoss(nn.Module): # For images that have ground truth
    '''
    input:
        predict shape: batch_size * class_num * H * W
        target shape: batch_size * H * W
    '''
    def __init__(self, n_class = 2):
        super().__init__()
        # self.loss = onecls_TverskyLoss(alpha=0.7)
        self.loss = onecls_DiceLoss()
        self.n_class = n_class

    def forward(self, predict, target):
        # target [B, 5, 1, H, W]
        target = make_one_hot(target, self.n_class)
        #print('channel:', chl)
        assert predict.shape == target.shape, "predict & target shape don't match"
        # print(target.shape)
        w = [0.5, 0.5]
        total_loss = 0
        # predict = predict.permute(0, 2, 1, 3, 4)
        # target = target.permute(0, 2, 1, 3, 4)
        for i in range(target.shape[1]):
            # print(target.shape)
            #print('sum_%d :'%i, torch.sum(target[:, i, 1:, :, :]), torch.sum(target[:, i, :, :, :]))
            # loss_i = torch.pow(w[i]*self.loss(predict[:, :, i], target[:,:, i]), 0.75)
            loss_i = w[i]*self.loss(predict[:, i], target[:, i])
            # print('loss_%d_0 :'%i, loss_i, loss_i.shape)
            # loss_i += 0.3*(-torch.log(predict[:, :, i]+1e-6)*target[:,:, i]).mean((1,2,3))
            # print('loss_%d_1 :' % i, loss_i, loss_i.shape)
            total_loss += loss_i
            #print(target.shape)
        return total_loss # / target.shape[1]

