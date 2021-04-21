import torch
import torch.nn as nn
from torch.nn import functional as F

from ...builder import LOSSES


class IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        targets = targets.float()
        num = torch.sum(outputs * targets * mask)
        den = torch.sum(outputs * mask + targets * mask - outputs * targets * mask)
        return 1 - num / den


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([1 - alpha, alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.eps = 1e-10

    def forward(self, outputs, targets):
        outputs, targets = torch.sigmoid(outputs.view(-1, 1)), targets.view(-1, 1).long()  # (N, 1)
        outputs = torch.cat((1 - outputs, outputs), dim=1)  # (N, 2)

        pt = outputs.gather(1, targets).view(-1)
        logpt = torch.log(outputs + self.eps)
        logpt = logpt.gather(1, targets).view(-1)

        if self.alpha is not None:
            if self.alpha.type() != outputs.data.type():
                self.alpha = self.alpha.type_as(outputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class RegL1Loss(nn.Module):
    def __init__(self, ignore_index=255):
        super(RegL1Loss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, output, target, mask):
        _mask = mask.detach().clone()
        _mask[mask == self.ignore_index] = 0.
        loss = F.l1_loss(output * _mask, target * _mask, reduction='sum')
        loss = loss / (_mask.sum() + 1e-12)
        return loss


@LOSSES.register_module
class AFLoss(nn.Module):
    def __init__(self, loss_type="wbce", ignore_label=255):
        super(AFLoss, self).__init__()
        if loss_type == 'focal':
            self.criterion_1 = FocalLoss(gamma=2.0, alpha=0.25, size_average=True)
        elif loss_type == 'bce':
            ## BCE weight
            self.criterion_1 = torch.nn.BCEWithLogitsLoss()
        elif loss_type == 'wbce':
            ## BCE weight
            self.criterion_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.6]).cuda())
        self.criterion_2 = IoULoss()
        self.criterion_reg = RegL1Loss()
        self.ignore_label = ignore_label

    def forward(self, outputs, **label):
        input_seg, input_mask, input_af = label['seg'], label['mask'], label['af']
        _mask = (input_mask != self.ignore_label).float()
        loss_seg = self.criterion_1(outputs['hm'] * _mask, input_mask * _mask) + self.criterion_2(
            torch.sigmoid(outputs['hm']),
            input_mask)
        loss_vaf = 0.5 * self.criterion_reg(outputs['vaf'], input_af[:, :2, :, :], input_mask)
        loss_haf = 0.5 * self.criterion_reg(outputs['haf'], input_af[:, 2:3, :, :], input_mask)
        # loss =  loss_seg + loss_vaf + loss_haf
        return [loss_seg, loss_vaf, loss_haf]
