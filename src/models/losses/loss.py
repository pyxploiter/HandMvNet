import torch.nn as nn


class PoseLoss:
    @staticmethod
    def mse_loss(preds, labels, stacked_dim=None, weight=1.):
        mse_criterion = nn.MSELoss()
        if stacked_dim:
            labels = labels.unsqueeze(stacked_dim).expand_as(preds)
        return mse_criterion(preds, labels) * weight

    @staticmethod
    def l1_loss(preds, labels, stacked_dim=None, weight=1.):
        l1_criterion = nn.L1Loss()
        if stacked_dim:
            labels = labels.unsqueeze(stacked_dim).expand_as(preds)
        return l1_criterion(preds, labels) * weight

    @staticmethod
    def smooth_l1_loss(preds, labels, stacked_dim=None, weight=1.):
        smooth_l1_criterion = nn.SmoothL1Loss()
        if stacked_dim:
            labels = labels.unsqueeze(stacked_dim).expand_as(preds)
        return smooth_l1_criterion(preds, labels) * weight

    @staticmethod
    def bce_loss(preds, labels, stacked_dim=None, weight=1.):
        bce_criterion = nn.BCELoss()
        sigmoid = nn.Sigmoid()  # bce_loss expects values to be in 0-1 range
        if stacked_dim:
            labels = labels.unsqueeze(stacked_dim).expand_as(preds)
        return bce_criterion(sigmoid(preds), labels) * weight