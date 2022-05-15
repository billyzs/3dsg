import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, pos_weight, gamma):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, pred, label):
        prob = torch.sigmoid(pred)
        label_prob = torch.clip(torch.mul(label, prob) + torch.mul(1-label, 1-prob), 0.0000001, 0.999999)
        bce_loss = -torch.log(label_prob)
        focal_loss = torch.mul(torch.pow(1-label_prob, self.gamma), bce_loss)
        weighted_focal_loss = torch.mul(self.pos_weight, focal_loss)
        return weighted_focal_loss
