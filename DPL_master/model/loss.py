import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import os


class SelfAdaptiveTrainingCE():
    def __init__(self, num_classes=19, momentum=0.9, es=0):
        # initialize soft labels to onthot vectors

        self.momentum = momentum
        self.es = es
        self.num_classes = num_classes

    def __call__(self, logits, target, name='', save_root='', epoch=0):
        if epoch < self.es:
            return F.cross_entropy(logits, target)
        target=target*1.0
        for_save_softlabel=target
        target_mask = target[:, :, 19] != 1.0
        targets = target[target_mask]

        if not targets.data.dim():
            return Variable(torch.zeros(1))
        n, c, h, w = logits.size()
        for_save_logits=logits.clone()
        for_save_logits=for_save_logits.transpose(1, 2).transpose(2, 3).contiguous().view(n,h*w,c)
        for_save_prob = F.softmax(for_save_logits.detach(), dim=2)
        weights, _ = for_save_prob.max(dim=2)
        for_save_prob[torch.where(weights<0.9)]=0
        for_save_softlabel[:,:, :19]=self.momentum * for_save_softlabel[:,:, :19] + (1 - self.momentum) * for_save_prob
        for_save_softlabel=for_save_softlabel.float()
        for_save_softlabel = for_save_softlabel.cpu().detach().numpy()
        for i in range(n):
            np.save(os.path.join(save_root, name[i].split('/')[-1].split('.')[0]), for_save_softlabel[i])

        logits = logits.transpose(1, 2).transpose(2, 3).contiguous()
        logits = logits[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        a=None
        # obtain prob, then update running avg
        prob = F.softmax(logits.detach(), dim=1)
        uncertain, _ = prob.max(dim=1)
        prob[torch.where(prob < 0.9)] = 0
        soft_label = self.momentum * targets[:, :19] + (1 - self.momentum) * prob

        # obtain weights
        weights, _ = soft_label.max(dim=1)
        weights *= logits.shape[0] / weights.sum()

        # compute cross entropy loss, without reduction
        loss = torch.sum(-F.log_softmax(logits, dim=1) * soft_label, dim=1)

        # sample weighted mean
        loss = (loss * weights).mean()
        return loss