import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from options.test_options import TestOptions
from data import CreateTrgDataSSLLoader, CreateSrcDataSSLLoader
from PIL import Image
import json
import os.path as osp
import os
import numpy as np
from model import CreateSSLModel
import sys

import requests
import time
import threading
import queue

def main():
    opt = TestOptions()
    args = opt.initialize()
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    thresh = args.thresh
    threshlen = args.threshlen
    threshdelta=args.threshdelta
    model = CreateSSLModel(args)

    model.eval()
    model.cuda()
    sourceloader = CreateSrcDataSSLLoader(args)

    labels = []
    for i in range(19):
        labeli = []
        labels.append(labeli)
    image_name = []

    for index, batch in enumerate(sourceloader):
        if index % 100 == 0:
            print('%d processd' % index)
        if index % 20 == 0:
            image, _, _, name = batch
            output = model(Variable(image).cuda(), ssl=True)
            output = nn.functional.softmax(output, dim=1)
            output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[
                0].numpy()
            output = output.transpose(1, 2, 0)

            label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
            for i in range(19):
                cur_pred_list = prob[label == i].tolist()
                labels[i] += cur_pred_list
            image_name.append(name[0])

    thres = []
    for i in range(19):
        x = labels[i]
        print(len(x))
    for i in range(19):
        x = labels[i]
        print(len(x))
        if len(x) == 0:
            thres.append(0)
            continue
        np.sort(x)
        thres.append(x[np.int(np.round(len(x) * threshlen))])
        print(i)
    print(thres)
    thres = np.array(thres)
    thres[thres > thresh] = thresh
    print(thres)
    changes = np.zeros(shape=(19,19), dtype=np.float)
    for index, batch in enumerate(sourceloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, gt_label, _, name = batch
        output = model(Variable(image).cuda(), ssl=True)
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        gt_label=gt_label.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        gt_prob=np.zeros(gt_label.shape, dtype=np.float32)
        for i in range(19):
            gt_prob[gt_label==i]=output[gt_label==i,i]
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        for i in range(19):
            prob[(prob < thres[i]) * (label == i)] = 0
            label[(prob < thres[i]) * (label == i)] = 255



        mask=prob-gt_prob>=threshdelta
        for i in range(19):
            for j in range(19):
                changes[i][j]+=np.sum(mask*(gt_label==i)*(label==j))*1.0/max(np.sum(gt_label==i),1)

        gt_label[mask]=label[mask]


        output = np.asarray(gt_label, dtype=np.uint8)
        output = Image.fromarray(output)
        output.save('%s/%s' % (args.save, name[0]))
    changes/=(index-1)
    changes = np.round(changes, 4)
    print(changes)


if __name__ == '__main__':

    main()
