import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from options.test_options import TestOptions
from data import CreateTrgDataSSLLoader
from PIL import Image
import json
import os.path as osp
import os
import numpy as np
from model import CreateSSLModel


def main():
    opt = TestOptions()
    args = opt.initialize()
    print(args.save)
    print(os.path.exists(args.save))
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    model = CreateSSLModel(args)
    thresh=args.thresh
    threshlen=args.threshlen
    model.eval()
    model.cuda()   
    targetloader = CreateTrgDataSSLLoader(args)

    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))
    image_name = []
    
    for index, batch in enumerate(targetloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, _, name = batch
        output = model(Variable(image).cuda(), ssl=True)
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])
        
    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*threshlen))])
    print(thres)
    thres = np.array(thres)
    thres[thres>thresh]=thresh
    print(thres)
    for index in range(len(targetloader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[(prob<thres[i])*(label==i)] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (args.save, name))
    
if __name__ == '__main__':

    main()
    