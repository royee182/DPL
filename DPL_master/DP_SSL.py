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
import torch.nn.functional as F
import copy

def get_prediction(model,image):
    output = model(Variable(image).cuda(), ssl=True)
    output = nn.functional.softmax(output, dim=1)
    output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True)
    output = output.cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    return output

def main():
    opt = TestOptions()
    args = opt.initialize()
    args_cp=copy.copy(args)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    thresh=args.thresh
    threshlen=args.threshlen

    args.init_weights=args.init_weights_T
    model_T = CreateSSLModel(args)
    args.init_weights = args.init_weights_S
    model_S = CreateSSLModel(args)
    model_T.eval()
    model_T.cuda()
    model_S.eval()
    model_S.cuda()

    targetloader = CreateTrgDataSSLLoader(args)
    args.data_dir_target=args.data_dir_targetB
    targetloaderB = CreateTrgDataSSLLoader(args,translated=True)
    targetloaderB_iter = iter(targetloaderB)


    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))

    alpha=args_cp.alpha
    image_name = []
    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            if index % 100 == 0:
                print('%d processd' % index)

            image, _, name = batch
            imageB, _, _ = targetloaderB_iter.next()

            output = get_prediction(model_T,image)
            outputB = get_prediction(model_S,imageB)

            output=output*alpha+outputB*(1-alpha)
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
        print("Class-wise threshold:")
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
            predicted_label[index] = label
            output = np.asarray(label, dtype=np.uint8)
            output = Image.fromarray(output)
            name = name.split('/')[-1]
            output.save('%s/%s' % (args.save, name))

    
if __name__ == '__main__':

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    os.system('rm tmp')
    main()
    