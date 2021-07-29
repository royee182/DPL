
import torch
import torch.nn as nn
from torch.autograd import Variable
from options.test_options import TestOptions
from data import CreateTrgDataLoader
from PIL import Image
import json
import os.path as osp
import os
import numpy as np
from model import CreateModel
from evaluation import compute_mIoU
def main():
    opt = TestOptions()
    args = opt.initialize()
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    args.init_weights = args.init_weights_T
    model_T = CreateModel(args)
    args.init_weights = args.init_weights_S
    model_S = CreateModel(args)
    model_T.eval()
    model_T.cuda()
    model_S.eval()
    model_S.cuda()

    alpha=0.55
    targetloader = CreateTrgDataLoader(args)
    args.data_dir_target=args.data_dir_targetB
    targetloaderB = CreateTrgDataLoader(args,translated=True)
    targetloaderB_iter = iter(targetloaderB)
    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            if index % 100 == 0:
                print('%d processd' % index)

            image, _, name = batch
            imageB,_, _ = targetloaderB_iter.next()
            output =  nn.functional.softmax(model_T(Variable(image).cuda()), dim=1)
            outputB=nn.functional.softmax(model_S(Variable(imageB).cuda()), dim=1)

            output=output*alpha+outputB*(1-alpha)
            output = nn.functional.upsample(output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            output = output.transpose(1,2,0)
            output_nomask = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            output_nomask = Image.fromarray(output_nomask)
            name = name[0].split('/')[-1]
            output_nomask.save('%s/%s' % (args.save, name))
    compute_mIoU(args.gt_dir, args.save, args.devkit_dir,args.log_dir)

if __name__ == '__main__':
    
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    os.system('rm tmp')
    main()
    
    
