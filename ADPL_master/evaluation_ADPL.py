import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplabv2 import Res_Deeplab
from data import get_data_path, get_loader
import torchvision.transforms as transform
import time
from PIL import Image
import scipy.misc
from utils.loss import CrossEntropy2d
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
from model.fcn8s import VGG16_FCN8s

from data.cityscapes_loader_pd import cityscapesLoader_pd

MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)




def get_prediction(model,image):
    output = model(Variable(image).cuda())
    output = nn.functional.softmax(output, dim=1)
    output = nn.functional.upsample(output, (1024, 2048), mode='bilinear', align_corners=True)
    output = output.cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    return output
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UDA evaluation script")
    parser.add_argument("--model_S_restore", type=str,
                        help="which path to restore S model",default='ADPL_pretrained/Deep_GTA5/S.pth')
    parser.add_argument("--model_T_restore", type=str,
                        help="which path to restore T model",default='ADPL_pretrained/Deep_GTA5/T.pth')
    parser.add_argument("--model_name", type=str,
                        help="which model, DeepLab or VGG",default='DeepLab')
    parser.add_argument("--source_dataset_name", type=str,
                        help="which source dataset, gta or synthia",default='gta')
    parser.add_argument("--save_dir",
                        help="path to save output images",default='dual')
    return parser.parse_args()

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

def get_iou(data_list, class_num, save_path=None,i_iter=0,source_dataset_name='gta'):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(("road", "sidewalk",
        "building", "wall", "fence", "pole",
        "traffic_light", "traffic_sign", "vegetation",
        "terrain", "sky", "person", "rider",
        "car", "truck", "bus",
        "train", "motorcycle", "bicycle"))


    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]))

    print('meanIOU: ' + str(aveJ))
    print('mIoU16: ' + str(
        round(np.mean(np.asarray(j_list)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)))
    print('mIoU13: ' + str(
        round(np.mean(np.asarray(j_list)[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)))
    if save_path:
        with open(save_path, 'a') as f:
            f.write('iter :' + str(i_iter) + '\n')
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]) + '\n')

            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write('mIoU16: ' + str(
                round(np.mean(np.asarray(j_list)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)) + '\n')
            f.write(' mIoU13: ' + str(round(np.mean(np.asarray(j_list)[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)) + '\n')
    if source_dataset_name=='gta':
        return aveJ
    else:
        return np.mean(np.asarray(j_list)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]])

def get_iou16(data_list, class_num, dataset, save_path=None,i_iter=0):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(("road", "sidewalk",
        "building", "wall", "fence", "pole",
        "traffic_light", "traffic_sign", "vegetation",
        "sky", "person", "rider",
        "car", "bus",
         "motorcycle", "bicycle"))


    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')
    return aveJ


def resume(model,restore_from):

    saved_state_dict = torch.load(restore_from,map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict['model'])
    return model



def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    model_S_restore = args.model_S_restore
    model_T_restore = args.model_T_restore
    model_name = args.model_name
    source_dataset_name = args.source_dataset_name
    save_dir=args.save_dir
    save_dir = save_dir + '/results'
    data_path_ori = get_data_path('cityscapes', 'T', source_dataset_name, model_name, ablation="None")
    data_path_tran = get_data_path('cityscapes', 'S', source_dataset_name, model_name, ablation="None")

    if model_name == "DeepLab":
        model_T = Res_Deeplab(num_classes=19)
        model_S = Res_Deeplab(num_classes=19)

    else:
        model_T = VGG16_FCN8s(num_classes=19)
        model_S = VGG16_FCN8s(num_classes=19)

    model_S = resume(model_S, model_S_restore)
    model_T = resume(model_T, model_T_restore)
    print("restore S from %s" % (model_S_restore))
    print("restore T from %s" % (model_T_restore))

    model_T.eval()
    model_T.cuda()
    model_S.eval()
    model_S.cuda()

    data_loader = get_loader('cityscapes_label')
    test_dataset = data_loader(data_path_ori, img_size=(512, 1024), img_mean=IMG_MEAN, is_transform=True, split='val')
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    test_datasetB = data_loader(data_path_tran, img_size=(512, 1024), img_mean=IMG_MEAN, is_transform=True, split='val')
    testloaderB = data.DataLoader(test_datasetB, batch_size=1, shuffle=False, pin_memory=True)

    targetloaderB_iter = iter(testloaderB)

    data_list = []


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for index, batch in enumerate(testloader):
        image, label, size, name, _ = batch
        imageB, _, _, name,_ = targetloaderB_iter.next()
        size = size[0]

        with torch.no_grad():
            output = get_prediction(model_T, image)
            outputB = get_prediction(model_S, imageB)
            output = output * 0.5 + outputB * 0.5

            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            # output_col = colorize_mask(output)

            name = name[0].split('/')[-1]

            # output_col.save('%s/%s_color.png' % (save_dir, name.split('.')[0]))
            gt = np.asarray(label[0].numpy(), dtype=np.int)
            data_list.append([gt.flatten(), output.flatten()])

        if (index + 1) % 100 == 0:
            # break
            print('%d processed' % (index + 1))
    # np.save('gt.npy',gt)
    # np.save('output.npy',output)
    if save_dir:
        filename = os.path.join(save_dir, 'result.txt')
    else:
        filename = None
    mIoU = get_iou(data_list, 19,  filename, 0, source_dataset_name)




if __name__ == '__main__':

    main()
