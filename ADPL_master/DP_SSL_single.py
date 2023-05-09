import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from model.deeplabv2 import Res_Deeplab
from PIL import Image
import json
import os.path as osp
import os
from data import get_data_path, get_loader
import numpy as np
from data.cityscapes_loader_pd import cityscapesLoader_pd
from model.fcn8s import VGG16_FCN8s
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

import torch.nn.functional as F
import copy

def main():
    print("hello pseudo")

# S_thresh1=[0.9992380738258362, 0.7279524207115173, 0.9888620376586914, 0.6502715945243835, 0.6621962785720825, 0.6472988128662109, 0.7358214259147644, 0.7260012030601501, 0.9795948266983032, 0.8141217231750488, 0.9996448159217834, 0.8897566199302673, 0.7717059850692749, 0.9968376159667969, 0.8584262132644653, 0.9161942601203918, 0.6292511224746704, 0.8705059885978699, 0.913240909576416]
# T_thresh1=[0.9999994039535522, 0.9995578527450562, 0.9999756813049316, 0.9966528415679932, 0.9969298839569092,0.8596767783164978, 0.9287092089653015, 0.9993431568145752, 0.9998055696487427, 0.9992105960845947, 0.9999780058860779, 0.9923610091209412, 0.9811285734176636, 0.9999764561653137, 0.9999130368232727, 0.9999348521232605, 0.9999962449073792, 0.9998480081558228, 0.9987118244171143]
#
#
# S_thresh2=[0.9, 0.9,0.9,0.86375505,0.86667717,0.70775074,0.77573496, 0.89710218 ,0.9,0.9,0.9 ,0.9,0.85581583 ,0.9,0.9, 0.9 , 0.9, 0.9,0.9]
# T_thresh2=[0.9, 0.9, 0.9, 0.88716769, 0.89810914, 0.7251147, 0.78902125, 0.9 ,  0.9, 0.9, 0.9,0.9, 0.87353975, 0.9, 0.9, 0.9 , 0.9 , 0.9, 0.9]
# S_thresh=np.asarray(S_thresh2)
# T_thresh=np.asarray(T_thresh2)
# print(np.nanmean(S_thresh))
# print(np.nanmean(T_thresh))
# w1=np.exp(T_thresh)/(np.exp(S_thresh)+np.exp(T_thresh))
# print(w1)
def get_prediction(model,image):
    output = model(Variable(image).cuda())
    output = nn.functional.softmax(output, dim=1)
    output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True)
    output = output.cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    return output

def resume(model,restore_from):

    saved_state_dict = torch.load(restore_from)
    if 'st2' in restore_from:
        model.load_state_dict(saved_state_dict['model'])
        return model
    try:
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    except:
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)
    return model

def pseudo_gener(model_S_restore,model_T_restore,save_path,thresh_path,model_name,data_path_ori,data_path_tran,dataloader='cityscapes'):
    num_classes = 19


    if model_name=="DeepLab":
        model_T = Res_Deeplab(num_classes=num_classes)
        model_S = Res_Deeplab(num_classes=num_classes)

    else:
        model_T = VGG16_FCN8s(num_classes=num_classes)
        model_S = VGG16_FCN8s(num_classes=num_classes)

    model_S=resume(model_S,model_S_restore)
    model_T=resume(model_T,model_T_restore)
    print("restore S from %s"%(model_S_restore))
    print("restore T from %s"%(model_T_restore))

    model_T.eval()
    model_T.cuda()
    model_S.eval()
    model_S.cuda()

    target_loader = get_loader(dataloader)
    ori_dataset = target_loader( data_path_ori, img_size=(512,1024), img_mean = IMG_MEAN, is_transform=True, split='train')
    targetloader = data.DataLoader(ori_dataset, batch_size=1, shuffle=False, pin_memory=True)

    trans_dataset = target_loader( data_path_tran, img_size=(512,1024), img_mean = IMG_MEAN, is_transform=True, split='train')
    targetloaderB= data.DataLoader(trans_dataset, batch_size=1, shuffle=False, pin_memory=True)


    targetloaderB_iter = iter(targetloaderB)

    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))


    predicted_label_B = np.zeros((len(targetloader), 512, 1024))
    predicted_prob_B = np.zeros((len(targetloader), 512, 1024))

    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            if index % 100 == 0 and index!=0:
                print('%d processd' % index)
            if index%2!=0:
                continue
            image, name = batch
            imageB,name= targetloaderB_iter.next()

            output = get_prediction(model_T,image)
            outputB = get_prediction(model_S,imageB)

            output = np.asarray(output)
            label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
            predicted_label[index] = label.copy()
            predicted_prob[index] = prob.copy()

            output_B = np.asarray(outputB)
            label_B, prob_B = np.argmax(output_B, axis=2), np.max(output_B, axis=2)
            predicted_label_B[index] = label_B.copy()
            predicted_prob_B[index] = prob_B.copy()


        thres = []
        for i in range(19):
            x = predicted_prob[predicted_label==i]
            if len(x) == 0:
                thres.append(0)
                continue
            # x = np.sort(x)
            thres.append(np.mean(x[x!=0]))
        print("Class-wise threshold:")
        print(thres)

        thres_B = []
        for i in range(19):
            x = predicted_prob_B[predicted_label_B==i]
            if len(x) == 0:
                thres_B.append(0)
                continue
            # x = np.sort(x)
            thres_B.append(np.mean(x[x!=0]))
        print("Class-wise threshold:")
        print(thres_B)

        thres=np.asarray(thres)
        thres_B=np.asarray(thres_B)

        np.save(save_path+'/perf_t.npy',thres)
        print("saved in "+save_path+"/perf_t.npy")

        np.save(save_path+'/perf_s.npy',thres_B)
        print("saved in "+save_path+"/perf_s.npy")
        return thres,thres_B

        torch.cuda.empty_cache()
    
if __name__ == '__main__':

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    os.system('rm tmp')
