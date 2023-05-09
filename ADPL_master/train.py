import argparse

import os

import sys
import random
import timeit
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform
from model.dis import CreateDiscriminator
from model.deeplabv2 import Res_Deeplab
import glob
from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d
from shutil import copytree,rmtree,copyfile
from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluation import evaluate
import time
import warnings
warnings.filterwarnings('ignore')
start = timeit.default_timer()

import copy
import threading
from DP_SSL_single import pseudo_gener
from DP_SSL_weight_soft import soft_pseudo_gener
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')
from model.fcn8s import VGG16_FCN8s
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("-c", "--config", type=str, default='./configs/configUDA_gta_deep_st1.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default="GTA2City",
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    parser.add_argument("--checkpoint_dir", type=str, default='chpt/Deep_GTA',
                        help='Path to save checkpoints')
    parser.add_argument("--reweight_thresh", type=float, default=0.968,
                        help='reweight thresh')
    parser.add_argument("--tau", type=float, default=0.7)
    return parser.parse_args()

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / 10000) ** (power))

def adjust_learning_rate(optimizer, i_iter,learning_rate_base):
    # print(total_iterations)
    lr = lr_poly(learning_rate_base, i_iter, total_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model):
    #ema_model = getattr(models, config['arch']['type'])(self.train_loader.dataset.num_classes, **config['arch']['args']).to(self.device)
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    if len(gpus)>1:
        #return torch.nn.DataParallel(ema_model, device_ids=gpus)
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)#data0*mask
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def strongTransform_ammend(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def strongTransform_class_mix(image1, image2, label1, label2, mask_img, mask_lbl, cls_mixer, cls_list, strong_parameters): #image1, image2, label1, label2, mask_img, mask_lbl image1,image2 mix by mask(on image1)
    inputs_, _ = transformsgpu.oneMix(mask_img, data=torch.cat((image1.unsqueeze(0), image2.unsqueeze(0)))) #mask*image1+(1-mask)*image2
    _, targets_ = transformsgpu.oneMix(mask_lbl, target=torch.cat((label1.unsqueeze(0), label2.unsqueeze(0))))
    #inputs_, targets_ = cls_mixer.mix(inputs_.squeeze(0), targets_.squeeze(0), cls_list)
    out_img, out_lbl = strongTransform_ammend(strong_parameters, data=inputs_, target=targets_)
    return out_img, out_lbl

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target



class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('../visualiseImages/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('../visualiseImages', str(epoch)+ id + '.png'))

def _save_checkpoint(miou,iteration, model, optimizer, model_D,optimizer_D,config,ema_model, save_best=False, overwrite=False,train_path="T"):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'optimizer_D':optimizer_D.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        checkpoint['model_D'] = model_D.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        checkpoint['model_D'] = model_D.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()


    if save_best:
        # filelist = glob.glob(os.path.join(checkpoint_dir,'*.pth'))
        # if filelist:
        #     os.remove(filelist[0])
        filename = os.path.join(checkpoint_dir, f'{train_path}_{miou}best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'{train_path}_{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')

        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'{train_path}_{iteration - save_checkpoint_every}.pth'))
            except:
                pass
def write_numpy(psuedo_path,aftermix,h1,h2,w1,w2,i):
    origin_prob = np.load(psuedo_path[i])
    origin_prob[h1[i]:h2[i], w1[i]:w2[i], :] = aftermix[i, :, :, :]
    np.save(psuedo_path[i], origin_prob)

def rewrite_probmap(psuedo_path,aftermix,params):
    h1,h2,w1,w2=params[0],params[1],params[2],params[3]
    t=[]
    for i in range(len(psuedo_path)):
        t.append(threading.Thread(target=write_numpy, args=[psuedo_path,aftermix,h1,h2,w1,w2,i]))
        t[i].start()
    for i in range(len(psuedo_path)):
        t[i].join()

def _resume_checkpoint(resume_path, model,optimizer, model_D,optimizer_D, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    try:
        model.module.load_state_dict(checkpoint['model'])
    except:
        model.load_state_dict(checkpoint['model'])

    try:
        model_D.module.load_state_dict(checkpoint['model_D'])
    except:
        model_D.load_state_dict(checkpoint['model_D'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model,optimizer,model_D,optimizer_D,ema_model

def restore_model(restore_from,model):
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)
    if 'st2' in restore_from:
        model.load_state_dict(saved_state_dict['model'])
    else:
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


def logits_fusion(model, model_ema, images_target_cp,interp,w1):
    model.eval()
    model_ema.eval()
    with torch.no_grad():
        pw_target = model(images_target_cp)
        pw_target = nn.functional.softmax(pw_target, dim=1)
        logits_target = interp(pw_target)

        # logits_target = logits_target.cpu().data[0].numpy()
        # logits_target = logits_target.transpose(1, 2, 0)


        pw_target_ema = model_ema(images_target_cp)
        pw_target_ema = nn.functional.softmax(pw_target_ema, dim=1)
        logits_target_ema = interp(pw_target_ema)

        # logits_target_ema = logits_target_ema.cpu().data[0].numpy()
        # logits_target_ema = logits_target_ema.transpose(1, 2, 0)

        # print(logits_target.shape)

        B,C,H,W=logits_target.shape
        w1=torch.Tensor(w1).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B,1,H,W).cuda()
        # print(w1.shape)
        logits_target = w1 * logits_target + (1 - w1) * logits_target_ema
        max_probs_target, target_psuedo_label = torch.max(logits_target, dim=1)

    model.train()
    # model_ema.train()
        # logits_target=torch.Tensor(logits_target)
    return max_probs_target, target_psuedo_label


def update_perf(prob, label, perf, alpha=0.999):
    perf_cur = perf.copy()

    for i in range(19):
        if (label == i).sum() == 0:
            continue
        perf_cur[i] = prob[label == i].sum() / ((label == i).sum())
        perf_cur[i] = perf_cur[i] * (1 - alpha) + perf[i] * alpha
    return perf_cur

def do_epoch(train_path="T",start_iteration=0, num_iterations=1000,model=None,model_ema=None,model_D=None,optimizer=None,optimizer_D=None,best_mIoU=0,mix_para=0,thresh=None):

    try:
        model_for_reference=copy.deepcopy(model.module)
    except:
        model_for_reference = copy.deepcopy(model)
    model.train()
    model_ema.eval()
    model_D.train()

    cudnn.benchmark = True
    target_loader = get_loader('cityscapes')
    target_path = get_data_path('cityscapes',train_path,source_dataset_name,model_name,ablation)

    if random_crop:
        data_aug = Compose([RandomCrop_city(input_size)])
    else:
        data_aug = None

    #data_aug = Compose([RandomHorizontallyFlip()])
    target_dataset = target_loader(target_path, is_transform=True, augmentations=data_aug, img_size=input_size,psuedo_root=None, psoft=True,img_mean = IMG_MEAN)


    np.random.seed(random_seed)
    targetloader = data.DataLoader(target_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)

    targetloader_iter = iter(targetloader)


    #New loader for Domain transfer

    source_loader = get_loader(source_dataset_name)
    source_path,label_path = get_data_path(source_dataset_name,train_path,source_dataset_name,model_name,ablation)
    if random_crop:
        data_aug = Compose([RandomCrop_gta(input_size)])
    else:
        data_aug = None

    if source_dataset_name=='gta':
        list_path='./data/gta5_list/train.txt'
    else:
        list_path = './data/synthia_list/train.txt'
    #data_aug = Compose([RandomHorizontallyFlip()])
    source_dataset = source_loader(source_path,label_path, list_path = list_path, augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN)

    sourceloader = data.DataLoader(source_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    sourceloader_iter = iter(sourceloader)


    #Load new data for domain_transfer

    # optimizer for segmentation network


    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)




    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)


    epochs_since_start = 0
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_u_value = 0
        loss_l_value = 0
        loss_2_value = 0
        loss_mmd_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter,learning_rate)
            adjust_learning_rate(optimizer_D, i_iter,2e-4)

        # training loss for labeled data only
        try:
            batch = next(sourceloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(sourceloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            sourceloader_iter = iter(sourceloader)
            batch = next(sourceloader_iter)


        optimizer_D.zero_grad()
        for param in model_D.parameters():
            param.requires_grad = False

        images_source, labels_source, _, _ = batch

        images_source = images_source.cuda() #images:source image
        labels_source = labels_source.cuda().long() #labels:source labels

        try:
            batch = next(sourceloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(sourceloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            sourceloader_iter = iter(sourceloader)
            batch = next(sourceloader_iter)

        image_mix_source, label_mix_source, _, _ = batch

        image_mix_source = image_mix_source.cuda() #images_mix:source image mask
        label_mix_source = label_mix_source.cuda().long()

        lam = 0.9

        try:
            batch_target = next(targetloader_iter)
            if batch_target[0].shape[0] != batch_size:
                batch_target = next(targetloader_iter)
        except:
            targetloader_iter = iter(targetloader)
            batch_target = next(targetloader_iter)

        images_target, _ = batch_target
        images_target = images_target.cuda()
        images_target_cp=images_target.clone()
        max_probs_target, targets_pseudo_label=logits_fusion(model_for_reference,model_ema,images_target_cp,interp,mix_para)

        thresh=update_perf(max_probs_target,targets_pseudo_label,thresh)


        if stage == 'st1':
            for i in range(19):
                targets_pseudo_label[(max_probs_target <  conf[i]) * (targets_pseudo_label == i)] = 255


        try:
            batch_target = next(targetloader_iter)
            if batch_target[0].shape[0] != batch_size:
                batch_target = next(targetloader_iter)
        except:
            targetloader_iter = iter(targetloader)
            batch_target = next(targetloader_iter)
        images_mix_target, _ = batch_target
        images_mix_target = images_mix_target.cuda()
        images_target_mix_cp=images_mix_target.clone()
        max_probs_target_mix, target_mix_psuedo_label =logits_fusion(model_for_reference,model_ema,images_target_mix_cp,interp,mix_para)

        if stage=='st1':
            for i in range(19):
                target_mix_psuedo_label[(max_probs_target_mix <  conf[i]) * (target_mix_psuedo_label == i)] = 255


        MixMask_source=[]
        MixMask_lam_source=[]
        MixMask_tar=[]
        MixMask_tar_lam=[]

        if mix_mask == "class":
            for image_i in range(batch_size):
                classes = torch.unique(label_mix_source[image_i]).cpu().numpy()
                classes_choosen=classes[classes!=255]
                choose_prob=1-np.array(conf.copy())
                # print(choose_prob)
                choose_prob=choose_prob[classes_choosen]
                choose_prob = np.exp(choose_prob/tau) / np.sum(np.exp(choose_prob/tau))
                size=classes_choosen.size

                classes_tar = torch.unique(target_mix_psuedo_label[image_i]).cpu().numpy()
                # print(classes_tar)
                classes_choosen_tar=classes_tar[classes_tar!=255]
                choose_prob_tar=1-np.array(conf.copy())
                choose_prob_tar=choose_prob_tar[classes_choosen_tar]
                choose_prob_tar = np.exp(choose_prob_tar/tau) / np.sum(np.exp(choose_prob_tar/tau))
                size_tar=classes_choosen_tar.size
                if size<2:
                    MixMask_source.append(torch.zeros_like(label_mix_source[image_i]) .cuda())
                    MixMask_lam_source.append(MixMask_source[image_i] * lam)
                else:
                    size=np.max([int(size/2),2])
                    cls_to_use = np.random.choice(classes_choosen, size=size, p=choose_prob,replace=False)
                    for items in cls_to_use:
                        cut_instance_list[items] = cut_instance_list[items] + 1

                    cls_to_use = (torch.Tensor(cls_to_use).long()).cuda()
                    MixMask_source.append(transformmasks.generate_class_mask(labels_source[image_i], cls_to_use).unsqueeze(0).cuda())
                    MixMask_lam_source.append(MixMask_source[image_i] * lam)


                if size_tar<2:
                    MixMask_tar.append(torch.zeros_like(target_mix_psuedo_label[image_i]) .cuda())
                    MixMask_tar_lam.append(MixMask_tar[image_i] * lam)
                else:
                    size_tar=np.max([int(size_tar/2),2])
                    cls_to_use_tar = np.random.choice(classes_choosen_tar, size=size_tar, p=choose_prob_tar,replace=False)
                    for items in cls_to_use_tar:
                        cut_instance_list[items] = cut_instance_list[items] + 1
                    cls_to_use_tar = (torch.Tensor(cls_to_use_tar).long()).cuda()
                    MixMask_tar.append(transformmasks.generate_class_mask(target_mix_psuedo_label[image_i], cls_to_use_tar).unsqueeze(0).cuda())
                    MixMask_tar_lam.append(MixMask_tar[image_i] * lam)


        strong_parameters = {"Mix": MixMask_lam_source[0]}
        if random_flip:
            strong_parameters["flip"] = random.randint(0, 1)
        else:
            strong_parameters["flip"] = 0
        if color_jitter:
            strong_parameters["ColorJitter"] = random.uniform(0, 1)
        else:
            strong_parameters["ColorJitter"] = 0
        if gaussian_blur:
            strong_parameters["GaussianBlur"] = random.uniform(0, 1)
        else:
            strong_parameters["GaussianBlur"] = 0

        classes = [i for i in range(19)]

        conf_to_use = np.exp(conf) / np.sum(np.exp(conf))
        cls_to_use = np.random.choice(classes, size=2, p=conf_to_use)
        source_image_afmix=[]
        target_image_afmix=[]
        source_label_afmix=[]
        target_label_afmix=[]
        for i in range(batch_size):
            # print(target_mix_psuedo_label[i].shape)
            # print(labels_source[i].shape)

            source_image_afmixi, source_label_afmixi = strongTransform_class_mix( images_mix_target[i], images_source[i], target_mix_psuedo_label[i],
                                                                labels_source[i], MixMask_tar_lam[i], MixMask_tar[i],None, None,strong_parameters) #mask_image, origin_image, mask_label, origin_label, mask_for_iamge, mask_for_lbl
            target_image_afmixi, target_label_afmixi = strongTransform_class_mix(image_mix_source[i], images_target[i], label_mix_source[i], targets_pseudo_label[i],
                                                                MixMask_lam_source[i], MixMask_source[i],None, None, strong_parameters)
            source_image_afmix.append(source_image_afmixi)
            target_image_afmix.append(target_image_afmixi)
            source_label_afmix.append(source_label_afmixi)
            target_label_afmix.append(target_label_afmixi)
        source_image_afmix = torch.cat(source_image_afmix) #target_aftermix_strongaug
        target_image_afmix = torch.cat(target_image_afmix) #source_aftermix_strongaug

        source_label_afmix = torch.cat(source_label_afmix).long()
        target_label_afmix = torch.cat(target_label_afmix).long()

        pw_source_afmix = model(source_image_afmix)
        logits_source_afmix = interp(pw_source_afmix)

        pw_tar_afmix = model(target_image_afmix)
        logits_tar_afmix = interp(pw_tar_afmix)

        loss_mmd=0
        L_l2=0
        if pixel_weight == "threshold_uniform":
            unlabeled_weight = torch.sum(max_probs_target.ge(reweight_thresh).long() == 1).item() / np.size(np.array(targets_pseudo_label.cpu()))
            pixelWiseWeight_on_tar = unlabeled_weight * torch.ones(max_probs_target.shape).cuda()
            pixelWiseWeight_on_source = unlabeled_weight * torch.ones(max_probs_target.shape).cuda()
        elif pixel_weight == "threshold":
            pixelWiseWeight = max_probs_target.ge(0.968).float().cuda()
        else:
            pixelWiseWeight = torch.ones(max_probs_target.shape).cuda()

        onesWeights_on_tar = torch.ones((pixelWiseWeight_on_tar.shape)).cuda()
        onesWeights_on_source = torch.ones((pixelWiseWeight_on_tar.shape)).cuda()

        pixelWiseWeight_on_tars=[]
        pixelWiseWeight_on_sources=[]
        for i in range(batch_size):
            strong_parameters["Mix"] = MixMask_source[i]
            _, pixelWiseWeight_on_tari = strongTransform(strong_parameters, target = torch.cat((onesWeights_on_tar[i].unsqueeze(0),pixelWiseWeight_on_tar[i].unsqueeze(0))))
            pixelWiseWeight_on_tars.append(pixelWiseWeight_on_tari)

        for i in range(batch_size):
            strong_parameters["Mix"] = MixMask_tar[i]
            _, pixelWiseWeight_on_source_i = strongTransform(strong_parameters, target = torch.cat((pixelWiseWeight_on_source[i].unsqueeze(0),onesWeights_on_source[i].unsqueeze(0))))
            pixelWiseWeight_on_sources.append(pixelWiseWeight_on_source_i)

        pixelWiseWeight_a_tar = (torch.cat(pixelWiseWeight_on_tars)*lam).cuda()
        pixelWiseWeight_b_tar = (torch.cat((pixelWiseWeight_on_tars))*(1-lam)).cuda()

        pixelWiseWeight_a_source = (torch.cat(pixelWiseWeight_on_sources)*lam).cuda()
        pixelWiseWeight_b_source = (torch.cat((pixelWiseWeight_on_sources))*(1-lam)).cuda()



        L_l = consistency_weight * unlabeled_loss(logits_source_afmix, source_label_afmix, pixelWiseWeight_a_source).mean()
        + consistency_weight * unlabeled_loss(logits_source_afmix,labels_source, pixelWiseWeight_b_source).mean()
        L_l.backward()



        L_u = consistency_weight * unlabeled_loss(logits_tar_afmix, target_label_afmix, pixelWiseWeight_a_tar).mean()
        + consistency_weight * unlabeled_loss(logits_tar_afmix, targets_pseudo_label, pixelWiseWeight_b_tar).mean()

        loss_D_trg_fake = model_D(F.softmax(logits_tar_afmix, dim=1), 0).mean()
        loss = args.lambda_adv_target * loss_D_trg_fake + L_u


        loss_l_value += L_l.item()
        loss_2_value += 0
        loss_u_value += L_u.item()
        loss_mmd_value += 0

        loss.backward()
        optimizer.step()

        for param in model_D.parameters():
            param.requires_grad = True

        src_seg_score, trg_seg_score = logits_source_afmix.detach(), logits_tar_afmix.detach()

        loss_D_src_real = model_D(F.softmax(src_seg_score, dim=1), 0).mean()
        loss_D_trg_real = model_D(F.softmax(trg_seg_score, dim=1), 1).mean()

        loss_D = (loss_D_src_real + loss_D_trg_real) / 2

        loss_D_value=loss_D.item()
        loss_D.backward()

        optimizer_D.step()

        torch.cuda.empty_cache()


        base_lr=optimizer.param_groups[0]['lr']*1000
        base_lr_D=optimizer_D.param_groups[0]['lr']*10000

        if i_iter % 10 == 0 and i_iter!=0:
            print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}, lr x 103= {4:.3f}, lr_D lr x 104= {5:.3f}, unlabeled_weight = {6:.3f}, loss_adv = {7:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value, base_lr,base_lr_D,unlabeled_weight,loss_D_value))
        if i_iter%save_checkpoint_every==0 and i_iter!=0:
            _save_checkpoint(0, i_iter, model, optimizer,model_D,optimizer_D, config, ema_model=None, save_best=False,train_path=train_path)

        if i_iter % val_per_iter == 0 and i_iter != 0:
            model.eval()
            mIoU = evaluate(model, 'cityscapes', ignore_label=255, input_size=(1024, 2048), save_dir=checkpoint_dir,path=train_path,source_dataset_name=source_dataset_name,model_name=model_name,ablation=ablation,i_iter=i_iter)
            model.train()
            if mIoU > best_mIoU and save_best_model:
                best_mIoU = mIoU
                _save_checkpoint(mIoU,i_iter, model, optimizer,model_D,optimizer_D,config, ema_model=None, save_best=True,train_path=train_path)

            print('The best miou is %.4f' % best_mIoU)
    return model,optimizer,optimizer_D,best_mIoU,thresh

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')
    args = get_arguments()
    reweight_thresh=args.reweight_thresh
    config = json.load(open(args.config))
    restore_path=config['training']['restore_path']
    stage=config['training']['stage']

    model_name=config['model']
    num_classes=19
    checkpoint_dir=args.checkpoint_dir
    tau=args.tau

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir,restore_path.split('/')[-1]+"_"+stage)

    batch_size=config['training']['batch_size']
    print("Batchsize %d"%(batch_size))
    total_iterations = config['training']['num_iterations']
    iterations_each_epoch = config['training']['iterations_each_epoch']
    source_dataset_name=config['training']['source_dataset']['name']
    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))

    ablation="none"

    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    input_size = (h, w)
    print(restore_path)

    restore_from_T=os.path.join(restore_path,stage,"T.pth")
    restore_from_S=os.path.join(restore_path,stage,"S.pth")



    ignore_label = config['ignore_label']

    learning_rate = config['training']['learning_rate']
    print("learning_rate",learning_rate)

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable

    log_dir = checkpoint_dir
    val_per_iter = config['utils']['val_per_iter']
    save_checkpoint_every = config['utils']['save_checkpoint_every']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    if args.save_images:
        print('Saving unlabeled images')
        save_unlabeled_images = True
    else:
        save_unlabeled_images = False

    gpus = [0,1,2,3,4,5,6,7,8][:torch.cuda.device_count()]

    print(config)
    best_mIoU = 0

    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()

        else:
            unlabeled_loss = MSELoss2d().cuda()
    elif consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(
                CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda(), device_ids=gpus)
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()

    cudnn.enabled = True

    # create network
    if model_name == "DeepLab":
        args.lambda_adv_target = 0.001
        model_T = Res_Deeplab(num_classes=num_classes)
        model_S = Res_Deeplab(num_classes=num_classes)
    else:
        model_T = VGG16_FCN8s(num_classes=num_classes)
        model_S = VGG16_FCN8s(num_classes=num_classes)
        args.lambda_adv_target = 0.0001

    model_T = restore_model(restore_from_T, model_T)
    model_S = restore_model(restore_from_S, model_S).eval()
    model_Dt, optimizer_Dt = CreateDiscriminator()
    model_Ds, optimizer_Ds = CreateDiscriminator()

    model_Dt.cuda()
    model_Ds.cuda()





    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if len(gpus) > 1:
        if use_sync_batchnorm:
            model_T = convert_model(model_T)
            model_T = DataParallelWithCallback(model_T, device_ids=gpus)
            model_S = convert_model(model_S)
            model_S = DataParallelWithCallback(model_S, device_ids=gpus)

        else:
            model_T = torch.nn.DataParallel(model_T, device_ids=gpus)
            model_S = torch.nn.DataParallel(model_S, device_ids=gpus)


    model_T.cuda()
    model_S.cuda()




    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer_T = optim.SGD(model_T.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
            optimizer_S = optim.SGD(model_S.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer_T = optim.SGD(model_T.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
            optimizer_S = optim.SGD(model_S.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer_T = optim.Adam(model_T.module.optim_parameters(learning_rate_object),
                        lr=learning_rate,weight_decay=weight_decay)
            optimizer_S = optim.Adam(model_S.module.optim_parameters(learning_rate_object),
                        lr=learning_rate,weight_decay=weight_decay)
        else:
            optimizer_T = optim.Adam(model_T.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)
            optimizer_S = optim.Adam(model_S.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)


    data_path_ori=get_data_path('cityscapes','T',source_dataset_name,model_name,ablation)
    data_path_tran=get_data_path('cityscapes','S',source_dataset_name,model_name,ablation)
    if not os.path.exists(os.path.join(restore_path,stage)+'/perf_t.npy'):
        print('generating single path threshold')
        T_thresh,S_thresh=pseudo_gener(os.path.join(restore_path,stage,"S.pth"), os.path.join(restore_path,stage,"T.pth"), os.path.join(restore_path,stage), os.path.join(restore_path,stage), model_name,data_path_ori,data_path_tran)
    else:
        S_thresh = np.load(os.path.join(restore_path,stage)+ '/perf_s.npy')
        T_thresh = np.load(os.path.join(restore_path,stage)+ '/perf_t.npy')

    if not os.path.exists(os.path.join(restore_path,stage)+'/conf.npy'):
        print('generating dual path threshold')
        conf=soft_pseudo_gener(os.path.join(restore_path,stage,"S.pth"), os.path.join(restore_path,stage,"T.pth"), os.path.join(restore_path,stage), os.path.join(restore_path,stage), model_name,data_path_ori,data_path_tran)
    else:
        conf=np.load(os.path.join(restore_path,stage) + '/conf.npy')
        print("conf",conf)


    best_mIoU_T=0
    best_mIoU_S=0
    record_cut_instacne={}
    cut_instance_list=np.zeros(19).tolist()

    record_thresh={}

    for i in range(int(total_iterations*1.0/iterations_each_epoch)):
        st=i*(iterations_each_epoch+1)
        end=(i+1)*(iterations_each_epoch+1)
        record_thresh[str(st)+"_S"] = copy.deepcopy(S_thresh.tolist())
        record_thresh[str(st)+"_T"] = copy.deepcopy(T_thresh.tolist())
        print("S_thresh",S_thresh)
        print("T_thresh",T_thresh)
        w1 = np.exp(T_thresh) / (np.exp(S_thresh) + np.exp(T_thresh))
        print(w1)
        model_S,optimizer_S,optimizer_Ds,best_mIoU_S,S_thresh=do_epoch(train_path="S",start_iteration=st, num_iterations=end,model=model_S,model_ema=model_T, model_D=model_Ds,optimizer=optimizer_S,optimizer_D=optimizer_Ds,best_mIoU=best_mIoU_S,mix_para=1-w1,thresh=S_thresh)
        model_T,optimizer_T,optimizer_Dt,best_mIoU_T,T_thresh=do_epoch(train_path="T",start_iteration=st, num_iterations=end,model=model_T,model_ema=model_S,model_D=model_Dt,optimizer=optimizer_T,optimizer_D=optimizer_Dt,best_mIoU=best_mIoU_T,mix_para=w1,thresh=T_thresh)



