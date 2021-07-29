import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2


class SYN_SSLDataSet(data.Dataset):
    def __init__(self, root, list_path,label_root, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.label_root = label_root
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip()[4:] for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        name = self.img_ids[0]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):

        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "%s" % name)).convert('RGB')
        image = image.resize(self.crop_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        label = Image.open(osp.join(self.label_root, "%s" % name))
        label = label.resize(self.crop_size, Image.NEAREST)
        label = np.asarray(label, np.float32)
        return image.copy(), label.copy(), np.array(size), name
        # re-assign labels to match the format of Cityscapes






