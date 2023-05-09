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
from torchvision import transforms
class SynthiaDataSet(data.Dataset):
    def __init__(self, root, label_path,list_path, max_iters=None, augmentations = None, img_size=(321, 321), mean=(128,128,128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.img_size = img_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.augmentations = augmentations
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip()[-11:] for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.label_path=label_path


        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        #for name in self.img_ids:
        #    img_file = osp.join(self.root, "RGB/%s" % name)
        #    label_file = osp.join(self.root, "GT/LABELS/%s" % name)
        #    self.files.append({
        #        "img": img_file,
        #        "label": label_file,
        #        "name": name
        #    })
        print(self.root,self.label_path)

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "%s" % name)).convert('RGB')

        #image = Image.open(osp.join('../../dataset/syn_deeplab', "RGB/%s" % name)).convert('RGB')
        # label = np.asarray(imageio.imread(osp.join(self.label_path, "GT/LABELS/%s" % name), format='PNG-FI'))[:,:,0]  # uint16
        # label = Image.fromarray(label)

        label = cv2.imread(osp.join(self.label_path, "GT/LABELS/%s" % name), -1)
        label = label[:, :, -1]

        # resize
        image = image.resize(self.img_size, Image.BICUBIC)
        label = cv2.resize(label, self.img_size, interpolation=cv2.INTER_NEAREST)
        label = Image.fromarray(label)


        if self.augmentations is not None:
            image, label = self.augmentations(image, label)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
