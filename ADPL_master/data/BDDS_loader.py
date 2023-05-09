import os
import torch
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data

from data.city_utils import recursive_glob
from data.augmentations import *
class BDDSLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))


    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        img_norm=False,
        augmentations=None,
        version="cityscapes",
        return_id=False,
        psuedo_root=None,
        psoft=False,
        img_mean = np.array([104.00698793, 116.66876762, 122.67891434])
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.psuedo_root=psuedo_root
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = img_mean
        self.files = {}

        self.images_base = os.path.join(self.root, self.split)
        self.annotations_base = os.path.join(
            '../data/bdd100k/labels', self.split
        )

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")
        self.files[split].sort()
        self.psoft=psoft
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))
        print(self.images_base)
        if self.psuedo_root is not None:
            print(self.psuedo_root)

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        self.return_id = return_id

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        if self.split=='train':
            if self.psoft:
                psuedo_path = os.path.join(
                    self.psuedo_root,
                    os.path.basename(img_path.replace('.png', '.npy'))
                )
            else:
                psuedo_path = os.path.join(
                    self.psuedo_root,
                    os.path.basename(img_path))
        else:

            lbl_path = os.path.join(
                self.annotations_base,
                os.path.basename(img_path).replace('.jpg', '.png')
            )
            if not os.path.exists(lbl_path):
                lbl_path = os.path.join(
                    self.annotations_base,
                    img_path.split('/')[-1].split('_')[0],  # temporary for cross validation
                    os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
                )

        img = Image.open(img_path).convert('RGB')

        if self.psoft:
            psuedo_lbl = np.load(psuedo_path)
        elif self.split == 'train':
            psuedo_lbl = Image.open(psuedo_path)
        else:
            lbl = Image.open(lbl_path)
            lbl = np.array(lbl, dtype=np.uint8)
#img(1024,2048)->
        # [augmentatiton:
        #   [resizeto(512,1024)]]
        #    crop(512,512) with para]


        if self.augmentations is not None:
            if self.psoft:
                img, psuedo_lbl,params = self.augmentations(img, lsoft=psuedo_lbl)
            elif self.psuedo_root:
                img, psuedo_lbl = self.augmentations(img, lbl=psuedo_lbl)
            else:
                img, lbl = self.augmentations(img, lbl=lbl)

        # if self.psoft==False and self.psuedo_root==None:
        #     lbl = self.encode_segmap(lbl)

        if self.transform is not None:
            if self.psoft:
                img, psuedo_lbl = self.transform(img, lpsoft=psuedo_lbl)

            elif self.psuedo_root:
                img, psuedo_lbl = self.transform(img,lpsoft=psuedo_lbl)
            else:
                img, lbl = self.transform(img, lbl=lbl)
        img_name = img_path.split('/')[-1]
        if self.return_id:
            return img, lbl, img_name, img_name, index
        if self.psuedo_root:
            if self.psoft:
                return img, psuedo_lbl, img_path, psuedo_path, img_name,params
            return img, psuedo_lbl, img_path, psuedo_path, img_name

        return img, lbl, img_path, lbl_path, img_name



    def transform(self, img, lbl=None, lpsoft=None):
        """transform

        :param img:
        :param lbl:
        """
        img = img.resize((self.img_size[1],self.img_size[0]), Image.BICUBIC)
        img = np.asarray(img, np.float32)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        if lbl is not None:
            lbl = np.asarray(lbl)
            lbl = lbl.astype(int)
            classes = np.unique(lbl)
            # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
            # lbl = lbl.astype(int)
            # if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            #     print("after det", classes, np.unique(lbl))
            #     raise ValueError("Segmentation map contained invalid class values")
            lbl = torch.from_numpy(lbl).long()
        img = torch.from_numpy(img).float()
        if lbl is not None:
            return img,lbl
        return img, lpsoft

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

'''
if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip()])

    local_path = "./data/city_dataset/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
'''
