# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageOps
import torch

class CCompose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask,centroid):
        img, mask = Image.fromarray(img, mode="RGB"), Image.fromarray(mask, mode="L")
        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask, centroid)
        return np.array(img), np.array(mask, dtype=np.uint8)

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask=None,lsoft=None,params=None):
        if mask is not None:
            mask = mask.resize(img.size, Image.NEAREST)
            assert img.size == mask.size
            for a in self.augmentations:
                img, mask = a(img, mask)
            return img, mask
        elif lsoft is not None:
            lpsoft = torch.from_numpy(np.transpose(lsoft,(2,0,1))).float()
            # lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[img.size[1], img.size[0]], mode='bilinear', align_corners=True)[0]
            for a in self.augmentations:
                img, lpsoft ,param= a(img, lpsoft=lpsoft)
            return img, lpsoft,param
        else:
            for a in self.augmentations:
                img, _ ,param= a(img,params=params)
            return img,param

class CentroidCCrop(object):
    """
    Take a random crop from the image.

    First the image or crop size may need to be adjusted if the incoming image
    is too small...

    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image

    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """
    def __init__(self, size, ignore_index=0, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size
        w, h = img.size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            return img, mask

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))



class RandomCrop(object):
    def __init__(self, size, padding=0):
        #if isinstance(size, numbers.Number):
            #self.size = (int(size), int(size))
        #else:
            #self.size = size
        self.size = tuple(size)
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                mask.resize((tw, th), Image.NEAREST),
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )



class RandomCrop_city(object):  # used for results in the CVPR-19 submission
    def __init__(self, size, padding=0):
        #if isinstance(size, numbers.Number):
            #self.size = (int(size), int(size))
        #else:
            #self.size = size
        self.size = tuple(size)
        self.padding = padding

    def __call__(self, img, mask=None,lpsoft=None,params=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            if mask:
                mask = ImageOps.expand(mask, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size

        if w>1024 and h>512 and w!=1280:
            rzw=int(w/2)
            rzh=int(h/2)
            img = img.resize((rzw, rzh), Image.BILINEAR)
            if mask:
                mask = mask.resize((rzw, rzh), Image.NEAREST)
        else:
            rzw=w
            rzh=h
        if params==None:
            x1 = random.randint(0, rzw - tw)
            y1 = random.randint(0, rzh - th)
        else:
            y1, _, x1, _=params

        if lpsoft is not None:
            lpsoft = lpsoft[:, y1:y1 + th, x1:x1 + tw]
        params = (y1, y1 + th, x1, x1 + tw)
        if mask:
            return (
                img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th))

            )
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            lpsoft,
            params
        )

class RandomCrop_gta(object):  # used for results in the CVPR-19 submission
    def __init__(self, size, padding=0):
        #if isinstance(size, numbers.Number):
            #self.size = (int(size), int(size))
        #else:
            #self.size = size
        self.size = tuple(size)
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size

        '''
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                mask.resize((tw, th), Image.NEAREST),
            )
        '''
        #img = img.resize((600, 300), Image.BILINEAR)
        #mask = mask.resize((600, 300), Image.NEAREST)
        #img = img.resize((512, 256), Image.BILINEAR)
        #mask = mask.resize((512, 256), Image.NEAREST)

        x1 = random.randint(0, int(w) - tw)
        y1 = random.randint(0, int(h) - th)

        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class RandomCrop_city_gnet(object):  # used for gnet training
    def __init__(self, size, padding=0):
        #if isinstance(size, numbers.Number):
            #self.size = (int(size), int(size))
        #else:
            #self.size = size
        self.size = tuple(size)
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        '''
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                mask.resize((tw, th), Image.NEAREST),
            )
        '''
        #img = img.resize((int(w/2), int(h/2)), Image.BILINEAR)
        #mask = mask.resize((int(w/2), int(h/2)), Image.NEAREST)
        img = img.resize((600, 300), Image.BILINEAR)
        mask = mask.resize((600, 300), Image.NEAREST)
        #img = img.resize((512, 256), Image.BILINEAR)
        #mask = mask.resize((512, 256), Image.NEAREST)

        x1 = random.randint(0, 600 - tw)
        y1 = random.randint(0, 300 - th)

        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )

class CenterCrop(object):
    def __init__(self, size):
        '''
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        '''
        self.size = tuple(size)

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            img.resize(self.size, Image.BILINEAR),
            mask.resize(self.size, Image.NEAREST),
        )


class Scale(object):
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, img, mask):
        assert img.size == mask.size
        f_w, f_h = self.size
        w, h = img.size
        if (w >= h and w == f_w) or (h >= w and h == f_h):
            return img, mask
        if w > h:
            ow = f_w
            oh = int(f_w * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = f_h
            ow = int(f_h * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )


class RSCrop(object):
    def __init__(self, size):
        self.size = size
        #self.size = tuple(size)

    def __call__(self, img, mask):
        assert img.size == mask.size
        #for attempt in range(10):
        #random scale (0.5 to 2.0)
        crop_size = self.size
        short_size = random.randint(int(self.size*0.5), int(self.size*2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        #deg = random.uniform(-10, 10)
        #img = img.rotate(deg, resample=Image.BILINEAR)
        #mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))

        return img, mask


class RSCrop_city(object):
    def __init__(self, size):
        #self.size = size
        self.size = tuple(size)
        self.base_size = 1024

    def __call__(self, img, mask):
        assert img.size == mask.size
        #for attempt in range(10):
        #random scale (0.5 to 2.0)
        #crop_size = self.size
        short_size = random.randint(int(self.base_size*0.25), int(self.base_size*1.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)

        '''
        # pad crop
        #if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        '''
        # random crop crop_size
        #w, h = img.size
        x1 = random.randint(0, w - self.size[0])
        y1 = random.randint(0, h - self.size[1])
        img = img.crop((x1, y1, x1+self.size[0], y1+self.size[1]))
        mask = mask.crop((x1, y1, x1+self.size[0], y1+self.size[1]))

        return img, mask

class RandomSizedCrop(object):
    def __init__(self, size):
        #self.size = size
        self.size = tuple(size)

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            f_w, f_h = self.size

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((f_w, f_h), Image.BILINEAR),
                    mask.resize((f_w, f_h), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            img.rotate(rotate_degree, Image.BILINEAR),
            mask.rotate(rotate_degree, Image.NEAREST),
        )


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )

        return self.crop(*self.scale(img, mask))
