from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.synthia_dataset import SYNDataSet
from data.cityscapes_dataset_label import cityscapesDataSetLabel
from data.gta5_dataset_SSL import GTA5DataSetSSL
from data.synthia_SSL import SYN_SSLDataSet
import numpy as np
from torch.utils import data

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
image_sizes = {'cityscapes': (1024, 512), 'gta5': (1024, 512), 'synthia': (1024, 512)}


def CreateSrcDataLoader(args, max_iters=0):
    if args.source == 'gta5':
        if args.source_ssl == True:
            source_dataset = GTA5DataSetSSL(args.data_dir, args.data_list, args.source_label_dir,
                                         max_iters=args.num_steps * args.batch_size,
                                         crop_size=image_sizes['gta5'], mean=IMG_MEAN)

        else:
            source_dataset = GTA5DataSet(args.data_dir, args.data_list, args.source_label_dir,
                                         max_iters=args.num_steps * args.batch_size,
                                         crop_size=image_sizes['gta5'], mean=IMG_MEAN)
    elif args.source == 'synthia':
        if args.source_ssl == True:
            source_dataset = SYN_SSLDataSet(args.data_dir, args.data_list, args.source_label_dir,
                                        max_iters=args.num_steps * args.batch_size,
                                        crop_size=image_sizes['synthia'], mean=IMG_MEAN)
        else:
            source_dataset = SYNDataSet(args.data_dir, args.data_list,args.source_label_dir, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['synthia'], mean=IMG_MEAN)
    else:
        raise ValueError('The target dataset mush be either gta5 or synthia')

    source_dataloader = data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True)

    return source_dataloader


def CreateTrgDataLoader(args, translated=False):

    if args.data_label_folder_target is not None:
        target_dataset = cityscapesDataSetLabel(args.data_dir_target, args.data_list_target,
                                                max_iters=args.num_steps * args.batch_size,
                                                crop_size=image_sizes['cityscapes'], mean=IMG_MEAN,
                                                set=args.set, label_folder=args.data_label_folder_target,translated=translated)
    else:
        if args.set == 'train':
            target_dataset = cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                               max_iters=args.num_steps * args.batch_size,
                                               crop_size=image_sizes['cityscapes'], mean=IMG_MEAN, set=args.set,translated=translated)
        else:
            target_dataset = cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                               crop_size=image_sizes['cityscapes'], mean=IMG_MEAN, set=args.set,translated=translated)

    if args.set == 'train':
        target_dataloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, pin_memory=True)
    else:
        target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)

    return target_dataloader


def CreateTrgDataSSLLoader(args,translated=False):
    target_dataset = cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                       crop_size=image_sizes['cityscapes'], mean=IMG_MEAN, set=args.set,translated=translated)
    target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)
    return target_dataloader




def CreateSrcDataSSLLoader(args):
    if args.source == 'gta5':
        source_dataset = GTA5DataSet(args.data_dir, args.data_list, args.source_label_dir,
                                     max_iters=None,
                                     crop_size=image_sizes['gta5'], mean=IMG_MEAN)
        source_dataloader = data.DataLoader(source_dataset, batch_size=1, shuffle=False, pin_memory=True)
        return source_dataloader
    else:
        source_dataset = SYNDataSet(args.data_dir, args.data_list, args.source_label_dir,
                                     max_iters=None,
                                     crop_size=image_sizes['synthia'], mean=IMG_MEAN)
        source_dataloader = data.DataLoader(source_dataset, batch_size=1, shuffle=False, pin_memory=True)
        return source_dataloader