import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.cityscapes_loader_pd import cityscapesLoader_pd
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SynthiaDataSet
from data.BDDS_loader import BDDSLoader
from data.BDDS_pd import BDDS_pd


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader_pd,
        "cityscapes_label":cityscapesLoader,
        "BDDS": BDDS_pd,
        "BDDS_label": BDDSLoader,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSet
    }[name]

def get_data_path(name,path='T',source_dataset_name='gta',model='DeepLab',ablation='none'):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        if path == "T":
            return '../data/Cityscapes/'
        if source_dataset_name=='gta':
            if model=='DeepLab':
                return '../data/Deep_GTA/target'#update with ''DPI2I_path_to_GTA52cityscapes'' under DeepLab backbone
            else:
                return '../data/VGG_GTA/target'#update with ''DPI2I_path_to_GTA52cityscapes'' under VGG backbone
        elif source_dataset_name=='synthia':
            if model=='DeepLab':
                return '../data/Deep_SYN/target'#update with ''DPI2I_path_to_SYN2cityscapes'' under DeepLab backbone
            else:
                return '../data/VGG_SYN/target' #update with ''DPI2I_path_to_SYN2cityscapes'' under VGG backbone


    if name == 'gta':
        if path == "S":
            return '../data/GTA5/images','../data/GTA5'
        if model == 'DeepLab':
            return '../data/Deep_GTA/source', '../data/GTA5'#update with ''DPI2I_path_to_cityscapes2GTA5'' under DeepLab backbone
        else:
            return '../data/VGG_GTA/source', '../data/GTA5'#update with ''DPI2I_path_to_cityscapes2GTA5'' under VGG backbone


    if name == 'synthia':
        if path == "S":
            return '../data/synthia/RGB','../data/synthia'
        if model == 'DeepLab':
            return '../data/Deep_SYN/source', '../data/synthia'#update with ''DPI2I_path_to_cityscapes2syn'' under DeepLab backbone
        else:
            return '../data/VGG_SYN/source', '../data/synthia'#update with ''DPI2I_path_to_cityscapes2syn'' under VGG backbone

