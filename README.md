# Dual Path Learning for Domain Adaptation of Semantic Segmentation
A [PyTorch](http://pytorch.org/) implementation of DPL.

To appear at CVPR 2021. [arXiv preprint]()

<!--

If you use this code in your research please consider citing
>@article{Citing
} 
-->
## Requirements

- Pytorch 3.6
- torch==1.5
- torchvision==0.6
- Pillow==7.1.2

## Dataset Preparations
For GTA5->Cityscapes scenario, download: 
- Source dataset: [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) 
- Target dataset: [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

For further evaluation on SYNTHIA->Cityscapes scenario, download:
- Source dataset: [SYNTHIA dataset](http://synthia-dataset.net/download/808/) (SYNTHIA-RAND-CITYSCAPES)

The folder should be structured as:
```
|DPL
|—— DPL_master/
|—— CycleGAN_DPL/
|—— data/
│   ├—— Cityscapes/  
|   |   ├—— data/
|   |       ├—— gtFine/
|   |       ├—— leftImg8bit/
│   ├—— GTA5/
|   |   ├—— images/
|   |   ├—— labels/
|   |   ├—— ...
│   ├—— synthia/ 
|   |   ├—— RGB/
|   |   ├—— GT/
|   |   ├—— Depth/
|   |   ├—— ...
```


## Evaluation
Download pre-trained models from [Pretrained_Resnet_GTA5](https://drive.google.com/file/d/1fSr-Ijs5vG7DuksUWBdBWUCDTbNkIhHO/view?usp=sharing) and save the unzipped models in `./DPL_master/DPL_pretrained`, download translated target images from [DPI2I_City2GTA_Resnet](https://drive.google.com/file/d/1rnO3OJGpW_m7GahxnqFPNbbVEFYFI_b5/view?usp=sharing) and save the unzipped images in `./DPL_master/DPI2I_images/DPI2I_City2GTA_Resnet/val`. Then you can evaluate DPL and DPL-Dual as following:
- Evaluation of DPL
    ```
    cd DPL_master
    python evaluation.py --init-weights ./DPL_pretrained/Resnet_GTA5_DPLst4_T.pth --save path_to_DPL_results/results --log-dir path_to_DPL_results
    ```
- Evaluation of DPL-Dual
    ```
    python evaluation_DPL.py --data-dir-targetB ./DPI2I_images/DPI2I_City2GTA_Resnet --init-weights_S ./DPL_pretrained/Resnet_GTA5_DPLst4_S.pth --init-weights_T ./DPL_pretrained/Resnet_GTA5_DPLst4_T.pth --save path_to_DPL_dual_results/results --log-dir path_to_DPL_dual_results
    ``` 

More pretrained models and translated target images on other settings can be downloaded from:

- GTA5->Cityscapes, FCN-8s with VGG16: [GTA5_VGG_chpt](https://drive.google.com/file/d/1LVnJEE9uHCwSiymD8YWEPybfCKCAKTJr/view?usp=sharing)
- SYNTHIA->Cityscapes, DeepLab-V2 with ResNet-101: [SYN_Resnet_chpt](https://drive.google.com/file/d/1YMkUAQSAZyUHP1J8jpN12pMShHByP6bk/view?usp=sharing)
- SYNTHIA->Cityscapes, FCN-8s with VGG16: [SYN_VGG_chpt](https://drive.google.com/file/d/1_f4bCMdbVzIXqFSjV7sT_hiPQHGY-Kgx/view?usp=sharing)
## Training
 
### Single Path Warm-up


Download ![1](http://latex.codecogs.com/svg.latex?M_{S}^{(0)}) trained with labeled source dataset [Source_only](https://drive.google.com/file/d/1tYldAGj1_JsgoPi1b09ZRYqGFdbHCSvU/view?usp=sharing).

1.  Train original cycleGAN (without Dual Path Image Translation).
    ```
    cd CycleGAN_DPL
    python train.py --dataroot ../data --name ori_cycle --A_setroot GTA5/images --B_setroot Cityscapes/leftImg8bit/train --model cycle_diff --lambda_semantic 0
    ```
2.  Generate transferred GTA5->Cityscapes images with original cycleGAN.

    ```
    python test.py --name ori_cycle --no_dropout --load_size 1024 --crop_size 1024 --preprocess scale_width --dataroot ../data/GTA5/images --model_suffix A  --results_dir path_to_ori_cycle_GTA52cityscapes
    ```
3. Train target model for warm up of ![1](http://latex.codecogs.com/svg.latex?M_{T}) and restore the best checkpoint in `path_to_pretrained_T`:
    ```
    cd ../DPL_master
    python DPL.py --snapshot-dir snapshots/pretrain_T --init-weights path_to_initialization_S --data-dir path_to_ori_cycle_GTA52cityscapes
    ```
4. Warm up ![1](http://latex.codecogs.com/svg.latex?M_{T}). 
    
    4.1. Generate labels on source dataset with label correction.
    ```
    python SSL_source.py --set train --data-dir path_to_ori_cycle_GTA52cityscapes --init-weights path_to_pretrained_T --threshdelta 0.3 --thresh 0.9 --threshlen 0.65 --save path_to_corrected_label_step1_or_step2 
    ```
    4.2. Generate pseudo labels on target dataset.
    ```
    python SSL.py --set train --data-list-target ./dataset/cityscapes_list/train.txt --init-weights path_to_pretrained_T  --thresh 0.9 --threshlen 0.65 --save path_to_pseudo_label_step1_or_step2 
    ```
    4.3. Train  ![1](http://latex.codecogs.com/svg.latex?M_{T}) with label correction.
    
    ```
    python DPL.py --snapshot-dir snapshots/label_corr_step1_or_step2 --data-dir path_to_ori_cycle_GTA52cityscapes --source-ssl True --source-label-dir path_to_corrected_label_step1_or_step2 --data-label-folder-target path_to_pseudo_label_step1_or_step2 --init-weights path_to_pretrained_T          
    ```

   4.4 Update `path_to_pretrained_T` with  path to best model in 4.3, repeat 4.1-4.3 for 1 more round.  
### DPL training
1. Train dual path image generation module.

    ```
    cd ../CycleGAN_DPL
    python train.py --dataroot ../data --name dual_path_I2I --A_setroot GTA5/images --B_setroot Cityscapes/leftImg8bit/train --model cycle_diff --lambda_semantic 1 --init_weights_S path_to_model_S --init_weights_T path_to_model_T
    ```
2. Generate transferred images with dual path image generation module.
   - Generate transferred GTA5->Cityscapes images.
   
   ```
   python test.py --name dual_path_I2I --no_dropout --load_size 1024 --crop_size 1024 --preprocess scale_width --dataroot ../data/GTA5/images --model_suffix A  --results_dir DPI2I_path_to_GTA52cityscapes
   ```
   - Generate transferred Cityscapes->GTA5 images.
   ```
    python test.py --name dual_path_I2I --no_dropout --load_size 1024 --crop_size 1024 --preprocess scale_width --dataroot ../data/Cityscapes/leftImg8bit/train --model_suffix B  --results_dir DPI2I_path_to_cityscapes2GTA5/train
    
    python test.py --name dual_path_I2I --no_dropout --load_size 1024 --crop_size 1024 --preprocess scale_width --dataroot ../data/Cityscapes/leftImg8bit/val --model_suffix B  --results_dir DPI2I_path_to_cityscapes2GTA5/val
    ```

3. Train dual path adaptive segmentation module

    3.1. Generate dual path pseudo label.
    
    ```
    cd ../DPL_master
    python DP_SSL.py --save path_to_dual_pseudo_label_stepi --init-weights_S path_to_model_S --init-weights_T path_to_model_T --thresh 0.9 --threshlen 0.3 --data-list-target ./dataset/cityscapes_list/train.txt --set train --data-dir-targetB DPI2I_path_to_cityscapes2GTA5 --alpha 0.5
    ```
    
    3.2. Train ![1](http://latex.codecogs.com/svg.latex?M_{S})  and ![1](http://latex.codecogs.com/svg.latex?M_{T}) with dual path pseudo label respectively.

    ```
    python DPL.py --snapshot-dir snapshots/DPL_modelS_step_i --data-dir-target DPI2I_path_to_cityscapes2GTA5 --data-label-folder-target path_to_dual_pseudo_label_stepi --init-weights path_to_model_S --domain S
    ```

    ```
    python DPL.py --snapshot-dir snapshots/DPL_modelT_step_i --data-dir DPI2I_path_to_GTA52cityscapes --data-label-folder-target path_to_dual_pseudo_label_stepi --init-weights path_to_model_T
    ```




    3.3. Update `path_to_model_S`with path to best ![1](http://latex.codecogs.com/svg.latex?M_{S}) model, update `path_to_model_T`with path to best ![1](http://latex.codecogs.com/svg.latex?M_{T}) model, then repeat 3.1-3.2 for 3 more rounds.

## More Experiments
- For SYNTHIA to Cityscapes scenario, please train DPL with "--source synthia" and change the data path.
- For training on "FCN-8s with VGG16", please train DPL with "--model VGG". 

## Acknowledgment
This code is heavily borrowed from [BDL](https://github.com/liyunsheng13/BDL).