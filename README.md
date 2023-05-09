
This repo contains the offical implementations of the conference paper DPL and its journal version ADPL:
-  Dual Path Learning for Domain Adaptation of Semantic Segmentation [[Paper](https://arxiv.org/pdf/2108.06337.pdf)]
-  ADPL: Adaptive Dual Path Learning for Domain Adaptation of Semantic Segmentation [[Paper](https://ieeexplore.ieee.org/abstract/document/10050808)] 

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
|—— ADPL_master/
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

## Dual Path Learning for Domain Adaptation of Semantic Segmentation

### Requirements

- python 3.6
- torch==1.5
- torchvision==0.6
- Pillow==7.1.2

### Evaluation
Download pre-trained models from Pretrained_Resnet_GTA5 [[Google_Drive](https://drive.google.com/file/d/1fSr-Ijs5vG7DuksUWBdBWUCDTbNkIhHO/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1bWGHDqnTZ21aYdgTOSCu3g)(Code:t7t8)] and save the unzipped models in `./DPL_master/DPL_pretrained`, download translated target images from DPI2I_City2GTA_Resnet [[Google_Drive](https://drive.google.com/file/d/1rnO3OJGpW_m7GahxnqFPNbbVEFYFI_b5/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/15SVGHz-dWDboXszwBGZRxg)(Code:cf5a)] and save the unzipped images in `./DPL_master/DPI2I_images/DPI2I_City2GTA_Resnet/val`. Then you can evaluate DPL and DPL-Dual as following:
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

- GTA5->Cityscapes, FCN-8s with VGG16: GTA5_VGG_chpt [[Google_Drive](https://drive.google.com/file/d/1LVnJEE9uHCwSiymD8YWEPybfCKCAKTJr/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/18ONFBKH1t0pdCG_sueXvLQ)(Code:fanp)]
- SYNTHIA->Cityscapes, DeepLab-V2 with ResNet-101: SYN_Resnet_chpt [[Google_Drive](https://drive.google.com/file/d/1YMkUAQSAZyUHP1J8jpN12pMShHByP6bk/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1c48K9Ta8-ya_gchoKo1tVw)(Code:drvo)]
- SYNTHIA->Cityscapes, FCN-8s with VGG16: SYN_VGG_chpt [[Google_Drive](https://drive.google.com/file/d/1_f4bCMdbVzIXqFSjV7sT_hiPQHGY-Kgx/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1MR9FhbsX6VEf2BMOp_khFQ)(Code:9vio)]

### Training
The training process of DPL consists of two phases: single-path warm-up and DPL training. The training example is given on default setting: GTA5->Cityscapes, DeepLab-V2 with ResNet-101.

#### Quick start for DPL training

 Downlad pretrained ![1](http://latex.codecogs.com/svg.latex?M_{S}^{(0)}) and ![1](http://latex.codecogs.com/svg.latex?M_{T}^{(0)}) [[Google_Drive](https://drive.google.com/file/d/1NLKn8XwVsfC6JrgWficGBjTKRThAhULW/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1JIiYxp75LMGF_fHNG8xttQ)(Code: 3ndm)], save ![1](http://latex.codecogs.com/svg.latex?M_{S}^{(0)}) to `path_to_model_S`, save ![1](http://latex.codecogs.com/svg.latex?M_{T}^{(0)}) to `path_to_model_T`, then you can train DPL as following:

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

    3.3. Update `path_to_model_S`with path to best ![1](http://latex.codecogs.com/svg.latex?M_{S}) model, update `path_to_model_T`with path to best ![1](http://latex.codecogs.com/svg.latex?M_{T}) model, adjust parameter `threshenlen` to 0.25, then repeat 3.1-3.2 for 3 more rounds.

### Single path warm up
If you want to train DPL from the very begining, training example of single path warm up is also provided as below:
<details>
<summary>
    <b>Single Path Warm-up</b>
</summary>

Download ![1](http://latex.codecogs.com/svg.latex?M_{S}^{(0)}) trained with labeled source dataset Source_only [[Google_Drive](https://drive.google.com/file/d/1tYldAGj1_JsgoPi1b09ZRYqGFdbHCSvU/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1T2a-BX1E6NoEKa3uh3QP4w)(Code:fjdw)].

1.  Train original cycleGAN (without Dual Path Image Translation).
    ```
    cd CycleGAN_DPL
    python train.py --dataroot ../data --name ori_cycle --A_setroot GTA5/images --B_setroot Cityscapes/leftImg8bit/train --model cycle_diff --lambda_semantic 0
    ```
2.  Generate transferred GTA5->Cityscapes images with original cycleGAN.

    ```
    python test.py --name ori_cycle --no_dropout --load_size 1024 --crop_size 1024 --preprocess scale_width --dataroot ../data/GTA5/images --model_suffix A  --results_dir path_to_ori_cycle_GTA52cityscapes
    ```

3. Before warm up, pretrain ![1](http://latex.codecogs.com/svg.latex?M_{T}) without SSL and restore the best checkpoint in `path_to_pretrained_T`:
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

4.4 Update `path_to_pretrained_T` with  path to best model in 4.3, repeat 4.1-4.3 for one more round.  

</details>


### More Experiments
- For SYNTHIA to Cityscapes scenario, please train DPL with "--source synthia" and change the data path.
- For training on "FCN-8s with VGG16", please train DPL with "--model VGG". 



## ADPL: Adaptive Dual Path Learning for Domain Adaptation of Semantic Segmentation

###  Requirements
Please install packages by:
```
conda create --name ADPL_env python=3.6
conda activate ADPL_env
pip install -r requirements.txt
```

### Evaluation
Download pre-trained models from [Google_Drive](https://drive.google.com/file/d/1qg86mo7pLPizVzWhgsKbGpIjymsggA5-/view?usp=share_link) or [BaiduYun](https://pan.baidu.com/s/1oe0L4rdjGq1K1Zzm3L49lw)(Code:4r6k), and save the unzipped models in `./ADPL_master/ADPL_pretrained/Deep_GTA5`, download translated target images from DPI2I_City2GTA_Resnet [[Google_Drive](https://drive.google.com/file/d/1rnO3OJGpW_m7GahxnqFPNbbVEFYFI_b5/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/15SVGHz-dWDboXszwBGZRxg)(Code:cf5a)] and save the unzipped images in `../data/Deep_GTA/target/val`. Then you can evaluate ADPL and ADPL-Dual as following:
- Evaluation of ADPL
    ```
    cd ADPL_master
    python evaluation.py 
    ```
- Evaluation of ADPL-Dual
    ```
    python evaluation_ADPL.py
    ``` 

More pretrained models and translated target images on other settings can be downloaded from [Google_Drive](https://drive.google.com/drive/folders/1S0G5J7YoLrOZvVwQPGvawiAVJTroBFeA?usp=share_link) or [BaiduYun](https://pan.baidu.com/s/1ytJNzhSyHYp5DRNg5Sn4tg)(Code:nn4a).
when evaluating VGG models, please add command option `--model_name VGG`
when the source is synhtia, please add command option `--source_dataset_name synthia`



### Training
The training process of ADPL consists of two phases: single-path warm-up and ADPL training. Since ADPL and DPL share the identical warm up stage, we only supplement the ADPL training instrucions in this section, please refer to "Single path warm up" for warm up training. The training example is given on default setting: GTA5->Cityscapes, DeepLab-V2 with ResNet-101.

 ADPL training consists of two parts: dual path image translation (DPIT) module training and dual path adaptive segmentation (DPAS) module training. The training process of DPIT is shared by ADPL and DPL, so please refer to the first two steps in "Quick start for DPL training" to train DPIT and generate translated images. To train the DPAS, please update the translated images paths in `ADPL/data/__init__.py` following the corresponding comment, place ![1](http://latex.codecogs.com/svg.latex?M_{S}^{(0)}) and ![1](http://latex.codecogs.com/svg.latex?M_{T}^{(0)}) in `./ADPL/chpt/Deep_GTA/st1`, then run the following commands to finish the 1-stage training:
 ```
    cd ADPL_master
    python train.py --config ./configs/configUDA_gta_deep_st1.json
```
For the 2-stage training, please place the best ![1](http://latex.codecogs.com/svg.latex?M_{S}) model and  best ![1](http://latex.codecogs.com/svg.latex?M_{T}) model of stage 1 in `./ADPL/chpt/Deep_GTA/st2`, then the 2nd stage can be trained by:
 ```
    python train.py --config ./configs/configUDA_gta_deep_st2.json --reweight_thresh 0
```

## Citation

If you find our paper and code useful in your research, please consider giving a star and citation. 

```
@inproceedings{cheng2021dual,
  title={Dual Path Learning for Domain Adaptation of Semantic Segmentation},
  author={Cheng, Yiting and Wei, Fangyun and Bao, Jianmin and Chen, Dong and Wen, Fang and Zhang, Wenqiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9082--9091},
  year={2021}
}

@ARTICLE{cheng2023ADPL,
  author={Cheng, Yiting and Wei, Fangyun and Bao, Jianmin and Chen, Dong and Zhang, Wenqiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={ADPL: Adaptive Dual Path Learning for Domain Adaptation of Semantic Segmentation}, 
  year={2023},
  pages={1-17},
  doi={10.1109/TPAMI.2023.3248294}}
```
