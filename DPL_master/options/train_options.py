import argparse
import os.path as osp
class TrainOptions():
    def initialize(self):

        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("--model", type=str, default='DeepLab',help="available options : DeepLab and VGG")
        parser.add_argument("--GPU", default='0,1', help="which GPU to use")
        parser.add_argument("--batch-size", type=int, default=4, help="input batch size.")
        parser.add_argument("--num-workers", type=int, default=0, help="number of threads.")

        #check_piont_strorage
        parser.add_argument("--snapshot-dir", type=str, default='snapshots',
                            help="Where to save snapshots of the model.")

        #begin-------------dataset_setting-------------------------------------------------------------------
        #source dir
        parser.add_argument("--source", type=str, default='gta5', help="source dataset : gta5 or synthia")
        parser.add_argument("--data-dir", type=str, default='../data/GTA5/images',
                            help="Path to the directory containing the source dataset.")
        parser.add_argument("--source-label-dir", type=str, default='../data/GTA5/labels',
                            help="Path to the directory containing the source dataset.")
        parser.add_argument("--data-list", type=str, default='./dataset/gta5_list/train.txt',
                            help="Path to the listing of images in the source dataset.")
        parser.add_argument("--source-ssl", type=bool, default=False,
                            help="Path to the listing of images in the source dataset.")
        #target dir
        parser.add_argument("--target", type=str, default='cityscapes', help="target dataset : cityscapes")
        parser.add_argument("--data-dir-target", type=str, default='../data/Cityscapes',
                            help="Path to the directory containing the target dataset.")
        parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.")
        parser.add_argument("--data-list-target", type=str, default='./dataset/cityscapes_list/train.txt',
                            help="list of images in the target dataset.")
        parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")
        #end-------------dataset_setting-------------------------------------------------------------------

        #begin-------------training paras-------------------------------------------------------------------

        #learning rate&loss weight
        parser.add_argument("--learning-rate", type=float, default=5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument("--learning-rate-D", type=float, default=2e-4, help="initial learning rate for discriminator.")
        parser.add_argument("--decay_step", type=float, default=50000, help="decay step for segmentation network.")
        parser.add_argument("--lambda-adv-target", type=float, default=0.001, help="lambda_adv for adversarial training.")
        parser.add_argument("--num-steps", type=int, default=250000, help="Number of training steps.")
        parser.add_argument("--num-steps-stop", type=int, default=120000,
                            help="Number of training steps for early stopping.")
        #frequent setting
        parser.add_argument("--save-pred-every", type=int, default=2000, help="Save summaries and checkpoint every often.")
        parser.add_argument("--print-freq", type=int, default=100, help="print loss and time fequency.")

        #restore_from
        parser.add_argument("--init-weights", type=str, default='init/Source_only.pth', help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None, help="Where restore model parameters from.")

        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")
        parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--domain", type=str, default='T', help="The training domain of DPL.")

        return parser.parse_args()

    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    
        # save to the disk
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')    
        