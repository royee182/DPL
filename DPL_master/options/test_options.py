import argparse
import os.path as osp
class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("--model", type=str, default='DeepLab', help="available options : DeepLab and VGG")
        parser.add_argument("--GPU", type=str, default='0', help="which GPU to use")

        parser.add_argument("--data-dir-target", type=str, default='../data/Cityscapes', help="Path to the directory containing the target dataset.")
        parser.add_argument("--data-list-target", type=str, default='./dataset/cityscapes_list/val.txt', help="list of images in the target dataset.")

        parser.add_argument("--source", type=str, default='gta5', help="source dataset : gta5 or synthia")
        parser.add_argument("--data-dir", type=str, default='../data/GTA5',
                            help="Path to the directory containing the source dataset.")
        parser.add_argument("--source-label-dir", type=str, default='../data/GTA5/labels',
                            help="Path to the directory containing the source dataset.")
        parser.add_argument("--data-list", type=str, default='./dataset/gta5_list/train.txt',
                            help="Path to the listing of images in the source dataset.")
        parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
        parser.add_argument("--set", type=str, default='val', help="choose test set.")
        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None, help="restore model parameters from")
        parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.")
        parser.add_argument("--save", type=str, default='results', help="Path to save result.")
        parser.add_argument('--gt-dir', type=str, default='../data/Cityscapes/gtFine/val', help='directory for CityScapes val gt images')
        parser.add_argument('--devkit-dir', type=str, default='./dataset/cityscapes_list', help='list directory of cityscapes')
        parser.add_argument("--log-dir", type=str, default='results', help="Path to save logs.")
        parser.add_argument("--thresh", type=float, default=0.9, help="The confidence threshold of pseudo label generation.")
        parser.add_argument("--threshlen", type=float, default=0.65, help=" The chosen ratio of pseudo label generation")
        parser.add_argument("--threshdelta", type=float, default=0.3, help="The correction rate for label correction.")
        parser.add_argument("--adap-learning", type=str, default='none', help="The dataset choosen for adptive learning,choosen from none|gta|city|all")
        parser.add_argument("--alpha", type=float, default=0.5,
                            help="The weighting coefficient of dual path prediction fusion.")
        parser.add_argument("--num-steps-stop", type=int, default=120000, help="Number of training steps for early stopping.")
        parser.add_argument("--data-dir-targetB", type=str, default='', help='Path to the directory containing the translated target dataset.')
        parser.add_argument("--init-weights_T", type=str, default='', help="Restore model T parameters from")
        parser.add_argument("--init-weights_S", type=str, default='', help="Restore model S parameters from")
        parser.add_argument("--domain", type=str, default='T', help="The testing domain of DPL.")

        return parser.parse_args()

   
