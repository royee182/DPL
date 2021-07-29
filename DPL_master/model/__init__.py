from model.deeplab import Deeplab
from model.fcn8s import VGG16_FCN8s
from model.discriminator import FCDiscriminator
import torch.optim as optim
import torch
import torch.nn as nn
from sync_batchnorm import convert_model
def CreateRestoreModel(args):
    if args.model == 'DeepLab':

        model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights,
                        restore_from=args.restore_from, phase=args.set)

        model.eval()
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        return model

def CreateModel(args):
    if args.model == 'DeepLab':
        if args.set == 'train':

            model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights,
                                restore_from=args.restore_from, phase=args.set)

            optimizer = optim.SGD(model.optim_parameters(args),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.zero_grad()
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            model = convert_model(model)
            return model, optimizer

        else:
            model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights,
                            restore_from=args.restore_from, phase=args.set)
            return model

    if args.model == 'VGG':
        model = VGG16_FCN8s(num_classes=19, init_weights=args.init_weights, restore_from=args.restore_from)
        if args.set == 'train':
            optimizer = optim.Adam(
                [
                    {'params': model.get_parameters(bias=False)},
                    {'params': model.get_parameters(bias=True),
                     'lr': args.learning_rate * 2}
                ],
                lr=args.learning_rate,
                betas=(0.9, 0.99))
            optimizer.zero_grad()
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            model = convert_model(model)
            return model, optimizer
        else:
            return model


def CreateDiscriminator(args):
    discriminator = FCDiscriminator(num_classes=args.num_classes)
    optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer.zero_grad()
    if args.restore_from is not None:
        discriminator.load_state_dict(torch.load(args.restore_from + '_D.pth'))

    discriminator = discriminator.cuda()
    discriminator = nn.DataParallel(discriminator, device_ids=range(torch.cuda.device_count()))
    discriminator = convert_model(discriminator)

    return discriminator, optimizer


def CreateSSLModel(args):
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights, restore_from=args.restore_from,
                        phase=args.set)
    elif args.model == 'VGG':
        model = VGG16_FCN8s(num_classes=19, init_weights=args.init_weights, restore_from=args.restore_from)
    else:
        raise ValueError('The model mush be either deeplab-101 or vgg16-fcn')
    return model


