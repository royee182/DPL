import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.print_networks()

    def forward(self, x, lbl):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        self.loss = self.bce_loss(x, Variable(torch.FloatTensor(x.data.size()).fill_(lbl)).cuda())

        return self.loss

    def adjust_learning_rate(self, args, optimizer, i):
        if args.model == 'DeepLab':
            lr = args.learning_rate_D * ((1 - float(i) / args.num_steps) ** (args.power))
            optimizer.param_groups[0]['lr'] = lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = lr * 10 
        else:
            optimizer.param_groups[0]['lr'] = args.learning_rate_D * (0.1**(int(i/50000)))
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.learning_rate_D * (0.1**(int(i/50000))) * 2  			

    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')

        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % ("discriminator", num_params / 1e6))
        print('-----------------------------------------------')