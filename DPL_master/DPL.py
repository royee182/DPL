import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from options.train_options import TrainOptions
import os
import numpy as np
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from model import CreateModel
from model import CreateDiscriminator
from utils.timer import Timer
import tensorboardX
from evaluation import val
import copy
def main():
    
    opt = TrainOptions()
    args = opt.initialize()
    
    _t = {'iter time' : Timer()}
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)   
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)
    sourceloader=CreateSrcDataLoader(args)
    if args.domain=='T':
        targetloader =  CreateTrgDataLoader(args)
    else:
        targetloader = CreateTrgDataLoader(args,translated=True)
    targetloader_iter, sourceloader_iter = iter(targetloader), iter(sourceloader)
    
    model, optimizer = CreateModel(args)
    model_D, optimizer_D = CreateDiscriminator(args)

    print("model.device_ids")
    print(model.device_ids)

    start_iter = 0
    train_writer = tensorboardX.SummaryWriter(os.path.join(args.snapshot_dir, "logs", model_name))

    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()
    model_D.train()
    model_D.cuda()
    loss = ['loss_seg_src', 'loss_seg_trg', 'loss_D_trg_fake', 'loss_D_src_real', 'loss_D_trg_real']
    _t['iter time'].tic()
    bestmIou = 0
    bestIter = 0
    for i in range(start_iter, args.num_steps):

        model.module.adjust_learning_rate(args, optimizer, i)
        model_D.module.adjust_learning_rate(args, optimizer_D, i)
        
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        for param in model_D.parameters():
            param.requires_grad = False 
            
        src_img, src_lbl, _, _ = sourceloader_iter.next()
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()
        src_seg_score,loss_seg_src = model(src_img, lbl=src_lbl)
        loss_seg_src=loss_seg_src.mean()
        loss_seg_src.backward()
        
        if args.data_label_folder_target is not None:
            trg_img, trg_lbl, _, _ = targetloader_iter.next()
            trg_img, trg_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda()
            trg_seg_score,loss_seg_trg = model(trg_img, lbl=trg_lbl)
            loss_seg_trg=loss_seg_trg.mean()

        else:
            trg_img, name, _ = targetloader_iter.next()
            trg_img = Variable(trg_img).cuda()
            trg_seg_score = model(trg_img)
            loss_seg_trg = 0

        loss_D_trg_fake = model_D(F.softmax(trg_seg_score,dim=1), 0).mean()
        loss_trg = args.lambda_adv_target * loss_D_trg_fake + loss_seg_trg
        loss_trg.backward()
        
        for param in model_D.parameters():
            param.requires_grad = True
        
        src_seg_score, trg_seg_score = src_seg_score.detach(), trg_seg_score.detach()
        
        loss_D_src_real = model_D(F.softmax(src_seg_score,dim=1), 0).mean()
        loss_D_trg_real = model_D(F.softmax(trg_seg_score,dim=1), 1).mean()
        loss_D=(loss_D_src_real+loss_D_trg_real)/2
        loss_D.backward()

        optimizer.step()
        optimizer_D.step()
        
        
        for m in loss:
            train_writer.add_scalar(m, eval(m), i+1)
            
        if (i+1) % args.save_pred_every == 0:
            print('taking snapshot ...')
            model_name = os.path.join(args.snapshot_dir, str(i+1) + '.pth')
            torch.save(model.state_dict(), model_name)
            args_cp=copy.deepcopy(args)
            args_cp.init_weights=model_name
            miou_resu=val(args_cp)
            if miou_resu>bestmIou:
                bestmIou=miou_resu
                bestIter=i+1
                print("current bestmIou: %.2f, best Iter %d"%(bestmIou,bestIter))
                model_name = os.path.join(args.snapshot_dir, 'best_model.pth')
                torch.save(model.state_dict(), model_name)

        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            if loss_seg_trg!=0:
                print('[it %d][src seg loss %.4f][org seg loss %.4f][D loss %4f][lr %.4f][%.2fs]' % \
                        (i + 1, loss_seg_src.data,loss_seg_trg.data,(loss_D_src_real.data+loss_D_trg_fake.data+loss_D_trg_real.data)/3, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff))
            else:
                print('[it %d][src seg loss %.4f][D loss %4f][lr %.4f][%.2fs]' % \
                        (i + 1, loss_seg_src.data,(loss_D_src_real.data+loss_D_trg_fake.data+loss_D_trg_real.data)/3, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff))
            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()
            
if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    main()