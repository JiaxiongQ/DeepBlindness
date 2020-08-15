# -*- coding: utf-8 -*-

from datetime import datetime
import shutil
import socket
import time
import torch
from torch.optim import lr_scheduler

# from dataloaders.kitti_dataloader import KittiFolder
from dataloaders.CUHK_dataloader import KittiFolder
from metrics import AverageMeter, Result
import utils
import criteria
import os
import torch.nn as nn
import torch.nn.functional as F
from network import UNet
import numpy as np
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use single GPU

args = utils.parse_command()
print(args)

# if setting gpu id, the using single GPU
if args.gpu:
    print('Single GPU Mode.')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

best_result = Result()
best_result.set_to_worst()


def create_loader(args):
    root_dir = ''

    train_set = KittiFolder(root_dir, mode='train')
    test_set = KittiFolder(root_dir, mode='test')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader
def adjust_learning_rate(optimizer, epoch):
    if epoch <= 100:
        learning_rate = 0.001
    if epoch>100 and epoch <= 200:
        learning_rate = 0.0001
    if epoch>200 and epoch <=400:
        learning_rate = 0.00001
    if epoch>400 and epoch <= 800:
        learning_rate = 0.000001
    if epoch>800:
        learning_rate = 0.0000001

    print(learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def main():
    global args, best_result, output_directory

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print("Let's use GPU ", torch.cuda.current_device())

    train_loader, val_loader = create_loader(args)

    if args.resume:
        assert os.path.isfile(args.resume), \
            "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = 0
        # start_epoch = checkpoint['epoch'] + 1
        # best_result = checkpoint['best_result']
        # optimizer = checkpoint['optimizer']
        
        # solve 'out of memory'
        model = checkpoint['model']
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        # clear memory
        del checkpoint
        # del model_dict
        torch.cuda.empty_cache()
    else:
        print("=> creating Model")
        # input_shape = [args.batch_size,3,256,512]
        model = UNet(3,1)
        print("=> model created.")
        start_epoch = 0

        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # You can use DataParallel() whether you use Multi-GPUs or not
        model = nn.DataParallel(model).cuda()

    # when training, use reduceLROnPlateau to reduce learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience)

    # loss function
    criterion = criteria.myL1Loss()
    # criterion = nn.SmoothL1Loss()
    # create directory path
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')
    config_txt = os.path.join(output_directory, 'config.txt')

    # write training parameters to config file
    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str)

    for epoch in range(start_epoch, args.epochs):

        # remember change of the learning rate
        old_lr = 0.0
        # adjust_learning_rate(optimizer,epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])
        print("lr: %f" % old_lr)

        train(train_loader, model, criterion, optimizer, epoch)  # train for one epoch
        result,img_merge = validate(val_loader, model, epoch)  # evaluate on validation set

        # remember best mae and save checkpoint
        is_best = result.mae < best_result.mae
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}, mae={:.3f}, "
                    "t_gpu={:.4f}".
                        format(epoch, result.mae,
                               result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        # save checkpoint for each epoch
        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)

        # when mae doesn't fall, reduce learning rate
        scheduler.step(result.mae)
def cross_entropy_loss2d(inputs, targets, cuda=True, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    inputs = torch.unsqueeze(inputs,0)
    inputs = torch.unsqueeze(inputs,0)
    # targets = torch.unsqueeze(targets,0)
    targets = torch.unsqueeze(targets,0)
    # print(inputs.size(),targets.size())
    # exit(0)
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, :,:,:] = np.where(t==1,neg * 1. / valid, pos * balance / valid)

    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    inputs = F.sigmoid(inputs)
    loss = nn.BCELoss(weights, size_average=False)(inputs, targets)
    return loss
def cross_entropy_loss2d_2(inputs, targets, cuda=True, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    inputs = torch.unsqueeze(inputs,0)
    targets = torch.unsqueeze(targets,0)
    # print(inputs.size(),targets.size())
    # exit(0)
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, :,:,:] = np.where(t==1,neg * 1. / valid, pos * balance / valid)

    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    inputs = F.sigmoid(inputs)
    loss = nn.BCELoss(weights, size_average=False)(inputs, targets)
    return loss
def cross_entropy_loss1d(inputs, targets):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    loss = nn.BCELoss(size_average=True)(inputs, targets)
    return loss
# train
def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train()  # switch to train mode
    end = time.time()

    for i, (input, target,label,mask) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        label = label.cuda()
        mask = mask.cuda()
        # print('input size  = ', input.size())
        # print('target size = ', target.size())
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()

        pred,pred_mask,c1,c2,c3 = model(input)
        # pred,c1,c2,c3 = model(input)
        target = torch.squeeze(target,1)

        loss = 0.0
        lossM = 0.0
        lossC = 0.0
        loss_all = 0.0
        count = 0
        countM = 0
        countC = 0
        
        criterion2=criteria.MaskedL1Loss()
        #criterionM=criteria.FocalLoss()

        for j in range(len(label)):
            if label[j] == -1:
                loss += (criterion(pred[count,0,:,:], target[count,:,:]) + criterion(pred[count,1,:,:], target[count,:,:]) + criterion(pred[count,2,:,:], target[count,:,:]))*1.0/3
                lossC += cross_entropy_loss1d(c1[count,0], torch.zeros_like(label[j]).float()) + cross_entropy_loss1d(c2[count,0], torch.zeros_like(label[j]).float()) + cross_entropy_loss1d(c3[count,0], torch.zeros_like(label[j]).float())
            elif label[j] == 1:
                loss += criterion(pred[count, 1, :, :], target[count,:,:])#2 +0.5*(criterion(pred[count, 0, :, :], torch.zeros_like(pred[count, 1, :, :]))+criterion(pred[count, 2, :, :], torch.zeros_like(pred[count, 1, :, :])))
                lossM += cross_entropy_loss2d_2(pred_mask[count, :, :, :], mask[count,:,:])
                lossC += cross_entropy_loss1d(c1[count,0], torch.zeros_like(label[j]).float()) + cross_entropy_loss1d(c2[count,0], torch.ones_like(label[j]).float()) + cross_entropy_loss1d(c3[count,0], torch.zeros_like(label[j]).float())
                countM += 1
            elif label[j] == 0:
                loss += criterion(pred[count, 0, :, :], target[count,:,:])#+0.5*(criterion(pred[count, 1, :, :], torch.zeros_like(pred[count, 0, :, :]))+criterion(pred[count, 2, :, :], torch.zeros_like(pred[count, 0, :, :])))
                lossM += cross_entropy_loss2d_2(pred_mask[count, :, :, :], mask[count,:,:])
                lossC += cross_entropy_loss1d(c1[count,0], torch.ones_like(label[j]).float()) + cross_entropy_loss1d(c2[count,0], torch.zeros_like(label[j]).float()) + cross_entropy_loss1d(c3[count,0], torch.zeros_like(label[j]).float())
                countM += 1
            else:
                loss += criterion(pred[count, 2, :, :], target[count,:,:])#+0.5*(criterion(pred[count, 0, :, :], torch.zeros_like(pred[count, 2, :, :]))+criterion(pred[count, 1, :, :], torch.zeros_like(pred[count, 2, :, :])))
                lossM += cross_entropy_loss2d_2(pred_mask[count, :, :, :], mask[count,:,:])
                lossC += cross_entropy_loss1d(c1[count,0], torch.zeros_like(label[j]).float()) + cross_entropy_loss1d(c2[count,0], torch.zeros_like(label[j]).float()) + cross_entropy_loss1d(c3[count,0], torch.ones_like(label[j]).float())
                countM += 1
            count += 1
        # for j in range(len(label)):
        #     if label[j] == 1:
        #         # pred[count, 1, :, :] = pred[count, 1, :, :] + pred_mask[count, :, :, :]
        #         loss += cross_entropy_loss2d(pred[count, 1, :, :], mask[count,:,:])#2 +0.5*(criterion(pred[count, 0, :, :], torch.zeros_like(pred[count, 1, :, :]))+criterion(pred[count, 2, :, :], torch.zeros_like(pred[count, 1, :, :])))
        #         lossM += cross_entropy_loss2d_2(pred_mask[count, :, :, :], mask[count,:,:])
        #         lossC += cross_entropy_loss1d(c1[count,0], torch.zeros_like(label[j]).float()) + cross_entropy_loss1d(c2[count,0], torch.ones_like(label[j]).float()) + cross_entropy_loss1d(c3[count,0], torch.zeros_like(label[j]).float())
        #         countM += 1
        #     else:
        #         # pred[count, 2, :, :] = pred[count, 2, :, :] + pred_mask[count, :, :, :]
        #         loss += cross_entropy_loss2d(pred[count, 2, :, :], mask[count,:,:])#+0.5*(criterion(pred[count, 0, :, :], torch.zeros_like(pred[count, 2, :, :]))+criterion(pred[count, 1, :, :], torch.zeros_like(pred[count, 2, :, :])))
        #         lossM += cross_entropy_loss2d_2(pred_mask[count, :, :, :], mask[count,:,:])
        #         lossC += cross_entropy_loss1d(c1[count,0], torch.zeros_like(label[j]).float()) + cross_entropy_loss1d(c2[count,0], torch.zeros_like(label[j]).float()) + cross_entropy_loss1d(c3[count,0], torch.ones_like(label[j]).float())
        #         countM += 1
        #     count += 1

        lossm = 0.00001*lossM / countM#0.000005 0.00001
        lossC = 0.01*lossC / count#0.005 0.01
        # lossm = lossC
        loss =loss * 1.0/ count + lossm + lossC
        # loss =loss * 1.0/ count + lossC

        # print(pred.size(),target.size())
        # exit(0)
        #loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred, target,label)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f} '
                  'LossM={LossM:.5f} '
                  'LossC={LossC:.5f} '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  .format(
                epoch, i + 1, len(train_loader), data_time=data_time,
                gpu_time=gpu_time, Loss=loss.item(), LossM = lossm.item(), LossC = lossC.item(), result=result, average=average_meter.average()))


# validation
def validate(val_loader, model, epoch):
    average_meter = AverageMeter()

    model.eval()  # switch to evaluate mode

    end = time.time()

    skip = len(val_loader) // 8  # save images every skip iters
    count_b = 0
    count = 0
    x = random.randint(0, len(val_loader))
    for i, (input, target, label,mask) in enumerate(val_loader):

        input, target = input.cuda(), target.cuda()
        input = torch.squeeze(input,0)
        mask = mask.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred,pred_mask,c1,c2,c3 = model(input)
            # pred,c1,c2,c3 = model(input)
        b,c,h,w = pred.size()

        # temp0 = torch.zeros_like(pred_mask)
        # temp1 = torch.ones_like(pred_mask)
        # pred_mask2 = torch.where(pred_mask>0.5,temp1,temp0)
        #pred[:,1,:,:] = pred[:,1,:,:] * pred_mask2
        
        temp0_2 = torch.zeros_like(c1)
        temp1_2 = torch.ones_like(c1)
        c1_2 = torch.where(c1>0.5,temp1_2,temp0_2)
        c2_2 = torch.where(c2>0.5,temp1_2,temp0_2)
        c3_2 = torch.where(c3>0.5,temp1_2,temp0_2)
        
        torch.cuda.synchronize()

        gpu_time = time.time() - end

        target = torch.squeeze(target, 1)
        # measure accuracy and record loss
        c1_2 = c1_2.cpu().numpy()
        c2_2 = c2_2.cpu().numpy()
        c3_2 = c3_2.cpu().numpy()
        l = label.numpy()
        for k in range(l.shape[0]):
            if c1_2[k]==0 and c2_2[k]==0 and c3_2[k]==0 and l[k] == -1:
                count_b += 1
            if c1_2[k]==1 and c2_2[k]==0 and c3_2[k]==0 and l[k] == 0:
                count_b += 1
            if c1_2[k]==0 and c2_2[k]==1 and c3_2[k]==0 and l[k] == 1:
                count_b += 1
            if c1_2[k]==0 and c2_2[k]==0 and c3_2[k]==1 and l[k] == 2:
                count_b += 1
        count += l.shape[0]
        result = Result()
        result.evaluate(pred, target,label)

        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        rgb = input
        # print(rgb.size(),target.size(),pred.size())
        # exit(0)
        
        # if i == x:
        #     img_merge = utils.merge_into_row(rgb, target, pred, pred_mask2,label)
        #     filename = output_directory + '/comparison_' + str(epoch) + '.png'
        #     utils.save_image(img_merge, filename)
        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred, target,label)# (rgb, target, pred, pred_mask2,label)
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred, target,label)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)
            
        if (i + 1) % args.print_freq == 0:
            print("acc: %f" % (count_b*1.0/count))
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'MAE={result.mae:.2f}({average.mae:.2f}) '.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))
    avg = average_meter.average()
    
    print("epoch: %d, acc: %f" % (epoch,count_b*1.0/count))
    print('\n*\n'
          'MAE={average.mae:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))
    
    return avg,img_merge


if __name__ == '__main__':
    main()
