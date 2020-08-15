# -*- coding: utf-8 -*-

from datetime import datetime
import shutil
import socket
import time
import torch
from torch.optim import lr_scheduler

from dataloaders.CUHK_dataloader_test import KittiFolder
from metrics import AverageMeter, Result
import utils
import criteria
import os
import torch.nn as nn
import torch.nn.functional as F
from network import UNet
import numpy as np
from numpy import *
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import cv2
cmap = plt.cm.jet
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use single GPU

args = utils.parse_command()
print(args)
model = UNet(3,1)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()
# if setting gpu id, the using single GPU
print('Single GPU Mode.')


def create_loader(args):
    root_dir = ''
    test_set = KittiFolder(root_dir, mode='test', size=(256, 512))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return test_loader

def main():
    global args, best_result, output_directory

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    print("Let's use GPU ", torch.cuda.current_device())
    
    val_loader = create_loader(args)
    output_directory = utils.get_output_directory(args)
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load('/share2/public/fail_safe/kitti/DeepBlur/result/CUHK/run_2/model_best.pth.tar')
    # solve 'out of memory'
    model_sd = checkpoint['model'].state_dict()
    model.load_state_dict(model_sd)
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    test(val_loader,model)
    # clear memory
    del checkpoint
    # del model_dict
    torch.cuda.empty_cache()

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C
# validation
def mask_out(pred_mask,alpha):
    b_max =np.max(pred_mask*255.0)
    b_min =np.min(pred_mask*255.0)
    th1 = alpha * b_max + (1-alpha) * b_min
    mask = np.where(pred_mask*255.0>th1,255.0,0.0)
    return mask
def mse(gt,img):
    dif = gt - img
    error = np.sqrt(np.mean(dif**2))
    return error
def mae(gt,img):
    dif = gt - img
    error = np.mean(np.fabs(dif))
    return error
def test(val_loader, model):
    model.eval()  # switch to evaluate mode

    end = time.time()
    count=0
    count_0=0
    count_1=0
    count_2=0
    count_3=0
    count_b=0
    count_b0=0
    count_b1=0
    count_b2=0
    count_b3=0
    count_fb=0
    count_mb=0
    count_db=0
    temp=0
    acc = 0.0
    sigma = []
    mIoU = 0.0
    F_m = 0.0
    b=1.0
    gpu_time = 0.0
    mae_ref = 0.0
    mae_my = 0.0
    mse_ref = 0.0
    mse_my = 0.0
    temp_rec = 0.0
    temp_prec = 0.0
    fm_max = 0.0
    p_qiu = []
    for i, (input, target, label,mask,h1,w1) in enumerate(val_loader):
        print(i)
        input = torch.FloatTensor(input).cuda()
        # print(input.size())
        # exit(0)
        input = torch.squeeze(input,0)

        data_time = time.time() - end

        # compute output
        # end = time.time()
        with torch.no_grad():
            pred,pred_mask,c1,c2,c3,temp_gpu= model(input)
            # pred,c1,c2,c3= model(input)
        # torch.cuda.synchronize()
        # gpu_time += time.time() - end
        gpu_time += temp_gpu
        
        b,c,h,w = pred.size()

        temp0 = torch.zeros_like(pred_mask)
        temp1 = torch.ones_like(pred_mask)
        pred_mask = torch.where(pred_mask>0.5,temp1,temp0)
        
        temp0_2 = torch.zeros_like(c1)
        temp1_2 = torch.ones_like(c1)
        c1_2 = torch.where(c1>0.5,temp1_2,temp0_2)
        c2_2 = torch.where(c2>0.5,temp1_2,temp0_2)
        c3_2 = torch.where(c3>0.5,temp1_2,temp0_2)
        # measure accuracy and record loss
        c1 = c1.cpu().numpy() 
        c2 = c2.cpu().numpy() 
        c3 = c3.cpu().numpy() 
        c1_2 = c1_2.cpu().numpy()
        c2_2 = c2_2.cpu().numpy()
        c3_2 = c3_2.cpu().numpy()
        l = label.numpy()
   
        pred = torch.squeeze(pred)
        pred = pred.data.cpu().numpy()
        pred_mask = torch.squeeze(pred_mask)
        pred_mask = pred_mask.data.cpu().numpy()
        flag=0
        for k in range(l.shape[0]):
            if c2[k]>c3[k]:
                if l[k] == 1:
                    count_b += 1
                    count_b2+=1
                flag=2
            if c3[k]>c2[k]:
                if l[k] == 2:
                    count_b += 1
                    count_b3+=1
                flag=3
        
        ttf = target[0].split('/')[-1][:-4]
        if ttf == 'motion0027':
            p_qiu.append([flag,l[0],151])
        if ttf == 'motion0141':
            p_qiu.append([flag,l[0],141])
        if ttf == 'motion0153':
            p_qiu.append([flag,l[0],153])
        if ttf == 'out_of_focus0297':
            p_qiu.append([flag,l[0],297])
        if ttf == 'out_of_focus0579':
            p_qiu.append([flag,l[0],579])
        if ttf == 'out_of_focus0664':
            p_qiu.append([flag,l[0],664])
        count += 1
        # print(count_b*1.0/count)
        alpha = 0.4#0.455 0.4
        beta = 0.1#0.1

        temp_RGB = np.zeros((h,w,3))
        pred = np.where(pred<0.0,0.0,pred)
        pred = np.where(pred>1.0,1.0,pred)
        temp_RGB[:,:,0] = pred[0,:,:] * c1[0]
        temp_RGB[:,:,1] = pred[1,:,:] * c2[0]
        temp_RGB[:,:,2] = pred[2,:,:] * c3[0]
        RGB_out = temp_RGB * 255.0
        RGB_out = cv2.resize(RGB_out,(w1,h1),interpolation=cv2.INTER_LINEAR)
        # utils.save_image(RGB_out.astype(np.uint8),"/share2/public/fail_safe/kitti/DeepBlur/best_results_RGB_2/"+target[0].split('/')[-1][:-4]+".png")
        # bm = pred[1,:,:] * c2[0] + pred[2,:,:] * c3[0]
        bm = pred[1,:,:] * c2[0] + pred[2,:,:] * c3[0]  + beta * pred_mask #+ pred[0,:,:] * c1[0]
        bm = np.where(bm<0.0,0.0,bm)
        # b_m = bm*1.0/np.max(bm)
        # bm_out = b_m*255.0
        # bm_out = cv2.resize(bm_out,(w1,h1),interpolation=cv2.INTER_LINEAR)
        # utils.save_image(bm_out.astype(np.uint8),"/share2/public/fail_safe/kitti/DeepBlur/best_results/"+target[0].split('/')[-1][:-4]+".png")
        b_mask = mask_out(bm,alpha)
        # b_mask1 = mask_out(pred[1,:,:],alpha)
        # b_mask2 = mask_out(pred[2,:,:],alpha)
        # bm = b_mask1 + b_mask2
        # b_mask = np.where(bm>0.0,255.0,0.0)
        temp_file = '/share2/public/fail_safe/kitti/DeepBlur/temp_CUHK/PG18_result_binary/' + target[0].split('/')[-1][:-4] + '_ourbinary.png'

        ref_mask = cv2.imread(temp_file)[:,:,0].astype(np.float32)
        # print(ref_mask.shape)
        # exit(0)
        gt_mask = cv2.imread(target[0])[:,:,0].astype(np.float32)
        h2 = h1-(h1%16)
        w2 = w1-(w1%16)
        gt_mask2 = gt_mask[0:h2,0:w2]
        mse_ref += mse(gt_mask2*1.0 / 255.0,ref_mask*1.0 / 255.0)
        mae_ref += mae(gt_mask2*1.0 / 255.0,ref_mask*1.0 / 255.0)
        # end = time.time()
        b_mask = cv2.resize(b_mask,(w1,h1),interpolation=cv2.INTER_LINEAR)#.astype(np.uint8)
        # gpu_time += time.time() - end
        
        # b_mask = ref_mask
        # gt_mask = gt_mask2
        
        mse_my += mse(gt_mask*1.0 / 255.0,b_mask*1.0 / 255.0)
        mae_my += mae(gt_mask*1.0 / 255.0,b_mask*1.0 / 255.0)

        temp_acc = np.ones_like(gt_mask) - np.fabs(b_mask*1.0/255.0-gt_mask*1.0/255.0)
        temp_b1 = (gt_mask *1.0 / 255.0) * (b_mask * 1.0 / 255.0)
        temp_b2 = (gt_mask *1.0 / 255.0) + (b_mask * 1.0 / 255.0)
        temp_b2 = np.where(temp_b2>0.0,1.0,0.0)
        sum_b1 = np.sum(np.sum(temp_b1))
        sum_b_gt = np.sum(gt_mask[gt_mask>0]) *1.0 / 255.0
        # print(sum_b1,sum_b_gt)
        sum_b_b = np.sum(b_mask[b_mask>0])*1.0 / 255.0
        prec = sum_b1 * 1.0 / sum_b_b
        rec = sum_b1 * 1.0 / sum_b_gt
        
        if rec==0.0:
            F_m+=0.0
        else:
            temp_fm = (1+b)*prec*rec*1.0/(b*b*prec+rec)
            if temp_fm > fm_max:
                fm_max = temp_fm
            F_m += temp_fm
        print(prec,rec,F_m)
        temp_prec += prec
        temp_rec += rec
        sum_b2 = np.sum(np.sum(temp_b2))
        mIoU_t1 = sum_b1 * 1.0 / sum_b2
        gt_mask2 = 255.0*np.ones_like(gt_mask) - gt_mask
        b_mask2 = 255.0*np.ones_like(gt_mask) - b_mask
        temp_b1_2 = (gt_mask2 *1.0 / 255.0) * (b_mask2 * 1.0 / 255.0)
        temp_b2_2 = (gt_mask2 *1.0 / 255.0) + (b_mask2 * 1.0 / 255.0)
        temp_b2_2 = np.where(temp_b2_2>0.0,1.0,0.0)
        sum_b1_2 = np.sum(np.sum(temp_b1_2))
        sum_b2_2 = np.sum(np.sum(temp_b2_2))
        mIoU_t2 = sum_b1_2 * 1.0 / sum_b2_2
        t = np.mean(temp_acc)
        acc+=t
        sigma.append(t)
        mIoU += 0.5 * (mIoU_t1 + mIoU_t2)
        print(F_m/count)
        print(mIoU/count)
        print(acc/count)
        # print(gt_mask.shape,fb_mask.shape,mb_mask.shape,db_mask.shape)
        # exit(0)
        # vtitch = np.vstack([mb_mask,db_mask,gt_mask])
        # utils.save_image(vtitch, )
    sigma2 = np.var(sigma)
    print("mae: %f(my),%f(ref)"%(mae_my*1.0/count,mae_ref*1.0/count))
    print("mse: %f(my),%f(ref)"%(mse_my*1.0/count,mse_ref*1.0/count))
    print("F-measure: %f,max: %f" % (F_m*1.0/count, fm_max))    
    print("mIoU: %f" % (mIoU*1.0/count))  
    print("sigma2: %f" % (sigma2))  
    print("acc: %f" % (acc*1.0/count)) 
    print("mean prec: %f, rec: %f" % (temp_prec*1.0/count,temp_rec*1.0/count))   
    print("acc_c: %f" % (count_b*1.0/count))
    print("time: %f" % gpu_time)
    print(torch.cuda.get_device_name(0))
    print(p_qiu)
if __name__ == '__main__':
    main()
