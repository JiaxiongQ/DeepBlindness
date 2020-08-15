# -*- coding: utf-8 -*-

from datetime import datetime
import shutil
import socket
import time
import torch
from torch.optim import lr_scheduler

from dataloaders.kitti_dataloader_test import KittiFolder
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
    checkpoint = torch.load('/share2/public/fail_safe/kitti/DeepBlur/result/kitti/run_7/model_best.pth.tar')

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

def pca(X):
    num_data,dim = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim>num_data:
        M = dot(X,X.T)
        e,EV = linalg.eigh(M)
        tmp = dot(X.T,EV).T
        V = tmp[::-1]
        S = sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        U,S,V = linalg.svd(X)
        V = V[:num_data]
    return V,S,mean_X

def kmeans(X,n):
    estimator=KMeans(n_clusters=n)
    res=estimator.fit_predict(X)
    label_pred=estimator.labels_
    centroids=estimator.cluster_centers_
    inertia=estimator.inertia_
    return label_pred
def mse(gt,img):
    dif = gt - img
    error = np.mean(dif**2)
    return error
def mae(gt,img):
    dif = gt - img
    error = np.mean(np.fabs(dif))
    return error
# validation
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
    mse_my = 0
    mae_my = 0
    temp=0
    gpu_time = 0.0
    count_t =0
    for i, (input, target, input_f, label,mask,h1,w1) in enumerate(val_loader):
        print(i)
        input = torch.FloatTensor(input).cuda()
        # print(input.size())
        # exit(0)
        input = torch.squeeze(input,0)

        data_time = time.time() - end

        # compute output
        # end = time.time()
        with torch.no_grad():
            pred,pred_mask,c1,c2,c3,temp_t = model(input)
            # pred,c1,c2,c3 = model(input)
        # torch.cuda.synchronize()
        gpu_time += temp_t
        # gpu_time += time.time() - end
        # print("time: %f" % gpu_time)
        b,c,h,w = pred.size()
# 
        temp0 = torch.zeros_like(pred_mask)
        temp1 = torch.ones_like(pred_mask)
        pred_mask2 = torch.where(pred_mask>0.5,temp1,temp0)
        pred[:,1,:,:] = pred[:,1,:,:] * pred_mask2
        
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
        # if l[0]>=-1:
        #     gpu_time += temp_t
        #     count_t += 1

        flag=-1
        if c1_2[0]==0 and c2_2[0]==0 and c3_2[0]==0 and l[0] == -1:
            count_b0+=1
            count_b += 1
            flag=0
        else:
            if c1[0]>c2[0] and c1[0] > c3[0]:
                flag=1
                if l[0] == 0:
                    count_b += 1
                    count_b1+=1
                
            if c2[0]>c1[0] and c2[0] > c3[0]:
                flag=2
                if l[0] == 1:
                    count_b += 1
                    count_b2+=1
            if c3[0]>c1[0] and c3[0] > c2[0]:
                flag=3
                if l[0] == 2:
                    count_b += 1
                    count_b3+=1
                
        
        count += 1
        print(count_b*1.0/count)
        
        if l[0]==-1:
            count_0+=1
        if l[0]==0:
            count_1+=1
        if l[0]==1:
            count_2+=1
        if l[0]==2:
            count_3+=1
        temp_c1 = 0
        temp_c2 = 0
        temp_c3 = 0
        temp_p = 0
        temp_gt = 0
        temp_i=0
        if (count_b*1.0/count) < 1 and temp==0:
           temp_c1 = c1[0]
           temp_c2 = c2[0]
           temp_c3 = c3[0]
           temp_p = flag
           temp_gt = l[0]
           temp_i=i
           temp=1
        #print(gpu_time)
        pred_mask2 = torch.squeeze(pred_mask2)
        pred_mask2 = pred_mask2.data.cpu().numpy()
        pred = torch.squeeze(pred)
        pred = pred.data.cpu().numpy()
        # h,w = target.shape
        # if w>1100:
        #     top_pad = 384-mask.shape[0]
        #     left_pad = 1248-mask.shape[1]
        # else:
        #     top_pad = 384-mask.shape[0]
        #     left_pad = 624-mask.shape[1]
        gt_img = cv2.imread(input_f[0])
        if l[0] == -1:
            gt_bm2 = np.zeros((h1,w1))
            f_out1 = '/share2/public/fail_safe/kitti/DeepBlur/source_images/images/no/' + str(i) + '.png'
            f_out2 = '/share2/public/fail_safe/kitti/DeepBlur/source_images/gt/no/' + str(i) + '.png'
        else:
            gt_bm2 = cv2.imread(target[0])[:,:,0]
            if l[0] == 0:
                gt_bm2 = 255.0 - gt_bm2
                f_out1 = '/share2/public/fail_safe/kitti/DeepBlur/source_images/images/fb/' + str(i) + '.png'
                f_out2 = '/share2/public/fail_safe/kitti/DeepBlur/source_images/gt/fb/' + str(i) + '.png'
            if l[0] == 1:
                f_out1 = '/share2/public/fail_safe/kitti/DeepBlur/source_images/images/mb/' + str(i) + '.png'
                f_out2 = '/share2/public/fail_safe/kitti/DeepBlur/source_images/gt/mb/' + str(i) + '.png'
            if l[0] == 2:
                f_out1 = '/share2/public/fail_safe/kitti/DeepBlur/source_images/images/db/' + str(i) + '.png'
                f_out2 = '/share2/public/fail_safe/kitti/DeepBlur/source_images/gt/db/' + str(i) + '.png'  
        # cv2.imwrite(f_out1,gt_img)
        gt_bm2 = gt_bm2 * 1.0/np.max(gt_bm2)
        cv2.imwrite(f_out2,(gt_bm2*255.0).astype(np.uint8))
        if l[0]!=-1:
            gt_bm = cv2.imread(target[0])[:,:,0].astype(np.float32)
            if l[0] == 0:
               gt_bm = np.ones_like(gt_bm)*1.0 - gt_bm * 1.0 / 255.0
            if l[0] == 1:
               gt_bm = gt_bm * 1.0 / 255.0
            if l[0] == 2:
               gt_bm = gt_bm * 1.0 / 200.0
        else:
            gt_bm = np.zeros((h1,w1)).astype(np.float32)

        if flag<=0:
            my_bm = np.zeros((h1,w1)).astype(np.float32)
        else:
            if flag == 1:
                my_bm = pred[0,:,:]
            if flag == 2:
                my_bm = pred[1,:,:]
            if flag == 3:
                my_bm = pred[2,:,:]

        my_bm = cv2.resize(my_bm,(w1,h1),interpolation=cv2.INTER_LINEAR).astype(np.float32)
        my_bm = np.where(my_bm<0.0,0.0,my_bm)
        my_bm = np.where(my_bm>1.0,1.0,my_bm)
        if i<394:
            t_i=i
        else:
            t_i = i + 1
        # my_bm = 
        # print(gt_bm.shape,my_bm.shape)
        # exit(0)
        if flag>=1:
            if flag==1:
                fn_fb = '/share2/public/fail_safe/kitti/DeepBlur/result_ICRA/ours/fb/' + str(t_i) + '.png'
                out_my = my_bm * 255.0
            if flag>=2:
                fn_fb = '/share2/public/fail_safe/kitti/DeepBlur/result_ICRA/ours/mb_2/' + str(t_i) + '.png'
                if flag==2:
                    out_my = my_bm * 255.0
                if flag==3:
                    out_my = my_bm * 200.0
                out_my = out_my * 1.0/ np.max(out_my)
                out_my = out_my * 255.0
                # utils.save_image(out_my.astype(np.uint8),fn_fb)

        if l[0] < -1:
            my_bm = cv2.imread('/share2/public/fail_safe/kitti/DeepBlur/result_ICRA/db+mb_result/'+str(t_i)+'_FinalMap.png')[:,:,0].astype(np.float32)
            my_bm = cv2.resize(my_bm,(w1,h1),interpolation=cv2.INTER_LINEAR)
            # my_bm = np.ones_like(my_bm)*1.0 - my_bm * 1.0 / 255.0
            my_bm = my_bm * 1.0 / 255.0
            # bm_sum = np.sum(np.sum(my_bm))
            # if bm_sum!=0:
            mse_my += mse(gt_bm,my_bm)
            mae_my += mae(gt_bm,my_bm)
            count_t += 1
        RGB = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
        # pred_mask2 = torch.squeeze(pred_mask2)
        # tempM = pred_mask2.cpu().numpy()
        
        pred_fb = colored_depthmap(pred[0,:,:],np.min(pred[0,:,:]),np.max(pred[0,:,:]))
        pred_mb = colored_depthmap(pred[1,:,:],np.min(pred[1,:,:]),np.max(pred[1,:,:]))
        pred_db = colored_depthmap(pred[2,:,:],np.min(pred[2,:,:]),np.max(pred[2,:,:]))
        
        RGB = cv2.resize(RGB,(w1,h1),interpolation=cv2.INTER_LINEAR)
        
        
        # if flag>0:
        #     temp_mb = cv2.resize(pred[1,:,:],(w1,h1),interpolation=cv2.INTER_LINEAR).astype(np.float32)
        #     temp_mb = np.where(temp_mb<0.0,0.0,temp_mb)
        #     temp_mb = np.where(temp_mb>1.0,1.0,temp_mb)
        #     temp_db = cv2.resize(pred[2,:,:],(w1,h1),interpolation=cv2.INTER_LINEAR).astype(np.float32)
        #     temp_db = np.where(temp_db<0.0,0.0,temp_db)
        #     temp_db = np.where(temp_db>1.0,1.0,temp_db)
        #     temp_mask = cv2.resize(pred_mask2,(w1,h1),interpolation=cv2.INTER_LINEAR).astype(np.float32)
        #     # fn_input = '/share2/public/fail_safe/kitti/DeepBlur/result_ICRA/image/' + str(i) + '.png'
        #     # utils.save_image(RGB.astype(np.uint8), fn_input)
        #     fn_fb = '/share2/public/fail_safe/kitti/DeepBlur/result_ICRA/' + str(i) + 'fb.png'
        #     utils.save_image((my_bm*255.0).astype(np.uint8),fn_fb)
        #     fn_mb = '/share2/public/fail_safe/kitti/DeepBlur/result_ICRA/' + str(i) + 'mb.png'
        #     utils.save_image((temp_mb*255.0).astype(np.uint8), fn_mb)
        #     fn_db = '/share2/public/fail_safe/kitti/DeepBlur/result_ICRA/' + str(i) + 'db.png'
        #     utils.save_image((temp_db*200.0).astype(np.uint8), fn_db)
            # fn_mask = '/share2/public/fail_safe/kitti/DeepBlur/result_ICRA/' + str(i) + 'mask.png'
            # utils.save_image((temp_mask*255.0).astype(np.uint8), fn_mask)
        # pred_fb = cv2.resize(pred_fb,(w1,h1),interpolation=cv2.INTER_LINEAR)
        # pred_mb = cv2.resize(pred_mb,(w1,h1),interpolation=cv2.INTER_LINEAR)
        # pred_db = cv2.resize(pred_db,(w1,h1),interpolation=cv2.INTER_LINEAR)

        # filename1 = '/share2/public/fail_safe/kitti/DeepBlur/result/result_0801/bm/fb' + '/fb_' + str(i) + '.png'
        # utils.save_image(pred_fb, filename1)
        # filename2 = '/share2/public/fail_safe/kitti/DeepBlur/result/result_0801/bm/mb' + '/mb_' + str(i) + '.png'
        # utils.save_image(pred_mb, filename2)
        # filename3 = '/share2/public/fail_safe/kitti/DeepBlur/result/result_0801/bm/db' + '/db_' + str(i) + '.png'
        # utils.save_image(pred_db, filename3)
        if flag==0:
            filename4 = '/share2/public/fail_safe/kitti/DeepBlur/result/result_0805/image/no/' + '/no_' + str(i) + '.png'
        if flag==1:
            filename4 = '/share2/public/fail_safe/kitti/DeepBlur/result/result_0805/image/fb/' + '/fb_' + str(i) + '.png'
        if flag==2:
            filename4 = '/share2/public/fail_safe/kitti/DeepBlur/result/result_0805/image/mb/' + '/mb_' + str(i) + '.png'
        if flag==3:
            filename4 = '/share2/public/fail_safe/kitti/DeepBlur/result/result_0805/image/db/' + '/db_' + str(i) + '.png'    
        # utils.save_image(RGB, filename4)
    print("error: %d (%d / %d) %f,%f,%f" % (temp_i,temp_p,temp_gt,temp_c1,temp_c2,temp_c3))
    print("acc: %f" % (count_b*1.0/count))
    print("precision: none:%f, fb:%f, mb:%f, db:%f" % (count_b0*1.0/count_0,count_b1*1.0/count_1,count_b2*1.0/count_2,count_b3*1.0/count_3))
    print("mae: %f, mse: %f" % (mae_my*1.0/count,mse_my*1.0/count))
    print("time: %f" % (gpu_time *1.0 / count))
def test_0(val_loader, model):
    model.eval()  # switch to evaluate mode

    end = time.time()

    for i, (input, target, label,mask) in enumerate(val_loader):
         
        input = torch.FloatTensor(input).cuda()
        input = torch.squeeze(input,0)

        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        gpu_time = time.time() - end
        print(gpu_time)

        target = torch.squeeze(target)
        target = target.data.cpu().numpy()
        mask = torch.squeeze(mask)
        mask = mask.data.cpu().numpy()
        pred = torch.squeeze(pred)
        pred = pred.data.cpu().numpy()
        h,w = target.shape
        if w>1100:
            top_pad = 384-mask.shape[0]
            left_pad = 1248-mask.shape[1]
        else:
            top_pad = 384-mask.shape[0]
            left_pad = 624-mask.shape[1]
        print(top_pad,left_pad)
        # print(target.shape,mask.shape)
        # exit(0)
        RGB = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))

        pred_fb = colored_depthmap(pred[0,top_pad:,:-left_pad],np.min(pred[0,:,:]),np.max(pred[0,:,:]))
        pred_mb = colored_depthmap(pred[1,top_pad:,:-left_pad],np.min(pred[1,:,:]),np.max(pred[1,:,:]))
        pred_db = colored_depthmap(pred[2,top_pad:,:-left_pad],np.min(pred[2,:,:]),np.max(pred[2,:,:]))

        filename1 = '/share2/public/fail_safe/kitti/DeepBlur/result/fb' + '/fb_0_' + str(i) + '.png'
        utils.save_image(pred_fb, filename1)
        filename2 = '/share2/public/fail_safe/kitti/DeepBlur/result/mb' + '/mb_0_' + str(i) + '.png'
        utils.save_image(pred_mb, filename2)
        filename3 = '/share2/public/fail_safe/kitti/DeepBlur/result/db' + '/db_0_' + str(i) + '.png'
        utils.save_image(pred_db, filename3)
        filename4 = '/share2/public/fail_safe/kitti/DeepBlur/result/img' + '/img_' + str(i) + '.png'
        utils.save_image(RGB, filename4)
if __name__ == '__main__':
    main()
