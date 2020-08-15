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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use single GPU

args = utils.parse_command()
print(args)
model = UNet(3,1)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()
# if setting gpu id, the using single GPU
print('Single GPU Mode.')

def main():
    global args, best_result, output_directory

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    print("Let's use GPU ", torch.cuda.current_device())

    output_directory = utils.get_output_directory(args)
    print("=> loading checkpoint '{}'".format(args.resume))
    # checkpoint = torch.load('/share2/public/fail_safe/kitti/DeepBlur/result/kitti/run_4/model_best.pth.tar')
    checkpoint = torch.load('/share2/public/fail_safe/kitti/DeepBlur/result/kitti/run_7/model_best.pth.tar')
    # solve 'out of memory'
    model_sd = checkpoint['model'].state_dict()
    model.load_state_dict(model_sd)
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    test(model)
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

from dataloaders import transforms as my_transforms
to_tensor = my_transforms.ToTensor()
def mse(gt,img):
    dif = gt - img
    error = np.mean(dif**2)
    return error
def mae(gt,img):
    dif = gt - img
    error = np.mean(np.fabs(dif))
    return error
# validation
def test(model):
    fps = 24
    size = (1024, 512) 
    video = cv2.VideoWriter("/share2/public/fail_safe/kitti/DeepBlur/temp4/data_db.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    model.eval()  # switch to evaluate modeSS
    count=0
    end = time.time()
    t = '/share2/public/fail_safe/kitti/qjx_data/defocus_dataset'
    # t = '/share2/public/fail_safe/kitti/DeepBlur/temp4/db2'
    # path1 = t + '/blurmap'
    # folder = os.path.exists(path1)
    # if not folder:                   
    #     os.makedirs(path1)
    # path2 = t + '/blur'
    # folder = os.path.exists(path2)
    # if not folder:                   
    #     os.makedirs(path2)
    # path3 = t + '/noblur'
    # folder = os.path.exists(path3)
    # if not folder:                   
    #     os.makedirs(path3)
    temp_path = t + '/images'
    temp_path2 = t + '/gts'
    files = os.listdir(temp_path)
    files.sort(key = lambda x: int(x.split('_')[-1][:-4]))
    files2 = os.listdir(temp_path2)
    files2.sort(key = lambda x: int(x.split('_')[-1][:-4]))
    blur_flag = 0 
    for i in range(len(files)):
        if i > 199:
            break
        print(files[i])
        print(files2[i])

        img = Image.open(os.path.join(temp_path,files[i]))
        img = img.convert('RGB')
        im_out = np.array(img)
        im = np.array(img)#.astype(np.float32)
        h1,w1,c = im.shape
        im = cv2.resize(im,(512,256),interpolation=cv2.INTER_AREA)
        gt = cv2.imread(os.path.join(temp_path2,files2[i]))
        gt_out = cv2.resize(gt,(512,256),interpolation=cv2.INTER_AREA)
        gt_out = gt_out*1.0/np.max(gt_out)
        gt_out = gt_out*255.0
        htitch1= np.hstack([im.astype('uint8'), gt_out.astype('uint8')])
        temp_out = np.zeros((512,1024,3))
        temp_out[128:384,:,:] = htitch1
        out_img=cv2.cvtColor(temp_out.astype('uint8'), cv2.COLOR_RGB2BGR)
        # utils.save_image(temp_out.astype('uint8'), 'test_d.png')
        # exit(0)        
        video.write(out_img)

        # transform = my_transforms.Compose([
        # ])

        # # transform = my_transforms.Compose([
        # #     my_transforms.Crop(h1-300, 10, h1-100, 1200),
        # #     my_transforms.Resize(460 / 240, interpolation='bilinear'),
        # #     my_transforms.CenterCrop((256,512))
        # # ])

        # im_ = transform(im)
        # im_ = np.array(im_).astype(np.float32)
        # im_ /= 255.0
        # im_ = to_tensor(im_)

        # input = torch.FloatTensor(im_).cuda()
        # input = torch.unsqueeze(input,0)

        # data_time = time.time() - end
        # # compute output
        # end = time.time()
        # with torch.no_grad():
        #     pred,pred_mask,c1,c2,c3,t_time = model(input)
        # b,c,h,w = pred_mask.size()
        # gpu_time = time.time() - end
        # print(gpu_time)
        # temp0 = torch.zeros_like(pred_mask)
        # temp1 = torch.ones_like(pred_mask)
        # pred_mask2 = torch.where(pred_mask>0.5,temp1,temp0)
        # pred[:,1,:,:] = pred[:,1,:,:] * pred_mask2
        # pred = torch.squeeze(pred)
        # pred = pred.data.cpu().numpy()
        
        # temp0_2 = torch.zeros_like(c1)
        # temp1_2 = torch.ones_like(c1)
        # c1_2 = torch.where(c1>0.5,temp1_2,temp0_2)
        # c2_2 = torch.where(c2>0.5,temp1_2,temp0_2)
        # c3_2 = torch.where(c3>0.5,temp1_2,temp0_2)
        
        # torch.cuda.synchronize()

        # # measure accuracy and record loss
        # t1 = c1[0].cpu().numpy()[0]
        # t2 = c2[0].cpu().numpy()[0]
        # t3 = c3[0].cpu().numpy()[0]
        # c1_2 = c1_2.cpu().numpy()
        # c2_2 = c2_2.cpu().numpy()
        # c3_2 = c3_2.cpu().numpy()
        # flag=-1
        # if t1==0.0 and t2==0.0 and t3==0.0:
        #     flag=0
        # fb_map = pred[0,:,:] * t1 / np.max(pred[0,:,:])
        # mb_map = pred[1,:,:] * t2 / np.max(pred[1,:,:])
        # db_map = pred[2,:,:] * t3 / np.max(pred[2,:,:])

        # pred_fb = colored_depthmap(fb_map,0.0,1.0)
        # # pred_fb = cv2.resize(pred_fb,(w1,h1),interpolation=cv2.INTER_LINEAR)
        # pred_mb = colored_depthmap(mb_map,0.0,1.0)
        # # pred_mb = cv2.resize(pred_mb,(w1,h1),interpolation=cv2.INTER_LINEAR)
        # pred_db = colored_depthmap(db_map,0.0,1.0)
        # # pred_db = cv2.resize(pred_db,(w1,h1),interpolation=cv2.INTER_LINEAR)
        # alpha = 0.3
        # fb_max =np.max(pred[0,:,:]*255.0)
        # fb_min =np.min(pred[0,:,:]*255.0)
        # th1 = alpha * fb_max + (1-alpha) * fb_min
        # fb_mask = np.where(pred[0,:,:]*255.0>th1,255.0,0.0)
        # fb_mask = cv2.resize(fb_mask,(w1,h1),interpolation=cv2.INTER_LINEAR)
        # mb_max =np.max(pred[1,:,:]*255.0)
        # mb_min =np.min(pred[1,:,:]*255.0)
        # th2 = alpha * mb_max + (1-alpha) * mb_min
        # mb_mask = np.where(pred[1,:,:]*255.0>th2,255.0,0.0)
        # mb_mask = cv2.resize(mb_mask,(w1,h1),interpolation=cv2.INTER_LINEAR)
        # db_max =np.max(pred[2,:,:]*200.0)
        # db_min =np.min(pred[2,:,:]*200.0)
        # th3 = alpha * db_max + (1-alpha) * db_min
        # db_mask = np.where(pred[2,:,:]*200.0>th3,255.0,0.0)
        # db_mask = cv2.resize(db_mask,(w1,h1),interpolation=cv2.INTER_LINEAR)
        
        # print(th1,th2,th3)
        # # utils.save_image(pred_fb,  t+'/blurmap/fb' + '/fbm_' + files[i])
        # # utils.save_image(pred_mb,  t+'/blurmap/mb' + '/mbm_' + files[i])
        # # utils.save_image(pred_db,  t+'/blurmap/db' + '/dbm_' + files[i])
        # sx1 = 10
        # sy1 = 15
        # if flag==0:
        #     count=0
        #     filename4 = t + '/noblur'+ '/nb_' + files[i]
        #     cv2.putText(im, "None", (int(sx1),int(sy1)), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (255, 0, 0) ,1)
        # else:
        #     count+=1
        #     filename4 = t + '/blur' + '/b_' + files[i]
        #     filename1 = t+'/blurmap' + '/bm_' + files[i]
            
        #     if t1>t2 and t1>t3:
        #         name = "Haze" 
        #     if t2>t1 and t2>t3:
        #         name = "Motion Blur" 
        #     if t3>t1 and t3>t2:
        #         name = "Defocus Blur" 
        
        #     cv2.putText(im, name, (int(sx1),int(sy1)), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (255, 0, 0) ,1)


        # namef = "Haze " + str(round(t1,3))
        # namem = "Motion Blur " + str(round(t2,3))
        # named = "Defocus Blur " + str(round(t3,3))
        # cv2.putText(pred_fb, namef, (int(sx1),int(sy1)), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (255, 0, 0) ,1)
        # cv2.putText(pred_mb, namem, (int(sx1),int(sy1)), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (255, 0, 0) ,1)
        # cv2.putText(pred_db, named, (int(sx1),int(sy1)), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (255, 0, 0) ,1)
        # # utils.save_image(pred_fb, filename1)
        # if count>=5:
        #     blur_flag=1
        # # print(t + '/gt/' + files[i][:-4] + '.png')
        # # gt_mask = cv2.imread(t + '/gt/' + files[i][:-4] + '.png')
        # # print(gt_mask.shape,fb_mask.shape,mb_mask.shape,db_mask.shape)
        # # exit(0)
        # # vtitch = np.vstack([fb_mask.astype('uint8'),mb_mask.astype('uint8'),db_mask.astype('uint8'),gt_mask[:,:,0]])
        # # utils.save_image(vtitch, t + '/all_masks/' + files[i][:-4] + '.png')

        # htitch1= np.hstack([im.astype('uint8'), pred_fb.astype('uint8')])
        # htitch2= np.hstack([pred_mb.astype('uint8'),pred_db.astype('uint8')])
        # vtitch = np.vstack([htitch1,htitch2])


        # temp_cm = np.ones((255,255))
        # print(temp_cm.shape)
        # for i in range(255):
        #     temp_cm[:,i] = i * temp_cm[:,i]
        # cm = colored_depthmap(temp_cm,0,255)
        # utils.save_image(cm.astype(np.uint8), "app_cm.png")
        # break
        # video.release()
        # exit(0)
        # out_img=cv2.cvtColor(vtitch, cv2.COLOR_RGB2BGR)
        # # utils.save_image(vtitch, '/share2/public/fail_safe/kitti/DeepBlur/temp4/icra_app/'+'icra_app'+ str(i) + '.png')        
        # video.write(out_img)
        # if i > 1000:
            # break
        
    video.release()
    if blur_flag==1:
        print("Be careful!!!")
    else:
        print("Have a nice day!")
if __name__ == '__main__':
    main()
