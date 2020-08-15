# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import cv2
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            return img.convert('RGB')
        else:
            return img.convert('I')
def fdm_loader(path):
    img = Image.open(path)
    img = img.convert('I')
    img = np.array(img).astype(np.float32)
    h = img.shape[0]
    w = img.shape[1]
    mask = np.zeros((h,w),dtype=np.float32)
    ratio = 255.0
    mask = np.where(img>0.0,1.0,0.0)
    img = img * 1.0 / ratio
    return img,mask

def readPathFiles(file_path, root_dir):
    im_gt_paths = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if '&' in line:
                im_path = os.path.join(root_dir, line.split('&')[0])
                if '\n' in os.path.join(root_dir, line.split('&')[1]):
                    gt_path = os.path.join(root_dir, line.split('&')[1])[:-1]
                else:
                    gt_path = os.path.join(root_dir, line.split('&')[1])
            else:
                if '\n' in os.path.join(root_dir, line):
                    im_path = os.path.join(root_dir, line)[:-1]
                else:
                    im_path = os.path.join(root_dir, line)
                gt_path = '&'
            im_gt_paths.append((im_path, gt_path))
    return im_gt_paths

# array to tensor
from dataloaders import transforms as my_transforms
to_tensor = my_transforms.ToTensor()

class KittiFolder(Dataset):

    def __init__(self, root_dir='',
                 mode='train', loader=pil_loader, gtloader = fdm_loader,
                 size=(256, 512)):
        super(KittiFolder, self).__init__()
        self.root_dir = root_dir

        self.mode = mode
        self.im_gt_paths = None
        self.loader = loader
        self.gtloader = gtloader
        self.size = size

        if self.mode == 'train':
            self.im_gt_paths = readPathFiles('/share2/public/fail_safe/kitti/DeepBlur/tool/filenames/CUHK_train2.txt', root_dir)

        elif self.mode == 'test':
            self.im_gt_paths = readPathFiles('/share2/public/fail_safe/kitti/DeepBlur/tool/filenames/CUHK_test2.txt', root_dir)

        else:
            print('no mode named as ', mode)
            exit(-1)

    def __len__(self):
        return len(self.im_gt_paths)

    def train_transform(self, im, gt, mask):
        im = np.array(im).astype(np.float32)
        im = cv2.resize(im,(512,256),interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt,(512,256),interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask,(512,256),interpolation=cv2.INTER_AREA)

        # h,w,c = im.shape
        # th, tw = 256,512
        # x1 = random.randint(0, w - tw)
        # y1 = random.randint(0, h - th)
        # img = im[y1:y1 + th, x1:x1 + tw, :]
        # gt = gt[y1:y1 + th, x1:x1 + tw]
        # mask = mask[y1:y1 + th, x1:x1 + tw]
        s = np.random.uniform(1.0, 1.5)  # random scaling
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        color_jitter = my_transforms.ColorJitter(0.4, 0.4, 0.4)

        transform = my_transforms.Compose([
            my_transforms.Rotate(angle),
            my_transforms.Resize(s),
            my_transforms.CenterCrop(self.size),
            my_transforms.HorizontalFlip(do_flip)
        ])

        im_ = transform(im)
        im_ = color_jitter(im_)

        gt_ = transform(gt)
        mask_ = transform(mask)
        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)
        mask_ = np.array(mask_).astype(np.float32)        

        im_ /= 255.0
        gt_ /= s
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)
        mask_ = to_tensor(mask_)
        
        gt_ = gt_.unsqueeze(0)
        mask_ = mask_.unsqueeze(0)
        
        return im_, gt_, mask_

    def val_transform(self, im, gt,mask):
        im = np.array(im).astype(np.float32)
        h,w,c = im.shape
        im = cv2.resize(im,(512,256),interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt,(512,256),interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask,(512,256),interpolation=cv2.INTER_AREA)
        
        transform = my_transforms.Compose([
            # my_transforms.Crop(130, 10, 240, 1200),
            # my_transforms.Resize(460 / 240, interpolation='bilinear'),
            # my_transforms.CenterCrop(self.size)
        ])

        im_ = transform(im)
        gt_ = transform(gt)
        mask_ = transform(mask)
        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)
        mask_ = np.array(mask_).astype(np.float32)
        im_ /= 255.0
        im_ = to_tensor(im_).numpy()
        gt_ = to_tensor(gt_)
        mask_ = to_tensor(mask_)
        gt_ = gt_.unsqueeze(0).numpy()
        mask_ = mask_.unsqueeze(0).numpy()
        # im_ = np.reshape(im_,[1,3,im.shape[0],im.shape[1]])
        # gt_ = np.reshape(gt_,[1,1,im.shape[0],im.shape[1]])
        # mask_=np.reshape(mask_,[1,1,im.shape[0],im.shape[1]])
        im_ = np.reshape(im_,[1,3,256,512])
        gt_ = np.reshape(gt_,[1,1,256,512])
        mask_=np.reshape(mask_,[1,1,256,512])
        # if w>1100:
        #     top_pad = 384-im.shape[0]
        #     left_pad = 1248-im.shape[1]
        # else:
        #     top_pad = 384-im.shape[0]
        #     left_pad = 624-im.shape[1]

        # im_ = np.lib.pad(im_,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        # gt_ = np.lib.pad(gt_,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        # mask_ = np.lib.pad(mask_,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        return im_, gt_, mask_

    def __getitem__(self, idx):
        im_path, gt_path = self.im_gt_paths[idx]
        im = self.loader(im_path)
        h,w,c = np.array(im).shape
        gt = np.zeros((h,w),dtype = np.float32)
        label = -1
        if '&' in gt_path:
            mask = np.zeros((h,w),dtype=np.float32)
        else:
            gt,mask = self.gtloader(gt_path)
            if "motion" in gt_path:
                label = 1
            elif "out_of_focus" in gt_path:
                label = 2
        if self.mode == 'train':
            im, gt, mask = self.train_transform(im, gt, mask)
        else:
            im, gt, mask = self.val_transform(im, gt, mask)
        # print(im.size(),gt.size(),mask.size(),label)
        # exit(0)
        return im, gt, label, mask

if __name__ == '__main__':
    paths = readPathFiles('/share2/public/fail_safe/kitti/DeepBlur/tool/filenames/output_test.txt','')
    for i in range(len(paths)):
        im_path, gt_path = paths[i]
        temp1 = pil_loader(im_path, rgb=True)
        if '&' in gt_path:
            pass
        else:
            temp2 = fdm_loader(gt_path)
