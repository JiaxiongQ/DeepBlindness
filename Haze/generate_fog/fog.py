import os
import sys
import numpy as np
import skimage
import skimage.io
import skimage.transform
import numpy as np
import random
import math
import cv2
def get_dark_channel(I, w):
    """Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = I.shape
    padded = np.pad(I, ((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch
    #impad = np.pad(I, [(int(ps/2),int(ps/2)), (int(ps/2),int(ps/2)) , (0,0)], 'edge')
    
    #Jdark is the Dark channel to be found
    #jdark = np.zeros((I.shape[0],I.shape[1]))
    
    #for i in range(int(ps/2), (I.shape[0]+int(ps/2))):
    #    for j in range(int(ps/2), (I.shape[1]+int(ps/2))):
    #        #creates the patch P(x) of size ps x ps centered at x
    #        patch = impad[i-int(ps/2):i+1+int(ps/2), j-int(ps/2):j+1+int(ps/2)]
            #selects the minimum value in this patch and set as the dark channel of pixel x
    #        jdark[i-int(ps/2), j-int(ps/2)] = patch.min()
    
    #return jdark


def get_atmosphere(I, darkch, p):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    I:      the M * N * 3 RGB image data ([0, L-1]) as numpy array
    darkch: the dark channel prior of the image as an M * N numpy array
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    # reference CVPR09, 4.4
    #M, N = darkch.shape
    #flatI = I.reshape(M * N, 3)
    #flatdark = darkch.ravel()
    #searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes

    # return the highest intensity for each channel
    #return np.max(flatI.take(searchidx, axis=0), axis=0)
    imgavec = np.resize(I, (I.shape[0]*I.shape[1], I.shape[2]))
    jdarkvec = np.reshape(darkch, darkch.size)
    
    #the number of pixels to be considered
    numpx = np.int(darkch.size * p)
    
    #index sort the jdark channel in descending order
    isjd = np.argsort(-jdarkvec)

    asum = np.array([0.0,0.0,0.0])
    for i in range(0, numpx):
        asum[:] += imgavec[isjd[i], :]
  
    A = np.array([0.0,0.0,0.0])
    A[:] = asum[:]/numpx
    return A

def generate_fog(img,depth):
    #print(np.max(depth),np.min(depth))
    #depth = depth * 1.0
    I = img.astype(np.float32)*1.0/255
    darkch = get_dark_channel(I, 15)
    A = get_atmosphere(I, darkch, 0.02)
    print(A)
    #exit(0)
    #A = random.uniform(0.4,1.1)
    beta = random.uniform(0.02,0.08)
    t_map = np.zeros_like(depth)
    for i in range(t_map.shape[0]):
        for j in range(t_map.shape[1]):
            t_map[i][j] = math.exp(-1.0*beta*depth[i][j])
    tmap = np.zeros_like(img).astype(np.float32)
    tmap[:,:,0] = t_map
    tmap[:,:,1] = t_map
    tmap[:,:,2] = t_map
    temp = np.ones_like(tmap) - tmap
    #temp[:,:,0] = A[0] * temp[:,:,0]
    #temp[:,:,1] = A[1] * temp[:,:,1]
    #temp[:,:,2] = A[2] * temp[:,:,2]
    result = img*tmap + np.mean(A)*temp*255 
    return t_map,result
inx = 0
for line in open("/share2/public/fail_safe/kitti/qjx_data/outputD.txt"):
    inx = inx + 1
    #if(inx < 1000): 
    #    continue
    print(inx)
    left_test = line.split('&')[0]
    depth_test = line.split('&')[1][:-1].replace('lidar_depth','dense_depth')

    img_path = left_test
    depth_path = depth_test

    img =  skimage.io.imread(img_path)
    depth = skimage.io.imread(depth_path).astype(np.float32)*1.0/256
    img = img[50:,:,:]
    depth = depth[50:,:]
    depth=cv2.medianBlur(depth,5)
    #kernel=np.ones((5,5),np.float32)/25
    #depth=cv2.filter2D(depth,-1,kernel) 
    #depth = cv2.bilateralFilter(depth,9,75,75)
    depth=cv2.blur(depth,(5,5))
    tmap,fog = generate_fog(img,depth)
    
    skimage.io.imsave(os.path.join('/share2/public/fail_safe/kitti/qjx_data/fog_dataset/images', 'output_fog_%d.jpg'%inx), fog.astype(np.uint8))
    skimage.io.imsave(os.path.join('/share2/public/fail_safe/kitti/qjx_data/fog_dataset/tmaps', 'output_tmap_%d.jpg'%inx), (tmap*255).astype(np.uint8))
       
    #exit(0)
