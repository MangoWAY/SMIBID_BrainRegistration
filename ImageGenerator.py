import torch as th
import SimpleITK as sitk
import torch.distributions as tdist
import pandas as pd
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time
import random


def generate_brain_volume(brainMask, label_num = 27,
                                    need_to_mask = False,
                                    rand_merge = None,
                                    seed = None, 
                                    a_mu = 25, 
                                    b_mu = 225, 
                                    a_sigma = 5, 
                                    b_sigma = 25,
                                    g_sigma = 0.75):

    """
    Given a mask to generate fake multi-modality data

    Args:
        brainMask: 
            brain mask volume data (numpy)
        label_num: 
            how many label in the mask (int)
        need_to_mask: 
            if need to convert [-1, 1] to [0, label_num]
        rand_merge: 
            if need to randomly select some regions to merge. example:[2,2,2], 2 for 1, 2 for 1, 2 for 1
        seed: 
            seed for reproducing experiments
        a_mu: 
            intensity mu for uniform sample [a_mu, b_mu]
        b_mu: 
            intensity mu for uniform sample [a_mu, b_mu]
        a_sigma: 
            intensity sigma for uniform sample [a_sigma, b_sigma]
        b_sigma: 
            intensity sigma for uniform sample [a_sigma, b_sigma]
        g_sigma: 
            gaussian blur sigma
    Return:
        vol:
            numpy volume intensities in [0,1]
    """
    if seed!=None:
        th.manual_seed(seed)
    dists = []
    for i in range(label_num):
        r1 = round(th.rand(1).item(),2)
        r2 = round(th.rand(1).item(),2)
        mu = a_mu + (b_mu - a_mu) * r1
        sigma = a_sigma + (b_sigma - a_sigma) * r2
        dists.append(tdist.Normal(mu,sigma))
    if need_to_mask:
        vol = (brainMask + 1) / 2 * label_num
    a = th.rand(1).item()
    if rand_merge != None and a <0.5: 
        vol = brainMask.astype(np.int32)
        is_used = []
        for num in rand_merge:
            regions = []
            for i in range(num):
                while(True):
                    r = th.randint(1,label_num,(1,1)).item()
                    if r not in is_used:
                        regions.append(r)
                        is_used.append(r)
                        break
            
            regions = sorted(regions)
            for v in regions:
                vol[vol == v] = regions[0]
    else:
        vol = brainMask
    vol = vol.astype(np.float32)
    size = vol.shape
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                l = int(vol[i,j,k])
                if l == 0:
                    continue
                vol[i,j,k] = dists[l - 1].sample((1,1)).item()
    vol = gaussian_filter(vol,g_sigma)
    vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
    return vol


def combine_dvf(dvf_list, flip_pro = 0.5):
    """
    Input a list of dvf and random combine them.

    dvf_list : dvf list, dvf is tensor ans shape is [dim1, dim2, dim3, 3]

    inverse_pro: the probability of flip a dvf to -dvf (default: 0.5)
    """
    num = len(dvf_list)
    weights = th.rand(num)
    weights = weights/th.sum(weights)
    new_dvf = th.zeros_like(dvf_list[0])
    for i in range(num):
        if random.random() < flip_pro:
            dvf_list[i] = - dvf_list[i]
        new_dvf += weights[i] * dvf_list[i]
    return new_dvf


def random_choice(seq, prob, k=1):
    '''
    random choice
    '''
    res = []
    for j in range(k):
        p = random.random()
        for i in range(len(seq)):
            if sum(prob[:i]) < p <= sum(prob[:i+1]):
                res.append(seq[i])
    return res

def show_some_sample(mask,seed = None, modality_num = 10, slice = [40,40,40]):
    """
    show some Fake Data

    Args:
        mask: 
            data segmentation mask volume (numpy)
        seed: 
            random seed (int)
        modality_num: 
            the number of modality (int)
        slice: 
            data slice index (list)
    """
    begin = time.time()
    if seed !=None:
        th.manual_seed(seed)
    for i in range(modality_num):
        newvol = generate_brain_volume(mask,rand_merge=[2,2,2])
        plt.subplot(3,modality_num,i + 1)
        plt.imshow(newvol[slice[0],:,:],cmap='gray')
        plt.subplot(3,modality_num,i + 1 + modality_num)
        plt.imshow(newvol[:,slice[1],:],cmap='gray')
        plt.subplot(3,modality_num,i + 1 + modality_num * 2)
        plt.imshow(newvol[:,:,slice[2]],cmap='gray')
        print("Done..",i)
    print(time.time() - begin)
    plt.show()

def generate_multi_modality_data(path, name, mask, spacing = (1,1,1), modality_num = 5, rand_merge=[2,2,2]):
    """
    Write Fake data to file

    Args:
        path:
            write path (str)
        mask:
            mask data (numpy)
        spacing:
            physical distance mm for itk image (list or tuple)
        modality_num:
            how many modalities (int)
    """
    for i in range(modality_num):
        brain_vol = generate_brain_volume(mask,rand_merge=rand_merge)
        brain_vol = sitk.GetImageFromArray(brain_vol)
        brain_vol.SetSpacing(spacing)
        sitk.WriteImage(brain_vol,os.path.join(path,name + "_{0}.nii.gz".format(i)))

