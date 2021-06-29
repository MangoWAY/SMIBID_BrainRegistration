import SimpleITK as sitk
from torch.nn.modules.module import register_module_backward_hook
import ImageGenerator as DG
import DVFGenerator as DVFG
import matplotlib.pyplot as plt
import torch as th
import torch.nn.functional as F
import utils

def image():
    path = "Dataset/SMIBD_masks/crop_10.nii.gz"
    vol = sitk.GetArrayFromImage(sitk.ReadImage(path))
    DG.show_some_sample(vol,modality_num=5,slice = [60,60,70])


def dvf(msize):
    G = DVFG.DVFGenerator()
    dvf = G.sample(1,"cuda:0")
    dvf = F.interpolate(dvf,size=msize)
    dvf = dvf.squeeze(0).permute(1,2,3,0)

    vol = sitk.GetArrayFromImage(sitk.ReadImage("Dataset/SMIBD_images/image_0_0.nii.gz"))
    plt.subplot(2,3,1)
    plt.imshow(vol[60,:,:],cmap="gray")
    plt.subplot(2,3,2)
    plt.imshow(vol[:,60,:],cmap="gray")
    plt.subplot(2,3,3)
    plt.imshow(vol[:,:,70],cmap="gray")


    vol = utils.warp_image(vol,dvf).cpu().detach().numpy()
    plt.subplot(2,3,4)
    plt.imshow(vol[60,:,:],cmap="gray")
    plt.subplot(2,3,5)
    plt.imshow(vol[:,60,:],cmap="gray")
    plt.subplot(2,3,6)
    plt.imshow(vol[:,:,70],cmap="gray")
    plt.show()


def main():
    image()
    dvf([128,128,144])

if __name__=='__main__':
    main()