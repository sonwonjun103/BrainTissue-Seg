import os
import random
import glob
import PIL
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from utils.parser import set_parser
from utils.seed import seed_everything

from models.Unet import Model

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.spatial.distance import directed_hausdorff

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True

# Load data
def get_path(folder):
    path = f"D:\\ACPC\\DATA\\"

    ctniipath, mrniipath, gmniipath, wmniipath, csfniipath = [], [], [], [], []

    for f in folder:
        fctpath = f"{path}{f}\\CT\\"
        for file in glob.glob(fctpath + "/*.nii.gz"):
            ctniipath.append(file)

        fmrpath = f"{path}{f}\\MRI\\"
        for file in glob.glob(fmrpath + "/*.nii.gz"):
            mrniipath.append(file)

        fgmpath = f"{path}{f}\\GM\\"
        for file in glob.glob(fgmpath + "/*.nii.gz"):
            gmniipath.append(file)   

        fwmpath = f"{path}{f}\\WM\\"
        for file in glob.glob(fwmpath + "/*.nii.gz"):
            wmniipath.append(file)      

        fcsfpath = f"{path}{f}\\CSF\\"
        for file in glob.glob(fcsfpath + "/*.nii.gz"):
            csfniipath.append(file)

    return ctniipath, mrniipath, gmniipath, wmniipath, csfniipath

def set_train_test():
    testinfo = pd.read_excel(f"D:\\ACPC\\testinfo_50.xlsx")
    test_patient = list(testinfo['test_index'])

    test_ct, test_mr, test_gm, test_wm, test_csf = get_path(test_patient)

    return test_ct, test_mr, test_gm, test_wm, test_csf

# Make Dataset
class Customdataset(Dataset):
    def __init__(self, ctpath, mrpath, gmpath, wmpath, csfpath, rgb=False,transform=None):
        self.ctpath = ctpath
        self.mrpath = mrpath
        self.gmpath = gmpath
        self.wmpath = wmpath
        self.csfpath = csfpath
        self.transform = transform
        self.rgb = rgb

    def __get_img(self, path):
        nii=nib.load(path)
        img=nii.get_fdata()

        img = np.flip(img, axis=0)

        return img
    
    def __min_max_normalization(self, img):
        small = np.min(img)
        big = np.max(img)

        if big==0:
            return img
        else:
            return (img - small) / (big-small)
        
    def __len__(self):
            return len(self.ctpath)
    
    def custom_resize(self, image, new_size):
        pil_image = PIL.Image.fromarray(image)
        pil_image = pil_image.resize(new_size, resample=PIL.Image.NEAREST)  # Use NEAREST resampling to avoid interpolation
        return np.array(pil_image)
    
    def __getitem__(self, index):
        ctpath = self.ctpath
        mrpath = self.mrpath
        gmpath = self.gmpath
        wmpath = self.wmpath
        csfpath = self.csfpath

        number = ctpath.split('\\')[3]
        slice_number = ctpath.split('\\')[5].split('.')[0]

        ctimg, mrimg, gmimg, wmimg, csfimg= self.__get_img(ctpath), self.__get_img(mrpath), self.__get_img(gmpath), self.__get_img(wmpath), self.__get_img(csfpath)

        ct_image = self.__min_max_normalization(ctimg)

        input_image = self.custom_resize(ct_image, (256,256))
        mr_image = self.custom_resize(mrimg, (256,256))
        gm_image = self.custom_resize(gmimg, (256,256))
        wm_image = self.custom_resize(wmimg, (256,256))
        csf_image = self.custom_resize(csfimg, (256,256))

        if self.rgb==True:
            h,w = input_image.shape
            rgb_image = np.zeros((h,w,3))

            rgb_image[:,:,0]=input_image
            rgb_image[:,:,1]=input_image
            rgb_image[:,:,2]=input_image

            return torch.from_numpy(rgb_image).permute(2,0,1), torch.from_numpy(gm_image).unsqueeze(0), torch.from_numpy(wm_image).unsqueeze(0), torch.from_numpy(csf_image).unsqueeze(0), number, slice_number
        else:            
            return torch.from_numpy(input_image).unsqueeze(0), torch.from_numpy(mr_image).unsqueeze(0), torch.from_numpy(gm_image).unsqueeze(0), torch.from_numpy(wm_image).unsqueeze(0), torch.from_numpy(csf_image).unsqueeze(0), number, slice_number

# Define Model
def get_model(args, path):
    model = Model(1, 1).to(args.device)

    model = nn.DataParallel(model).to(args.device)
    model.load_state_dict(torch.load(path))

    return model

# Evaluation Metric
def scale_img(img):
    image = (img - np.min(img) / np.max(img))
    return np.clip(image, 0, 1)

def matchimg(img, gpu):
    if gpu:
        return img.squeeze(0).squeeze(0).detach().cpu().numpy()
    else:
        return img.squeeze(0).squeeze(0).detach().numpy()
    
def cdice(pred, true):
    intersection = pred*true
    c = np.sum(intersection) / max(np.size(intersection[intersection>0]), 1)

    cDC = 2*(np.sum(intersection)) / (c*np.sum(true) + np.sum(pred))

    return cDC

def ssim_value(pred, true):
    return ssim(pred, true)

def psnr_value(pred, true):
    return psnr(pred, true, data_range=1)

def hausdorff_value(pred, true):
    atob =  directed_hausdorff(pred, true)[0]
    btoa = directed_hausdorff(true, pred)[0]

    return atob, btoa

def evaluation_metric(pred, true):
    dice = cdice(pred, true)
    ssim = ssim_value(pred, true)
    psnr = psnr_value(pred, true)
    ptot, ttop = hausdorff_value(pred, true)

    return dice, ssim, psnr, ptot, ttop

# For plot
def plot_result(args, ct, gm, wm, csf, number, slice_number,
                gmmodel1, gmmodel2, 
                wmmodel1, wmmodel2,
                csfmodel1, csfmodel2):
    gmpred1, wmpred1, csfpred1 = gmmodel1(ct.to(args.device).float()), wmmodel1(ct.to(args.device).float()), csfmodel1(ct.to(args.device).float())
    gmpred2, wmpred2, csfpred2 = gmmodel2(ct.to(args.device).float()), wmmodel2(ct.to(args.device).float()), csfmodel2(ct.to(args.device).float())

    ctimg, gmimg, wmimg, csfimg = matchimg(ct, 0), matchimg(gm, 0), matchimg(wm, 0), matchimg(csf, 0)

    gmpred1, wmpred1, csfpred1 = matchimg(gmpred1, 1), matchimg(wmpred1, 1), matchimg(csfpred1, 1)
    gmpred2, wmpred2, csfpred2 = matchimg(gmpred2, 1), matchimg(wmpred2, 1), matchimg(csfpred2, 1)

    gmpred1, wmpred1, csfpred1 = scale_img(gmpred1), scale_img(wmpred1), scale_img(csfpred1)
    gmpred2, wmpred2, csfpred2 = scale_img(gmpred2), scale_img(wmpred2), scale_img(csfpred2)

    gmdice1, gmssim1, gmpsnr1, gmptot1, gmttop1 = evaluation_metric(gmpred1, gmimg)
    wmdice1, wmssim1, wmpsnr1, wmptot1, wmttop1 = evaluation_metric(wmpred1, wmimg)
    csfdice1, csfssim1, csfpsnr1, csfptot1, csfttop1 = evaluation_metric(csfpred1, csfimg)

    gmdice2, gmssim2, gmpsnr2, gmptot2, gmttop2 = evaluation_metric(gmpred2, gmimg)
    wmdice2, wmssim2, wmpsnr2, wmptot2, wmttop2 = evaluation_metric(wmpred2, wmimg)
    csfdice2, csfssim2, csfpsnr2, csfptot2, csfttop2 = evaluation_metric(csfpred2, csfimg)

    plt.figure(figsize=(5,10))
    plt.style.use('grayscale')

    h, w = 4, 3
    plt.subplot(h, w, 1)
    plt.imshow(np.flip(ctimg, axis=0))
    plt.axis('off')
    plt.title('CT')

    plt.subplot(h,w, 4)
    plt.imshow(np.flip(gmimg, axis=0))
    plt.axis('off')
    plt.title('GM')
    
    plt.subplot(h,w, 5)
    plt.imshow(np.flip(gmpred1, axis=0))
    plt.axis('off')
    plt.title(f'L1&L2 {gmdice1:>.3f}')

    plt.subplot(h,w,6)
    plt.imshow(np.flip(gmpred2, axis=0))
    plt.axis('off')
    plt.title(f'L1&L2&Per {gmdice2:>.3f}')

    plt.subplot(h,w, 7)
    plt.imshow(np.flip(wmimg, axis=0))
    plt.axis('off')
    plt.title('WM')
    
    plt.subplot(h,w, 8)
    plt.imshow(np.flip(wmpred1, axis=0))
    plt.axis('off')
    plt.title(f'L1&L2 {wmdice1:>.3f}')

    plt.subplot(h,w,9)
    plt.imshow(np.flip(wmpred2, axis=0))
    plt.axis('off')
    plt.title(f'L1&L2&Per {wmdice2:>.3f}')

    plt.subplot(h,w, 10)
    plt.imshow(np.flip(csfimg, axis=0))
    plt.axis('off')
    plt.title('CSF')
    
    plt.subplot(h,w,11)
    plt.imshow(np.flip(csfpred1, axis=0))
    plt.axis('off')
    plt.title(f'L1&L2 {csfdice1:>.3f}')

    plt.subplot(h,w,12)
    plt.imshow(np.flip(csfpred2, axis=0))
    plt.axis('off')
    plt.title(f'L1&L2&Per {csfdice2:>.3f}')

    #print(f"{number} {slice_number}")
    #print(f"Scratch Unet L1&L2")
    #print(f"GM\n  Dice : {gmdice1:>.3f}\n  SSIM : {gmssim1:>.3f}\n  PSNR : {gmpsnr1:>.3f}\n  HAUS (ptot) : {gmptot1:>.3f} (ttop) : {gmttop1:>.3f}\n")
    #print(f"WM\n Dice : {wmdice1:>.3f}\n SSIM : {wmssim1:>.3f}\n PSNR : {wmpsnr1:>.3f}\n HAUS (ptot) : {wmptot1:>.3f} (ttop) : {wmttop1:>.3f}\n")
    #print(f"CSF\n Dice : {csfdice1:>.3f}\n SSIM : {csfssim1:>.3f}\n PSNR : {csfpsnr1:>.3f}\n HAUS (ptot) : {csfptot1:>.3f} (ttop) : {csfttop1:>.3f}\n")

    #print(f"Scratch Unet L1&L2&perceptual(5maxpool)")
    #print(f"GM\n  Dice : {gmdice2:>.3f}\n  SSIM : {gmssim2:>.3f}\n  PSNR : {gmpsnr2:>.3f}\n  HAUS (ptot) : {gmptot2:>.3f} (ttop) : {gmttop2:>.3f}\n")
    #print(f"WM\n Dice : {wmdice2:>.3f}\n SSIM : {wmssim2:>.3f}\n PSNR : {wmpsnr2:>.3f}\n HAUS (ptot) : {wmptot2:>.3f} (ttop) : {wmttop2:>.3f}\n")
    #print(f"CSF\n Dice : {csfdice2:>.3f}\n SSIM : {csfssim2:>.3f}\n PSNR : {csfpsnr2:>.3f}\n HAUS (ptot) : {csfptot2:>.3f} (ttop) : {csfttop2:>.3f}\n")

    plt.savefig(f"./Result/{number}_{slice_number}.png")

if __name__=='__main__':
    args = set_parser()

    test_ct, test_mr, test_gm, test_wm, test_csf = set_train_test()

    seed_everything(2023)
    datasize = len(test_ct)

    print(f"Test Data size : {datasize}")

    gmmodel1 = get_model(args, f"./model_parameters/GM_unet_L1_L2.pt")
    wmmodel1 = get_model(args, f"./model_parameters/WM_unet_L1_L2.pt")
    csfmodel1 = get_model(args, f"./model_parameters/CSF_unet_L1_L2.pt")
    gmmodel2 = get_model(args, f"./model_parameters/GM_unet_L1_L2_perceptual.pt")
    wmmodel2 = get_model(args, f"./model_parameters/WM_unet_L1_L2_perceptual.pt")
    csfmodel2 = get_model(args, f"./model_parameters/CSF_unet_L1_L2_perceptual.pt")

    print(f"Model Parameter : {sum(p.numel() for p in gmmodel1.parameters())}")

    Tgmdice1, Tgmssim1, Tgmpsnr1, Tgmptot1, Tgmttop1, Tgmahd1 = [],[],[],[],[],[]
    Tgmdice2, Tgmssim2, Tgmpsnr2, Tgmptot2, Tgmttop2, Tgmahd2 = [],[],[],[],[],[]
    Twmdice1, Twmssim1, Twmpsnr1, Twmptot1, Twmttop1, Twmahd1 = [],[],[],[],[],[]
    Twmdice2, Twmssim2, Twmpsnr2, Twmptot2, Twmttop2, Twmahd2 = [],[],[],[],[],[]
    Tcsfdice1, Tcsfssim1, Tcsfpsnr1, Tcsfptot1, Tcsfttop1, Tcsfahd1 = [],[],[],[],[],[]
    Tcsfdice2, Tcsfssim2, Tcsfpsnr2, Tcsfptot2, Tcsfttop2, Tcsfahd2 = [],[],[],[],[],[]

    #i = np.random.randint(0, datasize-1)                       
    for i in range(datasize-1):
        test_dataset = Customdataset(test_ct[i], test_mr[i], test_gm[i], test_wm[i], test_csf[i], rgb=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        ct, mr, gm, wm, csf, number, slice_number = next(iter(test_dataloader))

        plot_result(args, ct, gm, wm, csf, number, slice_number,
                        gmmodel1, gmmodel2, 
                        wmmodel1, wmmodel2,
                        csfmodel1, csfmodel2)
        print(f"{i} => {number} {slice_number} complete!")