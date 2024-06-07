from torch.utils.data import Dataset

import nibabel as nib
import torch
import PIL
import numpy as np

class Customdataset(Dataset):
    def __init__(self, args, rctpath, ctpath, mrpath, gmpath, wmpath, csfpath, transform=None):
        self.args = args
        self.rctpath = rctpath
        self.ctpath = ctpath
        self.mrpath = mrpath
        self.gmpath = gmpath
        self.wmpath = wmpath
        self.csfpath = csfpath 
        self.transform = transform

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
        ctpath = self.ctpath[index]
        rctpath = self.rctpath[index]
        mrpath = self.mrpath[index]
        gmpath = self.gmpath[index]
        wmpath = self.wmpath[index]
        csfpath = self.csfpath[index]

        rctimg, ctimg, mrimg, gmimg, wmimg, csfimg = self.__get_img(rctpath), self.__get_img(ctpath), self.__get_img(mrpath), self.__get_img(gmpath), self.__get_img(wmpath), self.__get_img(csfpath)

        ct_image = self.__min_max_normalization(ctimg)
        rct_image = self.__min_max_normalization(rctimg)

        if self.args.raw == 1:
            input_image = rct_image
        else:
            input_image = ct_image

        if self.args.region == 'GM':
            output_image = gmimg
        elif self.args.region == 'WM':
            output_image = wmimg
        elif self.args.region == 'CSF':
            output_image = csfimg

        input_image = self.custom_resize(input_image, (self.args.IMAGE_SIZE, self.args.IMAGE_SIZE))
        output_image = self.custom_resize(output_image, (self.args.IMAGE_SIZE, self.args.IMAGE_SIZE))

        if self.args.rgb == True:
            h,w = input_image.shape
            rgb_image = np.zeros((h,w,3))

            rgb_image[:,:,0]=input_image
            rgb_image[:,:,1]=input_image
            rgb_image[:,:,2]=input_image

            return torch.from_numpy(rgb_image).permute(2,0,1), torch.from_numpy(output_image).unsqueeze(0)

        else:
            return torch.from_numpy(input_image).unsqueeze(0), torch.from_numpy(output_image).unsqueeze(0)