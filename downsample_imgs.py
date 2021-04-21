import util
import os
import numpy as np
import torchvision.transforms as transforms
import scipy.io as io
import argparse
import torch
from PIL import Image

##parameters##
parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='./Test_Img/', type=str)
parser.add_argument('--out_dir', default='./Downsample', type=str)
parser.add_argument('--standard_deviation', default=4.5/np.sqrt(2), type=float)
parser.add_argument('--scale_factor', type=int, default=4)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####
in_dir = args.in_dir
# out_GT = './GT_x%d/' %scale_factor

##set the downsampling kernel##
scale_factor = args.scale_factor
standard_deviation = args.standard_deviation
sampling = util.get_gauss_flt(32, standard_deviation).to(device)
sampling = sampling/sampling.sum()


imgs = [f for f in os.listdir(in_dir) if '.png' in f]
imgs.sort()

for img in imgs:
    I = util.load_img_torch(in_dir + img, device)
    ##preprocess the image for kernel transformation##
    if I.shape[2] % scale_factor:
        I = I[:,:,:-(I.shape[2]%scale_factor),:]
    if I.shape[3] % scale_factor:
        I = I[:,:,:,:-(I.shape[3]%scale_factor)]
    # util.save_img_torch(I, out_GT + img)

    ##generated the saving path##
    out_dir_png = args.out_dir + '/Gaussian/'
    out_dir_mat = args.out_dir + '/Gaussian_mat/'
    out_dir_bicubic = args.out_dir + '/Bicubic/'
    out_dir_filter = args.out_dir + '/Filter/'

    if not os.path.exists(out_dir_png):
        os.makedirs(out_dir_png)
    if not os.path.exists(out_dir_mat):
        os.makedirs(out_dir_mat)
    if not os.path.exists(out_dir_bicubic):
        os.makedirs(out_dir_bicubic)
    if not os.path.exists(out_dir_filter):
        os.makedirs(out_dir_filter)

    ##get the downsampled image generated by gaussian kernel##
    y = util.fft_Down_(I, sampling, scale_factor)
    y_np = np.moveaxis(np.array(torch.abs(y)[0,:].cpu()), 0, 2)

    ##save the downsampled image generated by gaussian kernel##
    util.save_img_torch(torch.abs(y), out_dir_png + img[:-4] + '_Gauss_std%1.1f_x%d.png' % (standard_deviation, scale_factor))
    io.savemat(out_dir_mat + img[:-4] + '_Gauss_std%1.1f_x%d_s.mat' %(standard_deviation, scale_factor), {'img': y_np})

    ##get the downsampled image generated by bicubic kernel##
    I_PIL = transforms.ToPILImage()(I[0,:].cpu())
    W, H = I_PIL.size
    I_PIL_bic_down = I_PIL.resize((W//scale_factor, H//scale_factor), Image.BICUBIC)

    ##save the downsampled image generated by bicubic kernel##
    I_PIL_bic_down.save(out_dir_bicubic + img[:-4] + '_bicubic_down_PIL.png')

    ##generate the filter image##
    S = util.fft_torch(sampling, y.shape[2:4])
    s_ = torch.roll(torch.fft.ifftn(S, dim=(-2,-1)).real, (S.shape[2]//2, S.shape[3]//2), dims=(2,3))

    ##save the filter##
    util.save_img_torch(s_/s_.max(), out_dir_filter + img[:-4] + '_Gauss_std%1.1f_x%d_s.png' %(standard_deviation, scale_factor))
