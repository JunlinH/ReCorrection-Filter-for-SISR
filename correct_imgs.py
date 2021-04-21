import numpy as np
import torch
import util
import os
import correction_func
import scipy.io as io
import argparse

##parameters##
parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='./Downsample/gaussian_mat/', type=str)
parser.add_argument('--out_dir', default='./LR_Images_Corrected/', type=str)
parser.add_argument('--opt_suffix', default='', type=str)
parser.add_argument('--standard_deviation', default=4.5/np.sqrt(2), type=float)
parser.add_argument('--scale_factor', default=4, type=int)
parser.add_argument('--eps', default=0, type=float)
args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##create the saving path##
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


def define_basis():
    # Define the reconstruction basis
    reconstruction = util.get_bicubic(args.scale_factor).to(args.device)
    reconstruction = reconstruction / reconstruction.sum()

    # Define the sampling basis
    standard_deviation = args.standard_deviation
    s_size = 32 + args.scale_factor % 2
    sampling = util.get_gauss_flt(s_size, standard_deviation).to(args.device)
    sampling = sampling / sampling.sum()

    return reconstruction, sampling

def correct_img(reconstruction, sampling):
    imgs = [f for f in os.listdir(args.in_dir) if os.path.isfile(os.path.join(args.in_dir, f)) and ('.mat' in f)]
    imgs.sort()

    for img_in in imgs:
        y = np.moveaxis(io.loadmat(args.in_dir + img_in)['img'], 2, 0)
        y = torch.tensor(y.real).float().unsqueeze(0).to(args.device)

        Corr_flt = correction_func.Correction_Filter(sampling, args.scale_factor,
                                                     (y.shape[2] * args.scale_factor, y.shape[3] * args.scale_factor),
                                                     r=reconstruction, eps=args.eps, inv_type='Tikhonov')

        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        img = img_in[0:-4] + '_x%d_corr.png' % (args.scale_factor)

        y_h = Corr_flt.correct_img(y)

        util.save_img_torch(y_h.real, args.out_dir + img[0:-4] + '_corrected.png', clamp=True)


if __name__ == "__main__":
    reconstruction, sampling = define_basis()
    correct_img(reconstruction, sampling)