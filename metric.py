from skimage import metrics
from PIL import Image
from numpy import asarray
import glob
import os


def calculate_psnr(ground_truth_img_array, corrected_img_array, no_corrected_img_array):
    ##calculate the psnr values##
    corrected_psnr = metrics.peak_signal_noise_ratio(ground_truth_img_array, corrected_img_array)
    no_corrected_psnr = metrics.peak_signal_noise_ratio(ground_truth_img_array, no_corrected_img_array)
    print('psnr with correction filter:', corrected_psnr)
    print('psnr without correction filter:', no_corrected_psnr)


def calculate_ssim(ground_truth_img_array, corrected_img_array, no_corrected_img_array):
    ##calculate the ssim values##
    corrected_ssim = metrics.structural_similarity(ground_truth_img_array, corrected_img_array, multichannel=True)
    no_corrected_ssim = metrics.structural_similarity(ground_truth_img_array, no_corrected_img_array, multichannel=True)
    print('ssim with correction filter:', corrected_ssim)
    print('ssim without correction filter:', no_corrected_ssim)


def metric_evaluation():
    for file in glob.glob('./testing_data/*.png'):
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_length = len(file_name)
        result_imgs_list = []
        for result_file in glob.glob('./DBPN_Results/*.png'):
            result_file_name = os.path.splitext(os.path.basename(result_file))[0]
            result_file_name_list = list(result_file_name)
            result_file_name_list = result_file_name_list[:file_name_length]
            result_file_name_edit = ''.join(result_file_name_list)
            if file_name == result_file_name_edit:
                result_imgs_list.append(result_file)
        length = 0
        correct_img_index = 0
        ground_truth_img_path = ''

        for i in range(len(result_imgs_list)):
            if len(list(result_imgs_list[i])) > length:
                length = len(list(result_imgs_list[i]))
                ground_truth_img_path = result_imgs_list[i]
                correct_img_index = i

        if len(result_imgs_list) > 0:
            ##get the resulting HR image which has been applied with the correction filter##
            corrected_img_path = result_imgs_list[correct_img_index]
            corrected_img = Image.open(corrected_img_path)
            corrected_img_array = asarray(corrected_img)

            ##get the resulting HR image which has not been applied with the correction filter##
            no_corrected_img_index = 1 if correct_img_index == 0 else 0
            no_corrected_img_path = result_imgs_list[no_corrected_img_index]
            no_corrected_img = Image.open(no_corrected_img_path)
            no_corrected_img_array = asarray(no_corrected_img)

            ##get the ground truth HR image##
            h, w, c = corrected_img_array.shape
            ground_truth_img_path = file
            ground_truth_img = Image.open(ground_truth_img_path).resize((w, h))
            ground_truth_img_array = asarray(ground_truth_img)

            ##calculate psnr and ssim##
            print('Ground Truth Image:', file_name)
            calculate_psnr(ground_truth_img_array, corrected_img_array, no_corrected_img_array)
            calculate_ssim(ground_truth_img_array, corrected_img_array, no_corrected_img_array)
            print('\n')



if __name__ == "__main__":
    metric_evaluation()










