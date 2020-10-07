import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.io
from scipy import optimize

from feature_func import *
from preprocess import *
from utils import *


def fit_data(gt_count, feature_data, function):
    return optimize.curve_fit(function, feature_data, gt_count)


def plot_data(gt_count, feature_data, test_func=None):
    plt.scatter(feature_data, gt_count, label='raw data')

    if test_func != None:
        params, params_var = fit_data(gt_count, feature_data, test_func)
        x_linspace = np.linspace(min(feature_data), max(feature_data), num=len(feature_data))
        plt.plot(x_linspace, test_func(x_linspace, *params), label='Fitted quadratic polynomial')

def test_func(x, a2, a1, a0):
    return a2 * np.power(x, 2) + a1 * np.power(x, 1) + a0


def retrieve_data(image_root_path, mod=10):
    # processing ucsd pedestrian dataset
    sub_folder_index = 0
    image_count = 0
    images = []
    gt_count_in_images = []

    for sub_folder in image_root_path.glob('**/'):
        print(sub_folder.name.split('.')[0].split('_')[-1])
        if sub_folder_index == 0 or sub_folder.name.split('_')[0] != 'vidf1' or int(sub_folder.name.split('.')[0].split('_')[-1]) > 9:
            sub_folder_index += 1
            continue

        print(sub_folder.name)
        mat_path = annotation_root_path / (sub_folder.name.split('.')[0] + '_frame_full.mat')
        mat = read_mat(mat_path)

        for f in sub_folder.iterdir():
            if not f.is_file():
                continue

            frame_index = int(f.name[-7:-4]) - 1

            if image_count % mod == 0:
                img = cv2.imread(str(f), 0)
                images.append(img)
                gt_count_in_images.append(mat['frame'][0][frame_index][0][0][0].shape[0])

            image_count += 1

        sub_folder_index += 1

    return images, gt_count_in_images

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    background_image_path = '/home/osense-office/Documents/dataset/surveillance/ucsdpeds/background.png'
    background_image = cv2.imread(background_image_path, 0)
    image_root_dir = '/home/osense-office/Documents/dataset/surveillance/ucsdpeds/vidf'
    image_root_path = Path(image_root_dir)
    annotation_root_dir = '/home/osense-office/Documents/dataset/surveillance/ucsdpeds/vidf-cvpr'
    annotation_root_path = Path(annotation_root_dir)
    pmap = get_pmapxy('/home/osense-office/Documents/dataset/surveillance/ucsdpeds/vidf-cvpr/vidf1_33_dmap3.mat')

    images, gt_count_in_images = retrieve_data(image_root_path, mod=30)
    print(len(images))

    edited = get_abs_diff(images, background_image)
    blurred = get_foreground_mask(edited, threshold=25)
    seg_peri = get_seg_perimeter(blurred)
    # perspective_seg_size = get_seg_size(edited, pmapxy=pmap)

    plot_data(gt_count_in_images, seg_peri, test_func)
    plt.legend(loc='best')
    plt.title(label='segmentation perimeter against people count')
    plt.show()