import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.io
from scipy import optimize

from feature_func import *
from utils import *


def subtract_background(images, background):
    edited_image = []
    index = 0
    for image in images:
        difference = np.abs(image.astype(np.int32) - background.astype(np.int32))
        index += 1
        edited_image.append(difference.astype(np.uint8))
    return edited_image

def test_func(x, a2, a1, a0):
    return a2 * np.power(x, 2) + a1 * np.power(x, 1) + a0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    background_image_path = '/home/osense-office/Documents/dataset/surveillance/ucsdpeds/background.png'
    background_image = cv2.imread(background_image_path, 0)
    image_root_dir = '/home/osense-office/Documents/dataset/surveillance/ucsdpeds/vidf'
    image_root_path = Path(image_root_dir)
    annotation_root_dir = '/home/osense-office/Documents/dataset/surveillance/ucsdpeds/vidf-cvpr'
    annotation_root_path = Path(annotation_root_dir)
    pmap = get_pmapxy('/home/osense-office/Documents/dataset/surveillance/ucsdpeds/vidf-cvpr/vidf1_33_dmap3.mat')

    image_count = 0
    images = []
    gt_count_in_images = []
    mod = 20  # We don't want to get frames that are close in time

    # processing ucsd pedestrian dataset
    sub_folder_index = 0
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
            # print('frame: %s' % f)
            # print('frame no. : %d' % frame_index)

            if image_count % mod == 0:
                img = cv2.imread(str(f), 0)
                print(img.shape)
                images.append(img)
                gt_count_in_images.append(mat['frame'][0][frame_index][0][0][0].shape[0])
                """
                for position in mat['frame'][0][frame_index][0][0][0]:
                    cv2.drawMarker(img, position=(int(position[0]), int(position[1])), markerSize=2, color=(0))
                cv2.imshow("image", img)
                cv2.waitKey(0)
                """

            image_count += 1

        sub_folder_index += 1


    print(len(images))

    edited = subtract_background(images, background_image)
    seg_size = get_seg_size(edited)
    perspective_seg_size = get_seg_size(edited, pmapxy=pmap)
    params, params_covariance = optimize.curve_fit(test_func, seg_size, gt_count_in_images,)
    plt.scatter(seg_size, gt_count_in_images, label='raw data')
    plt.plot(seg_size, test_func(seg_size, *params), label='Fitted quadratic polynomial')
    print(params_covariance)
    print(params)
    plt.legend(loc='best')
    plt.errorbar()
    plt.title(label='Segmentation size against people count')
    plt.show()

