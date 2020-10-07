import cv2
import numpy as np


def get_seg_size(images, threshold=23, pmapxy=None):
    seg_sizes = []
    index = 0

    for image in images:
        """
       if index % 10 == 0:
            cv2.imshow("thres", thresh_image)
            cv2.waitKey(0)

        """
        index += 1

        if pmapxy is not None:
            thresh_image = np.multiply(image, pmapxy)
            seg_sizes.append(thresh_image.sum(1).sum(0))
            continue
        seg_sizes.append(cv2.countNonZero(thresh_image))
    return seg_sizes


def get_seg_perimeter(images):
    seg_perimeters = []
    index = 0

    for image in images:
        edge_image = cv2.Canny(image, threshold1=30, threshold2=40)
        """
        if index % 10 == 0:
            cv2.imshow("edge", edge_image)
            cv2.waitKey(0)
            cv2.imshow("image", image)
            cv2.waitKey(0)
        """

        seg_perimeters.append(cv2.countNonZero(edge_image))
        index += 1

    return seg_perimeters