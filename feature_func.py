import cv2
import numpy as np

def get_seg_size(images, threshold=23, pmapxy=None):
    seg_sizes = []
    index = 0

    for image in images:
        blurred = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=3, sigmaY=3)
        ret, thresh_image = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        """
       if index % 10 == 0:
            cv2.imshow("thres", thresh_image)
            cv2.waitKey(0)

        """

        if pmapxy is not None:
            thresh_image.multiply(pmapxy)

        seg_sizes.append(cv2.countNonZero(thresh_image))
        index += 1

    return seg_sizes


def get_seg_parameter(images):
    seg_perimeters = []
    for image in images:
        edge_image = cv2.Canny(image, threshold1=30, threshold2=40)
        seg_perimeters.append(cv2.countNonZero(edge_image))

    return seg_perimeters