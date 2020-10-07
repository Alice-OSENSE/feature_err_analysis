import cv2
import numpy as np

def get_abs_diff(images, background):
    edited_image = []
    index = 0
    for image in images:
        difference = np.abs(image.astype(np.int32) - background.astype(np.int32))
        index += 1
        edited_image.append(difference.astype(np.uint8))
    return edited_image


def get_foreground_mask(images, threshold=23):
    blurred_images = []
    for image in images:
        blurred = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=3, sigmaY=3)
        _, thresh_image = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        blurred_images.append(thresh_image)
    return blurred_images