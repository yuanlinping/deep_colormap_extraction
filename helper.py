import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import logging

logger = logging.getLogger("in helper")
logger.setLevel(logging.DEBUG)
logger.disabled = True

def whatever_to_rgb(images, color_space):
    images_copy = images.clone().detach()
    images_copy = images_copy.cpu().numpy()
    image = images_copy[0, :, :, :]
    max_in_img = np.max(image)
    if(max_in_img > 1.50):
        image = image / 255.
    image = np.transpose(image, (1, 2, 0)) * 255.
    image = image.astype(np.uint8)
    if(color_space == "Lab"):
        image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)
    elif(color_space == "Hsl"):
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("RGB after re-converted:")
        logger.debug(np.unique(image[0],axis=0))
    image = np.transpose(image, (2,0,1))
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    return image

def whatever_to_rgb_numpy(images, color_space):
    images_copy = images.clone().detach()
    images_copy = images_copy.cpu().numpy()
    image = images_copy[0, :, :, :]
    image = np.transpose(image, (1, 2, 0)) * 255.
    image = image.astype(np.uint8)
    if(color_space == "Lab"):
        image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)
    return image


