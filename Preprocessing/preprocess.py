import numpy as np
import torch
from PIL import Image
import cv2
from skimage.filters import threshold_local


def preprocess_image(image, function_names_and_arguments):
    if function_names_and_arguments is None:
        return image
    for (function_name, args) in function_names_and_arguments:
        if args is None:
            args = tuple()
        if function_name == 'maximize_contrast':
            image = maximize_contrast(image)
    return image


def maximize_contrast(image):
    if type(image) == str:
        image = np.array(Image.open(image).convert('RGB'))
    if type(image) == torch.Tensor:
        for ch in range(image.shape[0]):
            image[ch, :, :] = 255 * ((image[ch, :, :] - image[ch, :, :].min().item()) / max(image[ch, :, :].max().item() - image[ch, :, :].min().item(), 1))
    elif type(image) == np.ndarray:
        for ch in range(image.shape[2]):
            image[:, :, ch] = 255 * ((image[:, :, ch] - image[:, :, ch].min().item()) / max(image[:, :, ch].max().item() - image[:, :, ch].min().item(), 1))
    return image

