import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import cv2
import numpy as np
from PIL import Image


class HighPass:
    def __init__(self):
        pass

    def __call__(self, img):
        img_np = np.array(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        high = cv2.subtract(gray, blur)

        high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)
        high = cv2.cvtColor(high, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(high.astype(np.uint8))


def get_ssl_transforms(image_size=224):
    base_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(3),
        T.ToTensor(),
    ])

    highpass_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        HighPass(),
        T.ToTensor(),
    ])

    return base_aug, highpass_aug