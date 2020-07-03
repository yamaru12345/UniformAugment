import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UniformAugment():
    def __init__(self, NumOps=2, fillcolor=(128, 128, 128)):
        self.NumOps = NumOps
        self.augs = {
                    'shearX': [-0.3, 0.3],
                    'shearY': [-0.3, 0.3],
                    'translateX': [-0.45, 0.45],
                    'translateY': [-0.45, 0.45],
                    'rotate': [-30, 30],
                    'autocontrast': [0, 0],
                    'invert': [0, 0],
                    'equalize': [0, 0],
                    'solarize': [0, 256],
                    'posterize': [4, 8],
                    'contrast': [0.1, 1.9],
                    'color': [0.1, 1.9],
                    'brightness': [0.1, 1.9],
                    'sharpness': [0.1, 1.9],
                    'cutout': [0, 0.2] 
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert('RGBA').rotate(magnitude)
            return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)

        def cutout(img, magnitude, fillcolor):
            img = img.copy()
            w, h = img.size
            v = w * magnitude
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)
            x0 = int(max(0, x0 - v / 2.))
            y0 = int(max(0, y0 - v / 2.))
            x1 = min(w, x0 + v)
            y1 = min(h, y0 + v)
            xy = (x0, y0, x1, y1)
            ImageDraw.Draw(img).rectangle(xy, fillcolor)
            return img

        self.func = {
                    'shearX': lambda img, magnitude: img.transform(
                        img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0), Image.BICUBIC, fillcolor=fillcolor),
                    'shearY': lambda img, magnitude: img.transform(
                        img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0), Image.BICUBIC, fillcolor=fillcolor),
                    'translateX': lambda img, magnitude: img.transform(
                        img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0), fillcolor=fillcolor),
                    'translateY': lambda img, magnitude: img.transform(
                        img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude), fillcolor=fillcolor),
                    'rotate': lambda img, magnitude: rotate_with_fill(img, magnitude),
                    'color': lambda img, magnitude: ImageEnhance.Color(img).enhance(magnitude),
                    'posterize': lambda img, magnitude: ImageOps.posterize(img, int(magnitude)),
                    'solarize': lambda img, magnitude: ImageOps.solarize(img, int(magnitude)),
                    'contrast': lambda img, magnitude: ImageEnhance.Contrast(img).enhance(magnitude),
                    'sharpness': lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(magnitude),
                    'brightness': lambda img, magnitude: ImageEnhance.Brightness(img).enhance(magnitude),
                    'autocontrast': lambda img, magnitude: ImageOps.autocontrast(img),
                    'equalize': lambda img, magnitude: ImageOps.equalize(img),
                    'invert': lambda img, magnitude: ImageOps.invert(img),
                    'cutout': lambda img, magnitude: cutout(img, magnitude, fillcolor=fillcolor)
        }

    def __call__(self, img):
        operations = random.sample(list(self.augs.items()), self.NumOps)
        for operation in operations:
            aug, range = operation
            magnitude = random.uniform(range[0], range[1])
            probability = random.random()
            if random.random() < probability:
                img = self.func[aug](img, magnitude)
        return img

    def transform_by_func(self, img, operation, magnitude):
        img = self.func[operation](img, magnitude)
        return img
  
  
class ImageTransform():
    def __init__(self, resize, mean, std, train=True):
        if train:
            self.data_transform = transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                UniformAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.data_transform = transforms.Compose([
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std) 
            ])

    def __call__(self, img):
        return self.data_transform(img=img)
