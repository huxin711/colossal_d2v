import torch

from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic
from timm.data import create_transform

import os

from masking_generator import MaskingGenerator
from dataset_folder import ImageFolder

input_size=224
num_mask_patches=75
max_mask_patches_per_block=None
min_mask_patches_per_block=16
data_path='/data/huxin/xjtuhx/projects/oneyear/D2VDemo/datasets_dir/images/'
window_size=(14,14)

class DataAugmentationForBEiT(object):
    def __init__(self,):
        imagenet_default_mean_and_std = True
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD


        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=input_size)
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])


        self.masked_position_generator = MaskingGenerator(
            window_size, num_masking_patches=num_mask_patches,
            max_num_patches=max_mask_patches_per_block,
            min_num_patches=min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches = self.common_transform(image)
        return self.patch_transform(for_patches), self.masked_position_generator()
        
    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_beit_pretraining_dataset():
    transform = DataAugmentationForBEiT()
    print("Data Aug = %s" % str(transform))

    return ImageFolder(data_path, transform=transform)