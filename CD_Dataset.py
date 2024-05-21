import torch
import numpy as np
from torch import nn
from torchvision import transforms
from einops import rearrange
from torch.utils.data import Dataset


class CDDataset(Dataset):
    def __init__(self, im1, im2, pre_map, patch_size, isTrain):

        self.im1 = torch.from_numpy(im1).float() / 255.0
        self.im2 = torch.from_numpy(im2).float() / 255.0

        self.patch_size = patch_size
        self.isTrain = isTrain

        self.transform_ = transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip()
        ])

        # add dimension
        self.im1 = self.im1.unsqueeze(0)
        self.im2 = self.im2.unsqueeze(0)

        self.im1_channel = self.im1.shape[0]
        self.im2_channel = self.im2.shape[0]

        # Channel splicing
        im = torch.cat([self.im1, self.im2], dim=0).unsqueeze(0)

        # Image tiling
        if self.patch_size % 2 != 0:
            sub_net = nn.Sequential(
                      nn.ReflectionPad2d(self.patch_size // 2),
                      nn.Unfold(kernel_size=self.patch_size, stride=1),
            )
        else:
            sub_net = nn.Sequential(
                nn.ReflectionPad2d(
                    (
                        self.patch_size // 2,
                        self.patch_size // 2 - 1,
                        self.patch_size // 2,
                        self.patch_size // 2 - 1,
                    )
                ),
                nn.Unfold(kernel_size=self.patch_size, stride=1)
            )

        self.im_patches = sub_net(im)

        # Dimension conversion
        self.im_patches = rearrange(
            self.im_patches,
            "b (c p1 p2) n -> (b n) c p1 p2",
            p1=self.patch_size,
            c=(self.im1_channel + self.im2_channel),
        )

        # Select unchanged and changed positions
        self.pre_map = torch.from_numpy(pre_map).float()
        self.pre_map = self.pre_map.flatten()

        if self.isTrain:
            self.im_patches = self.im_patches[self.pre_map != 128]
            self.label = self.pre_map[self.pre_map != 128]/255

            np.random.seed(2024)
            random_array = np.random.choice(np.arange(0, len(self.label)),size=len(self.label), replace=False)
            self.im_patches = self.im_patches[random_array]
            self.label = self.label[random_array]


        else:
            self.im_patches = self.im_patches[self.pre_map == 128]
            self.label = self.pre_map[self.pre_map == 128]/256

    def __getitem__(self, index):
        cur_patch = self.im_patches[index]
        cur_label = self.label[index]
        if self.isTrain:
            cur_patch = self.transform_(cur_patch.unsqueeze(0))[0]

        return cur_patch[: self.im1_channel], cur_patch[self.im1_channel:], cur_label

    def __len__(self):
        return len(self.im_patches)
