import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from core.utils import transforms as tf


class UCF101(Dataset):

    num_class = 21
    ignore_label = 255
    background_label = 0

    def __init__(self, config):
        super(UCF101, self).__init__()
        dataset_path = 'data/UCFI-101'
        with open(os.path.join(dataset_path, config.data_list + '.txt')) as f:
            self.img_list = []

            for line in f:
                video_dir = line.rstrip().split(' ')[0]
                frames_idx = int(line.rstrip().split(' ')[1])
                self.img_list.append((video_dir, frames_idx))

        self.img_path = dataset_path
        self.config = config

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        video_dir = self.img_list[idx][0]
        frame_idx = self.img_list[idx][1]

        images = []

        for i in range(3):
            img = cv2.imread(
                os.path.join(self.img_path, video_dir,
                             '{0:06d}.png'.format(frame_idx + i))).astype(
                                 np.float32)
            images.append(img)

        # flip
        if hasattr(self.config, 'flip') and self.config.flip:
            images = tf.group_random_flip(images)

        target_size = self.config.crop_size
        # resize
        images = tf.group_rescale(
            images, 0, [cv2.INTER_LINEAR for _ in range(3)], dsize=target_size)

        # # crop and pad
        # if self.config.crop_policy == 'random':
        #     images = tf.group_random_crop(images, target_size)
        #     images = tf.group_random_pad(
        #         images, target_size,
        #         [self.config.input_mean for _ in range(3)])
        # elif self.config.crop_policy == 'center':
        #     images = tf.group_center_crop(images, target_size)
        #     images = tf.group_concer_pad(
        #         images, target_size,
        #         [self.config.input_mean for _ in range(3)])
        # else:
        #     ValueError('Unknown crop policy: {}'.format(
        #         self.config.crop_policy))

        if hasattr(self.config, 'rotation') and random.random() < 0.5:
            images = tf.group_rotation(
                images, self.config.rotation,
                [cv2.INTER_LINEAR for _ in range(3)],
                [self.config.input_mean for _ in range(3)])
        # blur
        if hasattr(self.config,
                   'blur') and self.config.blur and random.random() < 0.5:
            images = tf.blur(images)

        # norm
        for i in range(3):
            images[i] = tf.normalize(images[i], self.config.input_mean,
                                     self.config.input_std)
            images[i] = torch.from_numpy(images[i]).permute(
                2, 0, 1).contiguous().float()

        return torch.cat([images[0], images[1]], dim=0), images[2]
