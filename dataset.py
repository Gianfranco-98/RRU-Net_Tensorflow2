#!/usr/bin/env python3


from skimage.util import random_noise
from configuration import *
import tensorflow as tf
import numpy as np
import random
import time
import cv2
import os


class Forgery_Detection_Dataset:

    def __init__(self, augmentation=False):
        
        self.name = DATASET_NAME

        # Dataset directory setting
        self.dataset_dir = os.path.join(DATA_DIR, self.name)
        self.pristine_dir = os.path.join(self.dataset_dir, PRISTINE_FOLDER)
        self.forgered_dir = os.path.join(self.dataset_dir, FORGERED_FOLDER)
        self.ground_truth_dir = os.path.join(self.dataset_dir, GROUND_TRUTH_FOLDER)

        # Dataset info setting
        self.files = os.listdir(self.pristine_dir)
        self.ids = [f[:-4] for f in self.files]
        train_indices = int(TRAIN_PERCENT * len(self.ids))
        val_indices = train_indices + int(VAL_PERCENT * len(self.ids))
        test_indices = val_indices + int(VAL_PERCENT * len(self.ids))
        self.train_ids = self.ids[:train_indices]
        self.val_ids = self.ids[train_indices:val_indices]
        self.test_ids = self.ids[val_indices:test_indices]

        # Dataset augmentation setting
        self.augmentation = Augmentation.random_overturn if augmentation else None

    def __len__(self):

        return len(self.ids)
        
    def get_image(self, img_id, category=None):

        if category == "pristine" or category is None:
            directory = self.pristine_dir
        elif category == "forgery":
            directory = self.forgered_dir
        elif category == "ground_truth":
            directory = self.ground_truth_dir
        else:
            raise ValueError("Wrong category type")

        filename = [f for f in self.files if img_id in f][0]
        image = cv2.cvtColor(cv2.imread(os.path.join(directory, filename)), cv2.COLOR_BGR2RGB)
        return image
    
    def generate(self, mode):

        if mode == "train":
            ids = self.train_ids
        elif mode == "val":
            ids = self.val_ids
        elif mode == "test":
            ids = self.test_ids
        elif mode == "example":
            ids = self.ids

        for img_id in ids:

            # Get images
            if mode == "example":
                pristine_img = self.get_image(img_id, "pristine") / 255.
                pristine_img = cv2.resize(pristine_img, IMG_SHAPE) 
            forgery_img = self.get_image(img_id, "forgery") / 255.
            gt_img = self.get_image(img_id, "ground_truth") / 255.

            # Resize
            if mode != "test":
                forgery_img = cv2.resize(forgery_img, IMG_SHAPE)
                gt_img = cv2.resize(gt_img, IMG_SHAPE)

            # Augment
            if self.augmentation is not None:
                forgery_img, gt_img = self.augmentation(forgery_img, gt_img)

            if mode == "example":
                yield forgery_img, gt_img, pristine_img
            else:
                yield forgery_img, gt_img


class Augmentation:

    def __init__(self, compression_quality=COMPRESSION_QUALITY):
        self.compression_quality = compression_quality

    @staticmethod
    def gaussian_noise(image, variance=NOISE_VARIANCE):
        return random_noise(np.array(image), mode='gaussian', mean=0, var=variance, clip=True)       #TODO: replace with tf.random.normal function

    # TODO
    def jpeg_compression(self, image):
        pass                            

    @staticmethod
    def random_overturn(img, gt_img):
        cointoss1 = random.choice(range(10))
        cointoss2 = random.choice(range(10))
        if cointoss1 >= 5:
            if cointoss2 >= 5:
                img = tf.image.flip_left_right(img)
                gt_img = tf.image.flip_left_right(gt_img)
            else:
                img = tf.image.flip_up_down(img)
                gt_img = tf.image.flip_up_down(gt_img)
        
        return img, gt_img

    def complete_augmentation(self, image):
        image = self.random_overturn(self.gaussian_noise(image))
        return image        
