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

    def __init__(self, ids):
        self.name = DATASET_NAME
        self.ids = ids

        # Dataset directory setting
        self.dataset_dir = os.path.join(DATA_DIR, self.name)
        self.pristine_dir = os.path.join(self.dataset_dir, PRISTINE_FOLDER)
        self.forgered_dir = os.path.join(self.dataset_dir, FORGERED_FOLDER)
        self.ground_truth_dir = os.path.join(self.dataset_dir, GROUND_TRUTH_FOLDER)

    def __len__(self):
        return len(self.ids)
        
    def get_image(self, img_id, category=None):
        suffix = '.jpg'
        if category == "pristine" or category is None:
            directory = self.pristine_dir
        elif category == "forgery":
            directory = self.forgered_dir
        elif category == "ground_truth":
            directory = self.ground_truth_dir
            suffix = '_gt.png'
        else:
            raise ValueError("Wrong category type")

        filename = img_id + suffix
        image = cv2.cvtColor(cv2.imread(os.path.join(directory, filename)), cv2.COLOR_BGR2RGB)
        return image
    
    def generate(self, mode):
        
        for img_id in self.ids:

            # Get images
            if mode == "example":
                pristine_img = self.get_image(img_id, "pristine") / 255.
                pristine_img = cv2.resize(pristine_img, IMG_SHAPE)
            forgery_img = self.get_image(img_id, "forgery") / 255.
            gt_img = self.get_image(img_id, "ground_truth") / 255.
            if self.name == 'Spliced_COCO':
                gt_img = 1 - gt_img

            # Resize
            forgery_img = cv2.resize(forgery_img, IMG_SHAPE)
            gt_img = cv2.resize(gt_img, IMG_SHAPE)

            if mode == "example":
                yield forgery_img, gt_img, pristine_img
            else:
                yield forgery_img, gt_img
