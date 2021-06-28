#!/usr/bin/env python3

# __________________________________________ Libraries __________________________________________ #


# Learning
import tensorflow as tf

# Data
from configuration import *
from data import Forgery_Detection_Dataset, Augmentation

# Images
import matplotlib.pyplot as plt

# Time management
import time

# Other tools
from functools import partial

# ____________________________________________ Main ____________________________________________ #


if __name__ == "__main__":


    # 1. Dataset initialization

    ## Train dataset --------------------------------------------------------------------------
    print("Train dataset generation ...")
    train_generator = partial(Forgery_Detection_Dataset(augmentation=True).generate, mode='train')
    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=IMG_SHAPE[::-1]+(IMG_CHANNELS,), dtype=tf.float32),
            tf.TensorSpec(shape=IMG_SHAPE[::-1]+(IMG_CHANNELS,), dtype=tf.float32)
        )
    )
    train_dataset = train_dataset.shuffle(BATCH_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    ## Validation dataset ---------------------------------------------------------------------
    print("Validation dataset generation ...")
    val_generator = partial(Forgery_Detection_Dataset(augmentation=False).generate, mode='val')
    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=IMG_SHAPE[::-1]+(IMG_CHANNELS,), dtype=tf.float32),
            tf.TensorSpec(shape=IMG_SHAPE[::-1]+(IMG_CHANNELS,), dtype=tf.float32)
        )
    )
    val_dataset = val_dataset.batch(BATCH_SIZE)

    ## Example dataset ------------------------------------------------------------------------
    print("Example dataset generation ...")
    example_generator = partial(Forgery_Detection_Dataset(augmentation=False).generate, mode='example')
    example_dataset = tf.data.Dataset.from_generator(
        example_generator,
        output_signature=(
            tf.TensorSpec(shape=IMG_SHAPE[::-1]+(IMG_CHANNELS,), dtype=tf.float32),
            tf.TensorSpec(shape=IMG_SHAPE[::-1]+(IMG_CHANNELS,), dtype=tf.float32),
            tf.TensorSpec(shape=IMG_SHAPE[::-1]+(IMG_CHANNELS,), dtype=tf.float32)
        )
    )
    example_dataset = example_dataset.batch(BATCH_SIZE)


    # 2. Examples

    # Example pristine
    a = time.time()
    example_forgery, example_gt, example_pristine = iter(example_dataset).next()
    print(time.time() - a)
    plt.imshow(example_pristine[0])
    plt.show()
    plt.imshow(example_forgery[0])
    plt.show()
    plt.imshow(example_gt[0])
    plt.show()

    # Examples forgery 
    b = time.time()
    train_forgery, train_gt = iter(train_dataset).next()
    print(time.time() - b)
    val_forgery, val_gt = iter(val_dataset).next()
    plt.imshow(train_forgery[0])
    plt.show()
    plt.imshow(train_gt[0])
    plt.show()
    plt.imshow(train_forgery[-1])
    plt.show()
    plt.imshow(train_gt[-1])
    plt.show()
    plt.imshow(val_forgery[0])
    plt.show()
    plt.imshow(val_gt[0])
    plt.show()
    plt.imshow(val_forgery[-1])
    plt.show()
    plt.imshow(val_gt[-1])
    plt.show()