#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename : prepare_dataset.py
# @Date : 2019-11-25
# @Author : Wufei Ma

import os
import sys
import time

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

classes = ['DUM1178', 'DUM1154', 'DUM1297', 'DUM1144', 'DUM1150',
           'DUM1160', 'DUM1180', 'DUM1303', 'DUM1142', 'DUM1148', 'DUM1162']
data_csv = 'data-sep-30.csv'
original_dir = os.path.join('data')
clean_dir = os.path.join('ms', 'data')

save_size = (768, 768)


def crop_image(img):
    if img.shape[0] == 2048 and img.shape[1] == 2560:
        return img[:1920, :]
    elif img.shape[0] == 1428 and img.shape[1] == 2048:
        return img[:1408, :]
    elif img.shape[0] == 1024 and img.shape[1] == 1280:
        return img[:960, :]
    elif img.shape[0] == 1448 and img.shape[1] == 2048:
        return img[:1428, :]
    else:
        raise Exception('Unknown image size: {}'.format(img.shape))


def augment_image(img):
    """
    Data augmentation for an input image. List of augmentation techniques:

    Active:
    - image cropping to square images
    - rotation: 180

    Inactive:
    - rotation: 90, 270
    - noise
    - flip: ud, lr, up_lr

    :param img: given image with info bar cropped if the original image has one
    :return: a list of augmented images
    """
    if img.shape[0] == 1920 and img.shape[1] == 2560:
        img1 = cv2.resize(img[:, :1920], (1024, 1024))
        img2 = cv2.resize(img[:, 160:2080], (1024, 1024))
        img3 = cv2.resize(img[:, 320:2240], (1024, 1024))
        img4 = cv2.resize(img[:, 480:2400], (1024, 1024))
        img5 = cv2.resize(img[:, 640:2560], (1024, 1024))
    elif img.shape[0] == 1408 and img.shape[1] == 2048:
        img1 = cv2.resize(img[:, :1408], (1024, 1024))
        img2 = cv2.resize(img[:, 160:1568], (1024, 1024))
        img3 = cv2.resize(img[:, 320:1728], (1024, 1024))
        img4 = cv2.resize(img[:, 480:1888], (1024, 1024))
        img5 = cv2.resize(img[:, 640:], (1024, 1024))
    elif img.shape[0] == 960 and img.shape[1] == 1280:
        img1 = cv2.resize(img[:, :960], (1024, 1024))
        img2 = cv2.resize(img[:, 80:1040], (1024, 1024))
        img3 = cv2.resize(img[:, 160:1120], (1024, 1024))
        img4 = cv2.resize(img[:, 240:1200], (1024, 1024))
        img5 = cv2.resize(img[:, 320:], (1024, 1024))
    elif img.shape[0] == 1428 and img.shape[1] == 2048:
        img1 = cv2.resize(img[:, :1428], (1024, 1024))
        img2 = cv2.resize(img[:, 155:1583], (1024, 1024))
        img3 = cv2.resize(img[:, 310:1738], (1024, 1024))
        img4 = cv2.resize(img[:, 465:1893], (1024, 1024))
        img5 = cv2.resize(img[:, 620:], (1024, 1024))
    else:
        raise Exception('Unknown image size: {}'.format(img.shape))
    return [img1, img2, img3, img4, img5,
            np.rot90(img1, 2), np.rot90(img2, 2),
            np.rot90(img3, 2), np.rot90(img4, 2),
            np.rot90(img5, 2)]


if __name__ == '__main__':

    # About.
    print('Collect images from {:s}.'.format(original_dir))
    print('Save images to {:s}.'.format(clean_dir))

    # Set up directory.
    os.makedirs(clean_dir, exist_ok=True)
    '''
    for c in classes:
        os.makedirs(os.path.join(clean_dir, c), exist_ok=True)
    '''

    # Create clean dataset.
    for c in classes:
        rows = pd.read_csv(data_csv, index_col=0)
        rows = rows.loc[rows['met_id'] == c]
        rows = rows.loc[rows['img_proc'].isin(['LBE', 'LABE'])]
        rows = rows.loc[rows['img_size'].isin([
            '2048x2560',
            '1428x2048',
            '1024x1280',
            '1448x2048'
        ])]
        rows = rows.loc[rows['scale'].isin(['250X', '500X'])]
        img_names = np.asarray(rows['filename'])
        img_names = [os.path.join(original_dir, c, x)
                     for x in img_names]
        print('Found {:d} images from class {:s}.'.format(len(img_names), c))
        for name in img_names:
            img = cv2.imread(name)
            if img is None:
                raise Exception('Error: Fail to read image {:s}.'
                                .format(name))
            img = crop_image(img)

            augmented_images = augment_image(img)
            base_name = os.path.basename(name).strip().split('.')[0]
            extension = os.path.basename(name).strip().split('.')[1]
            for i, im in enumerate(augmented_images):
                save_name = base_name + '_{:d}.'.format(i) + extension
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # im = cv2.resize(im, save_size)

                # Depend on the structure of the dataset dir.
                # cv2.imwrite(os.path.join(clean_dir, c, save_name), im)
                cv2.imwrite(os.path.join(clean_dir, save_name), im)
