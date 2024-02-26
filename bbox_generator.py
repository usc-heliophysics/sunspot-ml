import os
import sys
import pandas as pd
import numpy as np
from PIL import Image

from astropy.io import fits
from skimage.io import imread

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square, disk, diamond, star
from skimage.color import label2rgb

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def open_fits_image(image_path, image_data_header_location):
    print(f"Reading {image_path}")
    image_file = open(image_path, "rb")
    hdu_list = fits.open(image_file)
    hdu_list.info()
    image_data = hdu_list[image_data_header_location].data
    return image_data


def crop_image(sunspot, image):
    offset_x = int(sunspot['x'])
    offset_y = int(sunspot['y'])
    width = int(sunspot["width"])
    height = int(sunspot["height"])

    # Crop image
    cropped_image = image[offset_y:offset_y + height, offset_x:offset_x + width]
    return cropped_image


def generate_YOLO_annotations(image, structuring_element=disk(2), cutoff_area=9):
    """
    Uses an automatic threshold to segment input image and draw
    bounding boxes around isolated regions. Outputs in YOLO bounding
    box annotation files (.txt) where each row has the following:

    class x_center y_center width height

    separated by spaces.
    """
    # apply Otsu's method for thresholding
    # exclude zero values (they lie off the disk)
    thresh = threshold_otsu(image[image != 0])
    bw = closing(image < thresh, structuring_element)

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)

    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image, cmap='gray')

    class_list = []  # classes: 0: pore; 1: sunspot
    x_center_list = []  # coordinates normalized to image size (1,1)
    y_center_list = []
    width_list = []
    height_list = []

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= cutoff_area:
            minr, minc, maxr, maxc = region.bbox

            if region.area <= 60:
                class_list.append(0)  # mark as pore
                # draw rectangle around segmented pore
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)
            else:
                class_list.append(1)  # mark as sunspot
                # draw rectangle around segmented sunspot
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='purple', linewidth=1)
                ax.add_patch(rect)

            # calculate bbox measurements
            width = maxc - minc
            height = maxr - minr
            x_center = minc + (width) / 2
            y_center = minr + (height) / 2

            # normalize to image shape
            image_height, image_width = image.shape
            width = width / image_width
            height = height / image_height
            x_center = x_center / image_width
            y_center = y_center / image_height

            x_center_list.append(x_center)
            y_center_list.append(y_center)
            width_list.append(width)
            height_list.append(height)

    annotations_dict = {
        'class': class_list,
        'x_center': x_center_list,
        'y_center': y_center_list,
        'width': width_list,
        'height': height_list,
    }
    annotations_df = pd.DataFrame(annotations_dict)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return annotations_df


def main():
    img_name = './fits-images/20150508/hmi.in_45s.20150508_000000_TAI.2.continuum.fits'
    base_image = open_fits_image(img_name, 0)

    crops_dict = {
        'x': [1220, 1881, 3173, 3443, 1952],
        'y': [2161, 1489, 1247, 2099, 2357],
        'width': [640, 640, 640, 640, 320],
        'height': [640, 640, 640, 640, 320],
    }
    crops_df = pd.DataFrame(crops_dict)

    out_dir = './training_crops/'
    timestamp = img_name.split('.')[-4].replace('TAI', '')

    crops_df.to_csv(out_dir + timestamp + 'crop_positions.csv', index=False)

    for i in range(len(crops_df)):
        crop = crop_image(crops_df.iloc[i], base_image)
        annots = generate_YOLO_annotations(crop, disk(2), 9)

        fname = timestamp + f'CROP_{i}'

        cropped_img = Image.fromarray(crop)
        cropped_img.save(out_dir + fname + '.png')
        with open(out_dir + fname + '.txt', 'a') as f:
            annotations_as_string = annots.to_string(header=False, index=False)
            f.write(annotations_as_string)


if __name__ == '__main__':
    main()