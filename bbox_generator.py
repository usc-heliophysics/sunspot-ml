import os
import pandas as pd
import numpy as np
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from astropy.io import fits
from PIL import Image
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


def generate_sunspot_annotations(image, structuring_element=disk(2), cutoff_area=9):
    # Threshold the image, excluding zero values
    thresh = threshold_otsu(image[image > 0])
    bw = closing(image < thresh, structuring_element)

    # Remove artifacts connected to the image border
    cleared = clear_border(bw)

    # Label image regions
    label_image = label(cleared)

    annotations = []

    for region in regionprops(label_image, intensity_image=image):
        # Filter out small regions
        if region.area >= cutoff_area:
            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr
            x_center = minc + width / 2
            y_center = minr + height / 2

            # Calculate detailed metrics
            x_centroid, y_centroid = region.centroid
            area = region.area
            centroid_intensity = image[int(y_centroid), int(x_centroid)]
            average_intensity = region.mean_intensity
            min_intensity = region.min_intensity
            max_intensity = region.max_intensity

            annotations.append([
                x_center, y_center, width, height, minc, minr,
                x_centroid, y_centroid, area, centroid_intensity,
                average_intensity, min_intensity, max_intensity
            ])

    # Convert annotations to DataFrame
    annotations_df = pd.DataFrame(annotations, columns=[
        'x_center', 'y_center', 'width', 'height', 'x', 'y',
        'x_centroid', 'y_centroid', 'area', 'centroid_intensity',
        'average_intensity', 'min_intensity', 'max_intensity'
    ])

    return annotations_df


def main():
    img_name = './fits-images/20150508/hmi.in_45s.20150508_000000_TAI.2.continuum.fits'
    base_image = open_fits_image(img_name, 0)

    crops_dict = {
        'x': [0, 1881, 3173, 3443, 1952, 2816],
        'y': [0, 1489, 1247, 2099, 2357, 1376],
        'width': [4096, 640, 640, 640, 320, 1280],
        'height': [4096, 640, 640, 640, 320, 1280],
    }
    crops_df = pd.DataFrame(crops_dict)

    out_dir = './training_crops/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    timestamp = img_name.split('.')[-4].replace('TAI', '')

    all_annotations = []  # List to store annotations from all crops

    for i, crop_info in crops_df.iterrows():
        crop = crop_image(crop_info, base_image)
        annotations_df = generate_sunspot_annotations(crop, disk(2), 9)
        annotations_df['crop_index'] = i  # Add crop index to distinguish between different crops

        all_annotations.append(annotations_df)

        fname = f"{timestamp}CROP_{i}"
        # Save cropped image
        cropped_img = Image.fromarray(crop.astype(np.uint8))
        cropped_img.save(os.path.join(out_dir, f"{fname}.png"))

    # Combine all annotations into a single DataFrame
    combined_annotations_df = pd.concat(all_annotations, ignore_index=True)
    # Save combined annotations to a single CSV file
    combined_annotations_df.to_csv(os.path.join(out_dir, f"{timestamp}_all_annotations.csv"), index=False)

    print(f"All annotations saved to '{timestamp}_all_annotations.csv'.")


if __name__ == '__main__':
    main()
