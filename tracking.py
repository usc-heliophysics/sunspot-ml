import numpy as np
import pandas as pd
import cv2
from astropy.io import fits
import matplotlib.pyplot as plt
import skimage as ski

import sys
import os
from glob import glob
import math

def open_fits_image(image_path, image_data_header_location):
    with open(image_path, "rb") as image_file:
        hdu_list = fits.open(image_file)
        hdu_list.info()
        image_data = hdu_list[image_data_header_location].data
    return image_data


def fits_as_ubyte(fits_image):
    fits_image[fits_image < 0] = 0
    rescaled_image = ski.exposure.rescale_intensity(fits_image)
    fits_as_ubyte = ski.util.img_as_ubyte(rescaled_image)
    return fits_as_ubyte


regprop_path = '../output/20150508/regionprops/'
regprop_list = sorted(glob(regprop_path + '*.csv'))
fits_files = sorted(glob("/home/jswen/dev/solar-yolo/data/fits_images/20150508/*.fits"))

out_path = '../tracking/20150508/'
prev_cen_pts = []
tracking_objs = {}
track_id = 0
if len(regprop_list) == len(fits_files):
    for i in range(len(regprop_list)):
    # for i in range(0, 5):
        frame = cv2.cvtColor(fits_as_ubyte(open_fits_image(fits_files[i], 0)), cv2.COLOR_GRAY2BGR)
        rp_df = pd.read_csv(regprop_list[i])
        cur_cen_pts = [tuple(pts) for pts in rp_df[['Centroid X', 'Centroid Y']].values.tolist()]

        dists = []
        for pt in cur_cen_pts:
            for pt2 in prev_cen_pts:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                dists.append(distance)
                if distance < 20:
                    tracking_objs[track_id] = pt
                    track_id += 1
        if dists:
            print(min(dists))

        for object_id, pt in tracking_objs.items():
            cv2.circle(frame, pt, 3, (0, 0, 255), -1)
            # cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        frame = np.fliplr(frame)
        cv2.imwrite(out_path + f'frame{i+1}.png', frame)
        
        prev_cen_pts = cur_cen_pts.copy()
else:
    print("mismatch between images and regionprops")