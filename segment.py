"""
Core methods used for segmenting sunspots from flattened continuum intensity images of the sun.

Authors: James Wen (jswen@usc.edu) and Sid Qian (sidqian@usc.edu)
"""


import os
import subprocess
import numpy as np
import cv2
import skimage as ski
from scipy import ndimage as ndi
from PIL.Image import Image
from astropy.io import fits

Image.MAX_IMAGE_PIXELS = None


def open_fits_image(image_path, image_data_header_location):
    """
    reads a FITS file as an image array
    """
    print(f"reading {image_path}...")
    with fits.open(image_path) as hdu_list:
        image_data = hdu_list[image_data_header_location].data
    # replace negative values with zero
    image_data[image_data < 0] = 0
    return image_data


def preprocess(image):
    """
    prepare a copy of image for segmentation
    """
    # binarize image and find bounds of non-black area
    binary = image.copy().astype("int32")
    binary[binary < 0] = 0
    binary[binary > 0] = 1
    white = np.where(binary == 1)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    crop = image.copy()[ymin:ymax, xmin:xmax]
    
    # replace off-disk pixels with mean value of a central tile
    nrows, ncols = crop.shape
    row, col = np.ogrid[:nrows, :ncols]
    cnt_row, cnt_col = nrows // 2, ncols // 2
    central_tile = crop.copy()
    ratio = 40 # ratio of full img width to tile width
    central_tile = central_tile[cnt_row-nrows//ratio:cnt_row+nrows//ratio, cnt_col-ncols//ratio:cnt_col+ncols//ratio]
    mean, std = np.mean(central_tile), np.std(central_tile)
    # mask out pixels that lie outside of a disk with radius 99% of img size
    outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > (nrows / 2 * 0.995)**2)
    crop[outer_disk_mask] = mean
    # clip overly bright values and set them to the mean value
    crop[crop > mean + 3*std] = mean

    # place cropped image back into original and fill edges with mean values
    result = np.full_like(image, mean)
    result[ymin:ymax, xmin:xmax] = crop
    return result


# currently unused.
def postprocess(binary_image):
    """
    remove small objects and fill holes in binary segmentation image. 
    """
    processed_img = binary_image.copy()
    # structure = np.ones((2, 2))
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])
    opened = ndi.binary_opening(processed_img, structure=structure)
    closed = ndi.binary_closing(opened, structure=structure)
    filled = ndi.binary_fill_holes(closed)
    return filled


def find_rois(image, num_stdevs=7, padding=50, min_count=4):
    """
    Finds the darkest pixels in an image and isolates the surrounding region of interest. 

    The dark pixels are chosen by the number of standard deviations they are below 
    the global mean intensity. A square region whose width is determined by the padding
    parameter is then drawn around each chosen pixel.

    NOTE: in general, more padding works better for segmenting images with highly variable backgrounds, 
    as more objects will be grouped in the same region of interest and the background is "smoothed out".

    Parameters
    ----------
    image: ndarray
        The input image.

    num_stdevs: int, optional
        The number of standard deviations below the mean intensity. Pixels darker than this
        will be used to locate regions of interest.

    padding: int, optional
        The amount of padding. This should be chosen such that the objects to be segmented 
        are not touching the edges of the region of interest.

    Returns
    -------
    regions: ndarray
        A copy of the input image, but with every value outside the regions of interest 
        set to zero. 
    """
    image = ski.util.img_as_float(image)
    vals = image.flatten()
    mean = np.mean(vals)
    std = np.std(vals)
    markers = np.zeros_like(image)
    regions = np.zeros_like(image)
    min_idxs = np.argwhere(image < mean - num_stdevs*std)
    for idx in min_idxs:
        # plot min idxs as markers
        markers[idx[0], idx[1]] = image[idx[0], idx[1]]
        # expand by padding a box surrounding the marker
        box = [idx[0]-padding, idx[1]-padding, idx[0]+padding-1, idx[1]+padding-1]
        # check if there's enough dark pixels in the box
        region_of_interest = image[box[0]:box[2], box[1]:box[3]]
        if (region_of_interest < mean - num_stdevs*std).sum() > min_count:
            regions[box[0]:box[2], box[1]:box[3]] = region_of_interest
    return regions


# apply K-means clustering with an optional gaussian blur
def kmeans(image, K, blur_strength=1, return_info=False):
    """
    Apply K-means clustering with an optional gaussian blur.

    Uses OpenCV's implementation of K-means to bin intensity values in an image.
    Noisy images may benefit from the optional Gaussian blur.

    NOTE: K-means clustering is an unsupervised learning algorithm, so its results
    may not be exactly reproducible. The `cv2.KMEANS_PP_CENTERS` flag should mitigate
    this and is enabled by default. Refer to OpenCV documentation for more info.

    Parameters
    ----------
    image: ndarray
        The input image.

    K: int
        The number of bins (clusters).

    blur_strength: int, optional
        The width of the Gaussian kernel used for blurring. Default: 1 (no blur applied)

    return_info: bool, optional
        Whether to return additional outputs from the clustering algorithm. Default: False

    Returns
    -------
    compactness: float
        A measure of how closely the member values of each cluster match its central value. 
        The K-means algorithm seeks to maximize this value. Only returned if `return_info` is True.

    labels: ndarray
        An array of labels for each value in the input image. The labels are indices for 
        the array of cluster centers. Only returned if `return_info` is True.

    centers: ndarray
        An array of cluster centers, where each element represents the cluster's central value. 
        Only returned if `return_info` is True.

    result: ndarray
        The resulting image, where each pixel is replaced by the central value of the cluster it
        belongs to. 
    """
    if blur_strength > 1:
        image = cv2.GaussianBlur(np.float32(image), (blur_strength,blur_strength), 0)
    else:
        image = np.float32(image)
    Z = image.reshape(-1)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
    flags = cv2.KMEANS_PP_CENTERS
    
    # apply K-means algorithm
    compactness, labels, centers, = cv2.kmeans(Z, K, None, criteria, 10, flags)

    # paint each pixel in array with its labeled center value
    painted_array = centers[labels.flatten()]
    result = painted_array.reshape(image.shape)
    
    if return_info:
        return compactness, labels, centers, result
    else:
        return result


def clear_bg(clustered_img, bwidth=20, return_vals=False):
    """
    Clears background of a k-clustered image, assuming border pixels are not objects of interest.

    The most abundant pixels in a border of certain width are considered background and set to zero.
    The objects of interest are assumed to be centered, substantially brighter/darker than the 
    background, and sufficiently padded by background pixels. 
    
    A pixel is considered background if it is within 3 standard deviations of the mean value of the 
    pixels that lie within the border.

    Parameters
    ----------
    clustered_img: ndarray
        The image to be cleared. Assumed to be processed with K-means clustering.

    bwidth: int, optional
        Width of the border in pixels. The shape of the border will be rectangular. Default: 20

    return_vals: bool, optional
        Whether to return the values and number of occurences of pixels found in the border.

    Returns
    -------
    cleared: ndarray
        The image with background removed.

    border_vals: ndarray
        1d array of pixel values that lie within the border.

    background_vals: ndarray
        1d array of each unique value that was selected as background.

    """
    R = clustered_img.flatten()
    # set highest value to zero. this is to deal with plages surrounding sunspots
    R[R == R.max()] = 0
    nrows, ncols = clustered_img.shape
    border = np.copy(clustered_img)
    border[bwidth:nrows-bwidth, bwidth:ncols-bwidth] = 0
    border_vals = border[np.nonzero(border)]
    unique_vals = np.unique(border_vals)
    mean = border_vals.mean()
    std = border_vals.std()
    upper_bound = mean + 3*std
    lower_bound = mean - 3*std
    background_vals = unique_vals[np.logical_and(unique_vals > lower_bound, unique_vals < upper_bound)]
    R[np.argwhere(np.isin(R, background_vals))] = 0
    cleared = R.reshape(clustered_img.shape)
    if return_vals:
        return cleared, border_vals, background_vals
    else:
        return cleared


def binarize_features(cleared_img, feature='penumbrae'):
    """
    Binarizes the cleared, k-clustered image with either umbrae or penumbrae as the target.

    If "penumbrae" is chosen, sets the lightest pixels to 1 and all others to 0.

    If "umbrae" is chosen, sets the darkest nonzero pixels to 1 and all others to 0.

    Parameters
    ----------
    cleared_img: ndarray
        The image to binarize. the background pixels should be 0.

    feature: str, optional
        The target feature, either penumbrae or umbrae. 
        Umbrae are assumed to be the darker of the two.

    Returns
    -------
    binarized: ndarray
        The binarized image. 
    """
    labeled_regions, num_regions = ndi.label(cleared_img)
    locs = ndi.find_objects(labeled_regions)
    binarized = np.copy(cleared_img)
    
    if feature == "penumbrae":
        for loc in locs:
            spot = np.copy(cleared_img[loc])
            if len(spot[np.nonzero(spot)]) < 4:
                binarized[loc] = 0
            else:
                spot[spot > 0] = 1
                binarized[loc] = spot
    elif feature == "umbrae":
        for loc in locs:
            spot = np.copy(cleared_img[loc])
            spot[spot == np.max(spot[np.nonzero(spot)])] = 0
            if spot.any():
                spot[spot > np.min(spot[np.nonzero(spot)])] = 0
                spot[spot == np.min(spot[np.nonzero(spot)])] = 1
            binarized[loc] = spot
    else:
        raise Exception("Detection of this feature is not supported.")
    return ndi.binary_fill_holes(binarized)


def segment_core(fits_path, image_path=None, output_path=None, feature="penumbrae", findroi_kwargs=None, kmeans_kwargs=None, clearbg_kwargs=None):
    # check if file exists
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file {fits_path} not found.")
    if image_path and not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")
       
    #extract the date from the fits file path
    date, time = fits_path.split('/')[-1].split('.')[2].split('_')[:2]

    # make output directories
    if not output_path:
        outpath = f'output/{date}/'
    else:
        outpath = output_path + f'/output/{date}/'
    rp_path = outpath + 'regionprops/'
    label_path = outpath + 'labeled/'
    seg_path = outpath + 'segmented/'
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    if not os.path.exists(rp_path):
        os.makedirs(rp_path, exist_ok=True)
    if not os.path.exists(label_path):
        os.makedirs(label_path, exist_ok=True)
    if not os.path.exists(seg_path):
        os.makedirs(seg_path, exist_ok=True)

    # set default keyword args for find_roi, kmeans, and clear_bg
    if not findroi_kwargs:
        findroi_kwargs = {"num_stdevs": 7, "padding": 50, "min_count": 4}
    if not kmeans_kwargs:
        kmeans_kwargs = {"K": 5, "blur_strength": 1}
    if not clearbg_kwargs:
        clearbg_kwargs = {"bwidth": 20}

    # read fits image and preprocess
    fits_image = open_fits_image(fits_path, 0)
    prepped_fits = preprocess(fits_image)

    if image_path:
        image = ski.io.imread(image_path, as_gray=True)
        prepped_image = preprocess(image)
    else:
        prepped_image = prepped_fits

    # adjust parameters below
    regions = find_rois(prepped_fits, **findroi_kwargs)
    labeled_regions, num_regions = ndi.label(regions)

    locs = ndi.find_objects(labeled_regions)
    segmentation = np.zeros_like(fits_image)
    for loc in locs:
        img = prepped_image[loc]
        clustered = kmeans(img, **kmeans_kwargs)
        cleared = clear_bg(clustered, **clearbg_kwargs)
        binarized = binarize_features(cleared, feature=feature)
        segmentation[loc] = binarized

    # segmentation = postprocess(segmentation)

    # label image
    label_image = ski.measure.label(segmentation)

    # GET FOLLOWING REGION PROPERTIES FROM THESE LABELS
    '''bounding box, area, centroid coords, centroid pixel, intensity, minimum intensity, coords of min intensity 
    pixel, average intensity'''

    props = ski.measure.regionprops(label_image, intensity_image=fits_image)
    regionprop_path = rp_path + f'region_properties_{date}_{time}_.csv'
    with open(regionprop_path, 'w') as file:
        file.write(
            "Region Label,BBox Min Y,BBox Min X,BBox Max Y,BBox Max X,Area,Centroid Y,Centroid X,"
            "Centroid Intensity,Min Intensity,Min Intensity Y,Min Intensity X,Avg Intensity\n"
        )
        for region in props:
            bbox = tuple([int(b) for b in region.bbox])
            centroid = tuple([int(c) for c in region.centroid])
            area = int(region.area) 

            # Get intensity data directly from the fits_image using adjusted coordinates
            centroid_intensity = fits_image[centroid]

            # To find the minimum intensity and its coordinates in the fits_image using the region's bbox
            x_min, y_min, x_max, y_max = bbox
            region_slice = fits_image[x_min:x_max + 1, y_min:y_max + 1]

            min_intensity = np.min(region_slice)
            min_intensity_index = np.argmin(region_slice)
            min_intensity_coords = np.unravel_index(min_intensity_index, region_slice.shape)

            # Convert local coordinates of the minimum to global coordinates
            min_intensity_coords = (min_intensity_coords[0] + x_min, min_intensity_coords[1] + y_min)

            mean_intensity = int(np.mean(region_slice))

            # Format the data as a string to write to the file
            region_data = (
                f"{region.label}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {area}, {centroid[0]}, {centroid[1]}, "
                f"{centroid_intensity}, {min_intensity}, {min_intensity_coords[0]}, {min_intensity_coords[1]}, "
                f"{mean_intensity}\n"
            )
            file.write(region_data)

    # rescale image intensity to 8-bit dtype so it can be overlaid with labels
    rescaled_image = ski.exposure.rescale_intensity(fits_image)
    base_as_ubyte = ski.util.img_as_ubyte(rescaled_image)
    image_label_overlay = ski.color.label2rgb(label_image, image=base_as_ubyte)


    # output images from each step
    flipped_segment_image = np.fliplr(segmentation)  # move origin to upper right to match FITS image
    seg_img_path = seg_path + f'segmented_{date}_{time}_.png'
    ski.io.imsave(seg_img_path, ski.util.img_as_ubyte(flipped_segment_image.astype('bool')), check_contrast=False)

    flipped_label_image = np.fliplr(image_label_overlay)    # move origin to upper right to match FITS image
    label_img_path = label_path + f'labeled_{date}_{time}_.png'
    ski.io.imsave(label_img_path, ski.util.img_as_ubyte(flipped_label_image), check_contrast=False)

    # remove upscaled image
    # os.remove(base_image_path)

    return (
        f"Found {label_image.max()} sunspot {feature} in {num_regions} regions of interest.\n" + 
        f"Labeled image saved as {label_img_path}\n" + 
        f"Region properties saved as {regionprop_path}\n" + 
        f"Segmented image saved as {seg_img_path}\n" + 
        "Done!\n"
    )


if __name__ == '__main__':
    run = segment_core("test_res/hmi.in_45s.20150508_000000_TAI.2.continuum.fits", feature='penumbrae')
    print(run)
