import os
import subprocess
import numpy as np
import skimage as ski
import cv2
from PIL.Image import Image
from astropy.io import fits
from skimage import measure

Image.MAX_IMAGE_PIXELS = None


def open_fits_image(image_path, image_data_header_location):
    print(f"Reading {image_path}")
    image_file = open(image_path, "rb")
    hdu_list = fits.open(image_file)
    hdu_list.info()
    image_data = hdu_list[image_data_header_location].data
    return image_data


def reshape_split(image: np.ndarray, kernel_size: int):
    img_height, img_width = image.shape
    tile_height = tile_width = kernel_size  # tile must be square

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array


def read_image(img_name):
    print("Attempting to read image with skimage...")
    if img_name.split('.')[-1] == 'fits':
        return open_fits_image(img_name, 0)
    else:
        return cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

# Upscale image using Real-ESRGAN
# Default scale factor is 4
def upscale_image(input_path, output_path, scale_factor=4):
    command = [
        './realesrgan-ncnn-vulkan',
        '-i', input_path,
        '-o', output_path,
        '-s', str(scale_factor)
    ]

    # Execute the command
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Upscaling successful:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during upscaling:", e.stderr)


def thresh_by_tile(base_image, tile_size=512):
    # split image into square tiles of length tile_size
    base_size = base_image.shape[0]  # assuming image is square
    divs = base_size // tile_size
    num_tiles = int(divs ** 2)  # number of tiles
    tiles = reshape_split(base_image, tile_size)

    # flatten tile array along one dimension
    tiles = tiles.reshape(num_tiles, tile_size, tile_size)

    # apply threshold to each tile
    thresholded = np.zeros_like(tiles, dtype='bool')
    for i in range(0, tiles.shape[0]):
        img = tiles[i]
        if img.any():
            ints = img.flatten()
            # mask out black values
            ints = ints[ints != 0]
            mean = np.mean(ints)
            std = np.std(ints)
            # check if there's anything significantly darker than the background
            if len(ints[ints < mean - 7 * std]) > 64:
                # threshold with Otsu's method
                thresh = ski.filters.threshold_otsu(img[img != 0])
                thresh = 0.90 * thresh
                otsu = img < thresh
                thresholded[i] = otsu

                # # morphological segmentation using Chan-Vese
                # img = ski.img_as_float(img)
                # cv = ski.segmentation.chan_vese(img, mu=0.01, lambda1=1, lambda2=1, tol=1e-5,
                #                max_num_iter=1000, dt=1, init_level_set="checkerboard",
                #                extended_output=True)
                # thresholded[i] = cv[0]
            else:
                thresholded[i] = np.zeros_like(img, dtype='bool')
        else:
            thresholded[i] = np.zeros_like(img, dtype='bool')

    # reassemble thresholded tiles into grid
    thresholded = thresholded.reshape(divs, divs, tile_size, tile_size)
    # stitch them together
    stitched = thresholded.swapaxes(1, 2).reshape(base_image.shape)
    return stitched


def find_objects(image):
    # remove small elements with morphological opening
    footprint = ski.morphology.square(3)
    closed = ski.morphology.binary_closing(image, footprint)
    opened = ski.morphology.binary_opening(closed, footprint)

    # remove artifacts connected to image border
    cleared = ski.segmentation.clear_border(opened, bgval=0)

    # label image regions
    labeled = ski.measure.label(cleared)
    return labeled


def main(image_path, fits_path, scale_factor=4, tile_size=2048):
    # read image and upscale if necessary
    if scale_factor > 1:
        try:
            base_image_path = 'upscaled_image.png'
            upscale_image(image_path, base_image_path, scale_factor)
            base_image = read_image("upscaled_image.png")
        except Exception as e:
            print(f"Error upscaling image: {e}")
            base_image = read_image(image_path)
    else:
        base_image = read_image(image_path)

    # read fits image
    fits_image = open_fits_image(fits_path, 0)

    # treshold it by tiling. play around with the tile size
    print(f"Thresholding {base_image.shape[0]}x{base_image.shape[1]} image with tile size {tile_size}...")
    thresholded = thresh_by_tile(base_image, tile_size=tile_size)

    # label image
    print("Labeling...")
    label_image = find_objects(thresholded)
    print(f"Found {label_image.max()} objects in image.")

    # GET FOLLOWING REGION PROPERTIES FROM THESE LABELS
    '''bounding box, area, centroid coords, centroid pixel, intensity, minimum intensity, coords of min intensity 
    pixel, average intensity'''

    print("Getting region properties...")
    props = measure.regionprops(label_image, intensity_image=base_image)
    print("Writing properties to file...")
    with open('output/region_properties.csv', 'w') as file:
        file.write(
            "Label, BBox Min Row, BBox Min Col, BBox Max Row, BBox Max Col, Area, Centroid Row, Centroid Col, "
            "Centroid Intensity, Minimum Intensity, Minimum Intensity Row, Minimum Intensity Col, Average Intensity\n"
        )
        for region in props:
            # Adjust bounding box and centroid by scale factor
            bbox = tuple([int(b / scale_factor) for b in region.bbox])
            centroid = tuple([int(c / scale_factor) for c in region.centroid])
            area = int(region.area / (scale_factor ** 2))  # Area scales with the square of the scale factor

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

            mean_intensity = np.mean(region_slice)

            # Format the data as a string to write to the file
            region_data = (
                f"{region.label}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {area}, {centroid[0]}, {centroid[1]}, "
                f"{centroid_intensity}, {min_intensity}, {min_intensity_coords[0]}, {min_intensity_coords[1]}, "
                f"{mean_intensity}\n"
            )
            file.write(region_data)

    # rescale image intensity to 8-bit dtype so it can be overlaid with labels
    rescaled_image = ski.exposure.rescale_intensity(base_image)
    base_as_ubyte = ski.util.img_as_ubyte(rescaled_image)
    print("Plotting features on base image...")
    image_label_overlay = ski.color.label2rgb(label_image, image=base_as_ubyte)


    # output images from each step
    print("Saving thresholded image...")
    ski.io.imsave('output/thresholded.png', ski.util.img_as_ubyte(thresholded))
    print("Saving labeled image...")
    ski.io.imsave('output/labeled.png', ski.util.img_as_ubyte(image_label_overlay))
    print("Done!")

    # remove upscaled image
    # os.remove(base_image_path)


if __name__ == '__main__':
    main("./test_res/input.jpg", "./test_res/input.fits", scale_factor=2)
    # main("/home/jswen/dev/solar-yolo/data/fits_images/20150508/hmi.in_45s.20150508_000000_TAI.2.continuum.fits", scale_factor=1, tile_size=512)
