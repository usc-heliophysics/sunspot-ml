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


def thresh_by_tile(base_image, tile_size=512, ):
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


def main(image_path, scale_factor=4, tile_size=2048):
    # read image
    base_image_path = 'upscaled_image.png'
    upscale_image(image_path, base_image_path, scale_factor)
    base_image = read_image("upscaled_image.png")
    # treshold it by tiling. play around with the tile size
    thresholded = thresh_by_tile(base_image, tile_size=tile_size)

    # label image
    label_image = find_objects(thresholded)
    print(type(label_image))
    print(label_image.shape)

    # GET FOLLOWING REGION PROPERTIES FROM THESE LABELS: bounding box, area, centroid coords, centroid pixel
    #  intensity, minimum intensity, coords of min intensity pixel, average intensity
    props = measure.regionprops(label_image, intensity_image=base_image)
    with open('output/region_properties.txt', 'w') as file:
        file.write(
            "Label, Bounding Box, Area, Centroid Coordinates, Centroid Intensity, Minimum Intensity, Coordinates of "
            "Min Intensity Pixel, Average Intensity\n")
        for region in props:
            # Adjust bounding box and centroid by scale factor
            bbox = tuple([int(b / scale_factor) for b in region.bbox])
            centroid = tuple([int(c / scale_factor) for c in region.centroid])
            area = int(region.area / (scale_factor ** 2))  # Area scales with the square of the scale factor

            # Get intensity data without scaling adjustments
            centroid_intensity = base_image[int(centroid[0]), int(centroid[1])]
            min_intensity = region.min_intensity
            min_intensity_index = region.intensity_image.argmin()
            min_intensity_coords = np.unravel_index(min_intensity_index, region.intensity_image.shape)
            min_intensity_coords = tuple([int(mc / scale_factor) for mc in min_intensity_coords])

            # Format the data as a string to write to the file
            region_data = f"{region.label}, {bbox}, {area}, {centroid}, {centroid_intensity}, {min_intensity}, {min_intensity_coords}, {region.mean_intensity}\n"
            file.write(region_data)
    # rescale image to 8-bit dtype so it can be overlaid with labels
    rescaled_image = ski.exposure.rescale_intensity(base_image)
    base_as_ubyte = ski.util.img_as_ubyte(rescaled_image)
    image_label_overlay = ski.color.label2rgb(label_image, image=base_as_ubyte)
    print(type(image_label_overlay))
    print(image_label_overlay.shape)

    # output images from each step
    ski.io.imsave('output/thresholded.png', ski.util.img_as_ubyte(thresholded))
    print("saved thresholded image")
    ski.io.imsave('output/labeled.png', ski.util.img_as_ubyte(image_label_overlay))
    print("saved labeled image")
    # remove upscaled image
    os.remove(base_image_path)


if __name__ == '__main__':
    main("./test_res/input.jpg")
