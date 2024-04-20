import numpy as np
import skimage as ski
from astropy.io import fits

def open_fits_image(image_path, image_data_header_location):
    print(f"Reading {image_path}")
    image_file = open(image_path, "rb")
    hdu_list = fits.open(image_file)
    hdu_list.info()
    image_data = hdu_list[image_data_header_location].data
    return image_data


def reshape_split(image: np.ndarray, kernel_size: int):
    img_height, img_width = image.shape
    tile_height = tile_width = kernel_size # tile must be square

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array


def read_image(img_name):
    if img_name.split('.')[-1] == 'fits':
        base_image = open_fits_image(img_name, 0)
    else:
        base_image = ski.io.imread(img_name, as_gray=True)
    return base_image

def thresh_by_tile(base_image, tile_size=512,):
    # split image into square tiles of length tile_size
    base_size = base_image.shape[0] # assuming image is square
    divs = base_size//tile_size
    num_tiles = int((divs)**2)      # number of tiles
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
            if len(ints[ints < mean - 7*std]) > 64:
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


def main():
    # read image
    img_name = '/home/jswen/dev/solar-yolo/data/fits_images/20140611/hmi.in_45s.20140611_000000_TAI.2.continuum.fits'
    base_image = read_image(img_name)

    # treshold it by tiling. play around with the tile size
    thresholded = thresh_by_tile(base_image, tile_size=512)

    # label image
    label_image = find_objects(thresholded)
    print(type(label_image))
    print(label_image.shape)

    # TODO: GET FOLLOWING REGION PROPERTIES FROM THESE LABELS: 
    # bounding box, area, centroid coords, centroid pixel intensity, minimum intensity, coords of min intensity pixel, average intensity

    # rescale image to 8-bit dtype so it can be overlaid with labels
    rescaled_image = ski.exposure.rescale_intensity(base_image)
    base_as_ubyte = ski.util.img_as_ubyte(rescaled_image)
    image_label_overlay = ski.color.label2rgb(label_image, image=base_as_ubyte)
    print(type(image_label_overlay))
    print(image_label_overlay.shape)

    # output images from each step
    ski.io.imsave('thresholded.png', ski.util.img_as_ubyte(thresholded))
    print("saved thresholded image")
    ski.io.imsave('labeled.png', ski.util.img_as_ubyte(image_label_overlay))
    print("saved labeled image")


if __name__ == '__main__':
    main()
