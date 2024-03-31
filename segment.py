import numpy as np
import skimage as ski
from astropy.io import fits
import matplotlib.pyplot as plt

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


# read fits image
img_name = '/home/jswen/dev/solar-yolo/data/fits_images/20140611/hmi.in_45s.20140611_000000_TAI.2.continuum.fits'
base_image = open_fits_image(img_name, 0)

# split image into square tiles of length tile_size
tile_size = 1024
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
        mean = np.mean(ints)
        std = np.std(ints)
        # check if there's anything significantly darker than the background
        if np.any(ints < mean - 5*std):
            # threshold with Otsu's method
            thresh = ski.filters.threshold_otsu(img[img != 0])
            otsu = img >= thresh
            thresholded[i] = otsu
            
            # # morphological segmentation using Chan-Vese
            # img = ski.img_as_float(img)
            # cv = ski.segmentation.chan_vese(img, mu=0.01, lambda1=1, lambda2=1, tol=1e-5,
            #                max_num_iter=1000, dt=1, init_level_set="checkerboard",
            #                extended_output=True)
            # thresholded[i] = cv[0]
        else:
            thresholded[i] = np.ones_like(img, dtype='bool')
    else:
        thresholded[i] = np.zeros_like(img, dtype='bool')

# reassemble thresholded tiles into grid
thresholded = thresholded.reshape(divs, divs, tile_size, tile_size)
# stitch them together
stitched = thresholded.swapaxes(1, 2).reshape(4096, 4096)

plt.figure(figsize=(15, 15))
plt.imshow(stitched, cmap='grey')
ax = plt.gca()
ax.set_axis_off()
plt.savefig('segmentation.png', dpi=300)
plt.show()
