import concurrent.futures
import os
from segment import segment_core

def process_images(image_list, max_workers=8, **segment_kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image in image_list:
            png_path = f"{image}.fits.png"
            # fits_path = f"{image}.fits"
            fits_path = image
            # Schedule the execution of the main function
            future = executor.submit(segment_core, fits_path, **segment_kwargs)
            futures.append(future)

        # Waiting for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

# fits file path parser
def list_fits_files(directory):
    image_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".fits"):
            image_files.append(os.path.join(directory, filename))
    return image_files


directory_path = '/home/jswen/dev/solar-yolo/data/fits_images/20150508'
fits_files = list_fits_files(directory_path)

image_files = [
    "test_res/hmi.in_45s.20150508_000000_TAI.2.continuum.fits",
    "test_res/hmi.in_45s.20150508_230000_TAI.2.continuum.fits",
    "test_res/hmi.in_45s.20150512_000000_TAI.2.continuum.fits",
    "test_res/hmi.in_45s.20150512_220000_TAI.2.continuum.fits",
]

if __name__ == '__main__':
    # process all files in directory
    # from glob import glob
    # image_files = glob("/home/jswen/dev/solar-yolo/data/fits_images/20150508/*.fits")

    # feature to be detected. choose between "umbrae" (dark sunspot cores) or
    # "penumbrae" (lighter region surrounding the umbrae)
    feature = "penumbrae"

    # these parameters have been found to work well for the various processing steps:
    findroi_kwargs = {"num_stdevs": 7, "padding": 50}
    kmeans_kwargs = {"K": 5, "blur_strength": 3}
    clearbg_kwargs = {"bwidth": 10, "bg_min_count": 50}

    process_images(fits_files, max_workers=4,
                   feature=feature, findroi_kwargs=findroi_kwargs,
                   kmeans_kwargs=kmeans_kwargs, clearbg_kwargs=clearbg_kwargs)
