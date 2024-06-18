import concurrent.futures
from segment import segment_core


def process_images(image_list, feature, max_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image in image_list:
            png_path = f"{image}.fits.png"
            fits_path = f"{image}.fits"
            # Schedule the execution of the main function
            future = executor.submit(segment_core, fits_path, feature=feature)
            futures.append(future)

        # Waiting for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            print(future.result())




image_files = [
    "test_res/hmi.in_45s.20150508_000000_TAI.2.continuum",
    "test_res/hmi.in_45s.20150508_230000_TAI.2.continuum",
    "test_res/hmi.in_45s.20150512_000000_TAI.2.continuum",
    "test_res/hmi.in_45s.20150512_220000_TAI.2.continuum"
]

process_images(image_files, feature="penumbrae", max_workers=2)
