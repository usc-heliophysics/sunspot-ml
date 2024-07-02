import os
import sys
import time
from mutithread_sunspot_dec import list_fits_files, process_images

directory_path = 'test_res/20150512_96'
fits_files = list_fits_files(directory_path)

feature = "penumbrae"

# These parameters have been found to work well for the various processing steps:
findroi_kwargs = {"num_stdevs": 7, "padding": 40}
kmeans_kwargs = {"K": 6, "blur_strength": 1}
clearbg_kwargs = {"bwidth": 10, "bg_min_count": 50}


# Function to suppress stdout and stderr
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

if __name__ == '__main__':
    # Loop to check benchmark for multiple workers
    for workers in range(4, 37, 4):  # Increment by 4
        start_time = time.time()

        with SuppressOutput():
            process_images(fits_files, max_workers=workers,
                           feature=feature, findroi_kwargs=findroi_kwargs,
                           kmeans_kwargs=kmeans_kwargs, clearbg_kwargs=clearbg_kwargs)

        end_time = time.time()
        duration = end_time - start_time

        print(f"Workers: {workers}, Time taken: {duration:.2f} seconds")
