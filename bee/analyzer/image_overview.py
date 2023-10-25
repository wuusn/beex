import numpy as np
import matplotlib.pyplot as plt
from preprocessor.img import read_image
from multiprocessing import Pool
from PIL import Image

def plot_images(images, ncols=10, save_path=None):
    n = len(images)
    nrows = n // ncols
    if n % ncols != 0:
        nrows += 1

    plt.figure()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    fig.set_facecolor('black')
    fig.tight_layout()

    for ax, img in zip(axes.flatten(), images):
        # check if image is grayscale
        ax.axis('off')
        # downsacle image level 4
        small_img = Image.fromarray(img).resize((img.shape[1]//4, img.shape[0]//4))
        img = np.array(small_img)
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def image_overview(image_files, ncpus, save_dir):
    """
    Plot image overview for all cohorts,
    each cohort has two rows of images, each row has 10 images.
    param: image_files: dict, {cohort_name: [image_file_paths]}
    param: ncpus: int, number of cpus
    param: save_dir: str, path to save dir
    """
    images = []
    n_cohort = len(image_files.keys())
    # find largest value can be divided by n_cohort range from [16,21]
    ncols = None
    for i in range(21, 15, -1):
        if i % n_cohort == 0:
            ncols = i
            break
    
    n_cohort_rows = ncols // n_cohort
    

    for cohort_name, files in image_files.items():
        # random select 20 files
        indices = np.random.choice(len(files), n_cohort_rows*ncols, replace=False)
        files = np.array(files)[indices]
        # using multi processing
        with Pool(ncpus) as p:
            images.extend(p.map(read_image, files))

    plot_images(images, ncols=ncols, save_path=save_dir+'/image_overview.png')
    print('image overview saved to {}'.format(save_dir+'/image_overview.png'))