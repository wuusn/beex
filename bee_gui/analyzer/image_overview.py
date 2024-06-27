import numpy as np
import matplotlib.pyplot as plt
from preprocessor.img import read_image
from multiprocessing import Pool
from PIL import Image
import os
import glob
from pathlib import Path

def plot_images(images, ncols=10, nrows=10, save_path=None):
    n_cohorts = len(images.keys())
    n = ncols * nrows//n_cohorts

    plt.figure()
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    fig.set_facecolor('white')
    fig.tight_layout()

    axs = axes.flatten()
    # print('max len of axs', len(axs))
    ax_i = 0
    for _, imgs in images.items():
        if len(imgs)<1:
            assert('No images found')
        for img in imgs:
            if ax_i >= len(axs):
                break
            ax = axs[ax_i]
            ax.axis('off')
            small_img = Image.fromarray(img).resize((img.shape[1]//4, img.shape[0]//4))
            img = np.array(small_img)
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax_i += 1
        
        if ax_i >= len(axs):
            break

        if len(imgs) % ncols!=0:
            skip = ncols - len(imgs) % ncols
            for _ in range(skip):
                ax = axs[ax_i]
                ax.axis('off')
                ax_i += 1
        if ax_i >= len(axs):
            break
    for j in range(ax_i, len(axs)):
        axs[j].axis('off')

    # for ax, img in zip(axes.flatten(), images):
    #     # check if image is grayscale
    #     ax.axis('off')
    #     # downsacle image level 4
    #     small_img = Image.fromarray(img).resize((img.shape[1]//4, img.shape[0]//4))
    #     img = np.array(small_img)
    #     if len(img.shape) == 2:
    #         ax.imshow(img, cmap='gray')
    #     else:
    #         ax.imshow(img)

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return save_path

def image_overview(image_files, mode, ncpus, save_dir):
    """
    Plot image overview for all cohorts,
    each cohort has two rows of images, each row has 10 images.
    param: image_files: dict, {cohort_name: [image_file_paths]}
    param: ncpus: int, number of cpus
    param: save_dir: str, path to save dir
    """
    save_dir = str(save_dir)
    images = {}
    for cohort_name in image_files.keys():
        images[cohort_name] = []
    n_cohort = len(image_files.keys())
    
    ncols = None
    for i in range(21, 15, -1):
        if i % n_cohort == 0:
            ncols = i
            break
    n_cohort_rows = max(ncols // n_cohort, 2)
    
    if mode == 'Radiology':
        for cohort_name, files  in image_files.items():
            new_files = []
            extracted_dir = os.path.join(save_dir, 'tmp_data', cohort_name+'_extracted')
            extracted_dir =  Path(extracted_dir)
            new_files = list(extracted_dir.glob('**/*.png'))
            image_files[cohort_name] = new_files
    
    for cohort_name, files in image_files.items():
        # print(len(files), n_cohort_rows*ncols)
        if len(files) < n_cohort_rows*ncols:
            # use all
            indices = np.arange(len(files))
        else:
            indices = np.random.choice(len(files), n_cohort_rows*ncols, replace=False)
        files = np.array(files)[indices]
        # using multi processing
        with Pool(ncpus) as p:
            for cohort_name, files in image_files.items():
                images[cohort_name] = p.map(read_image, files)
    image_path = save_dir+'/image_overview.png'
    return plot_images(images, ncols=ncols, nrows=n_cohort_rows*n_cohort, save_path=image_path)