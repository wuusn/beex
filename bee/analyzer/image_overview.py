import numpy as np
import matplotlib.pyplot as plt
from preprocessor.img import read_image

def plot_images(images, ncols=10, save_path=None):
    n = len(images)
    nrows = n // ncols
    if n % ncols != 0:
        nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    fig.set_facecolor('black')
    fig.tight_layout()
    plt.axis('off')

    for ax, img in zip(axes.flatten(), images):
        # check if image is grayscale
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def image_overview(image_files, save_dir):
    """
    Plot image overview for all cohorts,
    each cohort has two rows of images, each row has 10 images.
    param: image_files: dict, {cohort_name: [image_file_paths]}
    param: save_dir: str, path to save dir
    """
    images = []
    for cohort_name, files in image_files.items():
        # random select 20 files
        indices = np.random.choice(len(files), 20, replace=False)
        files = np.array(files)[indices]
        for file in files:
            img = read_image(file)
            images.append(img)
    plot_images(images, ncols=10, save_path=save_dir+'/image_overview.png')
    print('image overview saved to {}'.format(save_dir+'/image_overview.png'))