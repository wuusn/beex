import os
import sys
import numpy
import cv2
import copy
from PIL import Image
from ast import literal_eval as make_tuple
import skimage
from skimage import io, color, img_as_ubyte, morphology
from skimage.filters import sobel
from skimage.filters import gabor_kernel, frangi, gaussian, median, laplace
from skimage.color import convert_colorspace, rgb2gray, rgb2hsv, separate_stains, hed_from_rgb
from skimage.morphology import remove_small_objects, disk, binary_opening, dilation
from distutils.util import strtobool
import numpy as np
import scipy
from scipy import ndimage as ndi
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool



def getTissueMask(img):
    upper_thresh = .9
    lower_var = 10
    lower_thresh = -float('inf')
    upper_var = float('inf')
    img_var = img.std(axis=2)
    map_var = np.bitwise_and(img_var > lower_var, img_var < upper_var)
    img = color.rgb2gray(img)
    map = np.bitwise_and(img > lower_thresh, img < upper_thresh)
    map = np.bitwise_and(map, map_var)
    map = (map >0)
    #map = ~map
    #map = map.astype(np.uint8)
    return map

def getDarkTissueMask(img):
    upper_thresh = .15
    lower_var = -float('inf')
    lower_thresh = -float('inf')
    upper_var = float('inf')
    img_var = img.std(axis=2)
    map_var = np.bitwise_and(img_var > lower_var, img_var < upper_var)
    img = color.rgb2gray(img)
    map = np.bitwise_and(img > lower_thresh, img < upper_thresh)
    map = np.bitwise_and(map, map_var)
    map = (map >0)
    #map = ~map
    #map = map.astype(np.uint8)
    return map

def getBrightness(img, tmask):
    a = getBrightnessGray(img, tmask)
    b = getBrightnessOtherColor(img, tmask)
    return {**a, **b}

def getBrightnessGray(img, tmask):
    img_g = rgb2gray(img)
    img_gm = img_g[tmask]

    return {
            # 'Gray Brightness mean (W)': img_g.mean(),
            # 'Gray Brightness std (W)':img_g.std(),
            'Gray Brightness mean':img_gm.mean(),
            'Gray Brightness std':img_gm.std()
            }

def getBrightnessOtherColor(img, tmask):
    a = {}
    #color_spaces = ['RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ',
    #                'YPbPr', 'YCbCr']
    color_spaces = ['RGB', 'HSV', 'YUV']
    for color in color_spaces:
        if color != 'RGB':
            img2 = convert_colorspace(img, 'RGB', color)
        else:
            img2 = img
        for chan in range(0, 3):
            vals = img2[:, :, chan]
            vals_t = vals[tmask]
            # a[f'{color} {chan+1} Brightness mean (W)'] = vals.mean()
            # a[f'{color} {chan+1} Brightness std (W)'] = vals.std()
            a[f'{color} {chan+1} Brightness mean'] = vals_t.mean()
            a[f'{color} {chan+1} Brightness std'] = vals_t.std()

    return a

def getHueSaturation(img, tmask):
    a = {}
    hsv = rgb2hsv(img)

    # a[f'Hue mean (W)'] = hsv[:,:,0].mean()
    # a[f'Hue std (W)'] = hsv[:,:,0].std()
    # a[f'Saturation mean (W)'] = hsv[:,:,1].mean()
    # a[f'Saturation std (W)'] = hsv[:,:,1].std()
    a[f'Hue mean'] = hsv[:,:,0][tmask].mean()
    a[f'Hue std'] = hsv[:,:,0][tmask].std()
    a[f'Saturation mean'] = hsv[:,:,1][tmask].mean()
    a[f'Saturation std'] = hsv[:,:,1][tmask].std()

    return a

def getHE(img, tmask):
    a = {}
    
    stain_matrix = hed_from_rgb
    dimg = separate_stains(img, stain_matrix)

    # a[f'Stain H mean (W)'] = dimg[:,:,0].mean()
    # a[f'Stain H std (W)'] = dimg[:,:,0].std()
    # a[f'Stain E mean (W)'] = dimg[:,:,1].mean()
    # a[f'Stain E std (W)'] = dimg[:,:,1].std()
    a[f'Stain H mean'] = dimg[:,:,0][tmask].mean()
    a[f'Stain H std'] = dimg[:,:,0][tmask].std()
    a[f'Stain E mean'] = dimg[:,:,1][tmask].mean()
    a[f'Stain E std'] = dimg[:,:,1][tmask].std()

    return a

def getContrast(img, tmask):
    a = {}

    img = rgb2gray(img)
    sobel_img = sobel(img) ** 2
    tenenGrad_contrast = np.sqrt(np.sum(sobel_img)) / img.size
    max_img = img.max()
    min_img = img.min()
    michelson_contrast = (max_img - min_img) / (max_img + min_img)
    rms_contrast = np.sqrt(pow(img - img.mean(), 2).sum() / img.size)
    # a[f'Contrast TenenGrad (W)'] = tenenGrad_contrast
    # a[f'Contrast Michelson (W)'] = michelson_contrast
    #a[f'Contrast RMS (W)'] = rms_contrast

    img = img[tmask]
    sobel_img = sobel_img[tmask]
    tenenGrad_contrast = np.sqrt(np.sum(sobel_img)) / img.size
    max_img = img.max()
    min_img = img.min()
    michelson_contrast = (max_img - min_img) / (max_img + min_img)
    rms_contrast = np.sqrt(pow(img - img.mean(), 2).sum() / img.size)
    a[f'Contrast TenenGrad'] = tenenGrad_contrast
    a[f'Contrast Michelson'] = michelson_contrast
    #a[f'Contrast RMS (T)'] = rms_contrast

    return a

def getSmoothness(img, tmask):
    thresh = .01
    kernel_size = 10
    min_object_size = 500
    img = color.rgb2gray(img)
    avg = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    imf = scipy.signal.convolve2d(img, avg, mode="same")
    mask_flat = abs(imf - img) < thresh
    mask_flat = remove_small_objects(mask_flat, min_size=min_object_size)
    mask_flat = ~remove_small_objects(~mask_flat, min_size=min_object_size)
    ratio = np.sum(mask_flat) / (mask_flat.shape[0]*mask_flat.shape[1])
    a = {}
    a['Smoothness Ratio'] = ratio
    return a

def remove_large_objects(img, max_size):
    # code taken from morphology.remove_small_holes, except switched < with >
    selem = ndi.generate_binary_structure(img.ndim, 1)
    ccs = np.zeros_like(img, dtype=np.int32)
    ndi.label(img, selem, output=ccs)
    component_sizes = np.bincount(ccs.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    img_out = img.copy()
    img_out[too_big_mask] = 0
    return img_out

def getFatlikeTissue(img, tmask):
    a = {}
    kernel_size = 10
    max_keep_size = 1000
    fat_cell_size = 64
    img_reduced = morphology.remove_small_holes(tmask, area_threshold=fat_cell_size)
    img_small = img_reduced & np.invert(tmask)
    img_small = ~morphology.remove_small_holes(~img_small, area_threshold=9)
    mask_dilate = morphology.dilation(img_small, footprint=np.ones((kernel_size, kernel_size)))
    mask_dilate_removed = remove_large_objects(mask_dilate, max_keep_size)
    mask_fat = mask_dilate & ~mask_dilate_removed
    mask_fat = (mask_fat * 255) >0
    ratio = np.sum(mask_fat) / (mask_fat.shape[0] * mask_fat.shape[1])
    a = {'Fat Like Tissue Ratio': ratio}
    return a

def compute_features(img, params):
    features = params.get("features", "")
    feats=[]
    for feature in features:
        func = getattr(sys.modules[__name__], f'compute_{feature}')
        feats.append(func(img, params))
    return np.concatenate(feats, axis=2)

def compute_rgb(img, params):
    return img

def compute_laplace(img, params):
    laplace_ksize = int(params.get("laplace_ksize", 3))
    return laplace(rgb2gray(img), ksize=laplace_ksize)[:, :, None]

def compute_gaussian(img, params):
    gaussian_sigma = int(params.get("gaussian_sigma", 1))
    gaussian_multichan = strtobool(params.get("gaussian_multichan", False))

    if (gaussian_multichan):
        return gaussian(img, sigma=gaussian_sigma, multichannel=gaussian_multichan)
    else:
        return gaussian(rgb2gray(img), sigma=gaussian_sigma)[:, :, None]

def compute_median(img, params):
    median_disk_size = int(params.get("median_disk_size", 3))
    return median(rgb2gray(img), selem=disk(median_disk_size))[:, :, None]

def compute_gabor(img, params):
    if not params["shared_dict"].get("gabor_kernels", False):
        gabor_theta = int(params.get("gabor_theta", 4))
    gabor_sigma = make_tuple(params.get("gabor_sigma", "(1,3)"))
    gabor_frequency = make_tuple(params.get("gabor_frequency", "(0.05, 0.25)"))

    kernels = []
    for theta in range(gabor_theta):
        theta = theta / 4. * np.pi
        for sigma in gabor_sigma:
            for frequency in gabor_frequency:
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                        sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    params["shared_dict"]["gabor_kernels"] = kernels

    kernels = params["shared_dict"]["gabor_kernels"]
    imgg = rgb2gray(img)
    feats = np.zeros((imgg.shape[0], imgg.shape[1], len(kernels)), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(imgg, kernel, mode='wrap')
        feats[:, :, k] = filtered
    return feats

def compute_frangi(img, params):
    frangi_scale_range = make_tuple(params.get("frangi_scale_range", "(1, 10)"))
    frangi_scale_step = float(params.get("frangi_scale_step", 2))
    frangi_beta1 = float(params.get("frangi_beta1", .5))
    frangi_beta2 = float(params.get("frangi_beta2", 15))
    frangi_black_ridges = strtobool(params.get("frangi_black_ridges", "True"))
    feat = frangi(rgb2gray(img), scale_range = frangi_scale_range, scale_step =frangi_scale_step, beta =frangi_beta1, gamma=frangi_beta2, black_ridges  =frangi_black_ridges)
    return feat[:, :, None]  # add singleton dimension

def trainModelPen(example_path, example_mask_path):
    model_vals = []
    model_labels = np.empty([0, 1])
    img = io.imread(example_path)
    params = dict(
                threshold = .5,
                area_threshold = 100,
                features = ['frangi', 'laplace', 'rgb'],
                laplace_ksize = 3,
                frangi_scale_range = '(1,10)',
                frangi_scale_step = 2,
                frangi_beta1 = .5,
                frangi_beta2= 15,
                frangi_black_ridges= 'True',

                gabor_theta= 4,
                gabor_sigma= '(1,3)',
                gabor_frequency= '(0.05, 0.25)',

                lbp_radius= 3,
                lbp_points= 24,
                lbp_method= 'default',

                median_disk_size= 3,
             )
    eximg = compute_features(img, params)
    eximg = eximg.reshape(-1, eximg.shape[2])
    model_vals.append(eximg)
    mask = io.imread(example_mask_path, as_gray=True).reshape(-1,1)
    model_labels = np.vstack((model_labels, mask))

    model_vals = np.vstack(model_vals)
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(model_vals, model_labels.ravel())
    return clf

def trainCoverslipEdge(example_path, example_mask_path):
    params = dict(
                area_threshold = 15,
                features =  ['frangi', 'laplace', 'rgb'],
                dilate_kernel_size = 5,
    )
    model_vals = []
    model_labels = np.empty([0, 1])
    img = io.imread(example_path)
    eximg = compute_features(img, params)
    eximg = eximg.reshape(-1, eximg.shape[2])
    model_vals.append(eximg)
    mask = io.imread(example_mask_path, as_gray=True).reshape(-1,1)
    model_labels = np.vstack((model_labels, mask))

    model_vals = np.vstack(model_vals)
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(model_vals, model_labels.ravel())
    return clf

def getBlurryRegion(src_img, tmask):
    #img_work_size = 2.5x
    blur_radius = 100
    blur_threshold = .15
    img = rgb2gray(src_img)
    img_laplace = np.abs(skimage.filters.laplace(img))
    mask = skimage.filters.gaussian(img_laplace, sigma=blur_radius) <= blur_threshold
    mask = skimage.transform.resize(mask, src_img.shape, order=0)[:,:,1]
    mask = tmask & (mask >0)
    ratio = np.sum(mask) / mask.size
    a = {'Blurry Region Ratio': ratio}
    return a

def getPenMarking(img, tmask, model):
    params = dict(
                threshold = .5,
                area_threshold = 100,
                features = ['frangi', 'laplace', 'rgb'],
                laplace_ksize = 3,
                frangi_scale_range = '(1,10)',
                frangi_scale_step = 2,
                frangi_beta1 = .5,
                frangi_beta2= 15,
                frangi_black_ridges= 'True',

                gabor_theta= 4,
                gabor_sigma= '(1,3)',
                gabor_frequency= '(0.05, 0.25)',

                lbp_radius= 3,
                lbp_points= 24,
                lbp_method= 'default',

                median_disk_size= 3,
    )
    thresh = float(params.get("threshold", .5))
    clf = model
    feats = compute_features(img, params)
    cal = clf.predict_proba(feats.reshape(-1, feats.shape[2]))
    cal = cal.reshape(img.shape[0], img.shape[1], 2)
    mask = cal[:, :, 1] > thresh
    area_thresh = int(params.get("area_threshold", "5"))
    if area_thresh > 0:
        mask = remove_small_objects(mask, min_size=area_thresh, in_place=True)
    dilate_kernel_size = int(params.get("dilate_kernel_size", "0"))
    if dilate_kernel_size > 0:
        mask = dilation(mask, footprint=np.ones((dilate_kernel_size, dilate_kernel_size)))
    mask = tmask & (mask > 0)
    ratio = np.sum(mask)/np.sum(tmask)
    return ratio

def getCoverslipEdge(img, tmask, model):
    params = dict(
                area_threshold = 15,
                features =  ['frangi', 'laplace', 'rgb'],
                dilate_kernel_size = 5,
    )
    thresh = float(params.get("threshold", .5))
    clf = model
    feats = compute_features(img, params)
    cal = clf.predict_proba(feats.reshape(-1, feats.shape[2]))
    cal = cal.reshape(img.shape[0], img.shape[1], 2)
    mask = cal[:, :, 1] > thresh
    area_thresh = int(params.get("area_threshold", "5"))
    if area_thresh > 0:
        mask = remove_small_objects(mask, min_size=area_thresh, in_place=True)
    dilate_kernel_size = int(params.get("dilate_kernel_size", "0"))
    if dilate_kernel_size > 0:
        mask = dilation(mask, footprint=np.ones((dilate_kernel_size, dilate_kernel_size)))
    mask = tmask & (mask > 0)
    ratio = np.sum(mask)/np.sum(tmask)
    return ratio