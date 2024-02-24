"""
Data loading:
read WSI SVS files using openslide-python at 1.25x
read other pathological image files using PIL
read medical imaging files using #TODO
"""

import numpy as np
import openslide
import matplotlib.pyplot as plt
from PIL import Image

def rgba2rgb(slide, img):
    """
    Convert RGBA to RGB
    params slide: openslide object
    params img: RGBA image
    return: RGB PIL image
    """
    bg_color = "#" + slide.properties.get(openslide.PROPERTY_NAME_BACKGROUND_COLOR, "ffffff")
    thumb = Image.new("RGB", img.size, bg_color)
    thumb.paste(img, None, img)
    return thumb


def read_svs(filepath, target_mag=1.25):
    """
    Read a svs file at a given magnification
    params filepath: path to svs file
    params target_mag: target magnification, default 1.25
    return: RGB numpy array
    """
    slide = openslide.OpenSlide(filepath)
    img_base_size = slide.dimensions
    dim_width, dim_height = img_base_size
    img_bbox = (0,0,dim_width,dim_height)
    base_mag = slide.properties.get("openslide.objective-power") or \
                slide.properties.get("aperio.AppMag") or None
    base_mag = float(base_mag)

    target_sampling_factor = base_mag / target_mag
    target_dims = tuple(np.rint(np.asarray(img_base_size) / target_sampling_factor).astype(int))

    relative_down_factors_idx=[np.isclose(i/target_sampling_factor,1,atol=.01) for i in slide.level_downsamples]
    level=np.where(relative_down_factors_idx)[0]
    if level.size:
        level, isExactLevel =  (level[0], True)
    else:
        level, isExactLevel =  (slide.get_best_level_for_downsample(target_sampling_factor), False)

    if isExactLevel:
        tile = slide.read_region((0, 0), level, target_dims)
        if np.shape(tile)[-1]==4:
            return np.asarray(rgba2rgb(slide, tile))
        else:
            return np.asarray(tile)
    # scale down the thumb img from the next high level
    else:
        #im=  resizeTileDownward(s, target_sampling_factor, level)
        cloest_downsampling_factor = slide.level_downsamples[level]
        win_size = 2048
        # create a new img
        (bx, by, bwidth, bheight) = img_bbox
        end_x = bx + bwidth
        end_y = by + bheight
        output = []
        for x in range(bx, end_x, win_size):
            row_piece = []
            for y in range(by, end_y, win_size):
                win_width, win_height = [win_size] * 2
                # Adjust extraction size for endcut
                if end_x < x + win_width:
                    win_width = end_x - x
                if end_y < y +  win_height:
                    win_height = end_y - y

                
                win_down_width = int(round(win_width / target_sampling_factor))
                win_down_height = int(round(win_height / target_sampling_factor))
                
                win_width = int(round(win_width / cloest_downsampling_factor))
                win_height = int(round(win_height / cloest_downsampling_factor))
                
                # TODO Note: this isn't very efficient, and if more efficiency isneeded 
                # We should likely refactor using "paste" from Image.
                # Or even just set the pixels directly with indexing.
                cloest_region = slide.read_region((x, y), level, (win_width, win_height))
                if np.shape(cloest_region)[-1]==4:
                    cloest_region = rgba2rgb(slide, cloest_region)
                target_region = cloest_region.resize((win_down_width, win_down_height))
                row_piece.append(target_region)
            row_piece = np.concatenate(row_piece, axis=0)
            
            output.append(row_piece)
        output = np.concatenate(output, axis=1)
        return output

def read_tile(filepath, downscale=1):
    """
    Read a pathological image file
    params filepath: path to image file
    params downscale: downscale factor, default 4
    return: RGB numpy array
    """
    im = Image.open(filepath).convert('RGB')
    im = im.resize((im.size[0]//downscale, im.size[1]//downscale))
    return np.asarray(im)  

#TODO: add a function to read 3s medical imaging files

#TODO: need to update this to support more image types
def read_image(path):
    ext = path.split('.')[-1]
    if ext == 'svs':
        return read_svs(path)
    elif ext in ['png', 'jpg', 'jpeg']:
        return read_tile(path)
    else:
        raise NotImplementedError(f'Not implemented for {ext} file type')