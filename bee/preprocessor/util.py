import os
import sys
import glob

supported_exts = ['svs', 'tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp']

# get all file paths in a folder its subfolders with or w/t extension
def get_paths(folder, identifier=None, ext=None):
    """
    Get all file paths in a folder its subfolders with or w/t extension
    params folder: folder path
    params identifier: identifier of the cohort
    params ext: extension, list, default 
    return: list of file paths
    """
    filepaths = []
    ext = supported_exts if ext is None else ext
    for root, dirs, files in os.walk(folder):
        for file in files:
            if identifier is None:
                if file.split('.')[-1] in ext:
                    filepaths.append(os.path.join(root, file))
            elif identifier in file and file.split('.')[-1] in ext:
                filepaths.append(os.path.join(root, file))
    return filepaths

