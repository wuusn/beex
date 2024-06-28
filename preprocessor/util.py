import os
import sys
import glob
import pathlib

supported_exts = ['svs', 'tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp', 'nii.gz', 'gz', 'dcm']

# get all file paths in a folder its subfolders with or w/t extension
def get_paths(folder, identifier=None, exts=None):
    """
    Get all file paths in a folder its subfolders with or w/t extension
    params folder: folder path
    params identifier: identifier of the cohort
    params ext: extension, list, default 
    return: list of file paths
    """
    filepaths = []
    exts = supported_exts if exts is None else exts
    exts = list(set(exts))
    for root, dirs, files in os.walk(folder, followlinks=True):
        for file in files:
            file_ext = file.split('.')[-1]
            if identifier is None:
                for ext in exts:
                    if file_ext in ext:
                        filepaths.append(os.path.join(root, file))
            elif identifier in file:
                for ext in exts:
                    if file_ext in ext:
                        filepaths.append(os.path.join(root, file))
    return list(set(filepaths))

