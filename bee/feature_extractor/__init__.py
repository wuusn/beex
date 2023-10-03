import os
from .pathological_metrics import get_path_metrics

def extract_feature(image_files, save_dir, n_workers):
    """
    Extract features from images and save them to excel file
    param: image_files: dict, {cohort_name: [image_file1, image_file2, ...]}
    param: save_dir: str, path to save directory
    param: n_workers: int, number of workers
    return: pandas dataframe
    """
    # check medical and pathological
    # get one file extension
    ext = os.path.splitext(image_files[list(image_files.keys())[0]][0])[1]
    if ext in ['.nii.gz']:
        # medical image
        raise NotImplementedError('Medical image is not supported yet')
    else:
        # pathological image
        features = get_path_metrics(image_files, n_workers)
    # save to excel
    features.to_excel(os.path.join(save_dir, 'features.xlsx'), index=False)
    print('features saved to {}'.format(os.path.join(save_dir, 'features.xlsx')))
    return features
    