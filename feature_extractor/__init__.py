import os
from .pathological_metrics import get_path_metrics
from .radiological_metrics import get_radi_metrics

def extract_feature(image_files, mask_files, feature_mode, cohort_dirs, save_dir, n_workers):
    """
    Extract features from images and save them to excel file
    param: image_files: dict, {cohort_name: [image_file1, image_file2, ...]}
    param: mask_files: dict, {cohort_name: [mask_file1, mask_file2, ...]}
    param: feature_mode: str, 'radi' or 'path'
    param: save_dir: str, path to save directory
    param: n_workers: int, number of workers
    return: pandas dataframe
    """
    # check feature mode
    if feature_mode == 'radi':
        # radiological image
        features = get_radi_metrics(image_files, mask_files, cohort_dirs, save_dir, n_workers)
    elif feature_mode == 'path':
        # pathological image
        features = get_path_metrics(image_files, mask_files, n_workers)
    else:
        raise ValueError('feature_mode should be "radi" or "path"')
    # save to excel
    feature_path = os.path.join(save_dir, 'features.xlsx')
    # if os.path.exists(feature_path) is False:
    features.to_excel(os.path.join(save_dir, 'features.xlsx'), index=False)
    print('features saved to {}'.format(os.path.join(save_dir, 'features.xlsx')))
    return features
    