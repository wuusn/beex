import os
import pandas as pd
from .QC import run_qc

def get_radi_metrics(image_files, mask_files, cohort_dirs, save_dir, n_workers):
    """
    Get radiological metrics from medical images
    param: image_files: dict, {cohort_name: [image_file1, image_file2, ...]}
    param: mask_files: dict, {cohort_name: [mask_file1, mask_file2, ...]}
    param: n_workers: int, number of workers
    return: pandas dataframe
    """
    # create soft link for each cohort with files
    # this is for using selected files in the feature xlsx
    cohort_names = list(image_files.keys())
    # for cohort_name, files in image_files.items():
    #     old_cohort_dir = cohort_dirs[cohort_names.index(cohort_name)]
    #     new_cohort_dir = os.path.join(save_dir, 'tmp_data', cohort_name)
    #     if not os.path.exists(new_cohort_dir):
    #         os.makedirs(new_cohort_dir)
    #     for file in files:
    #         # target_file = os.path.join(cohort_dir, os.path.basename(file))
    #         target_file = file.replace(old_cohort_dir, new_cohort_dir)
    #         # print(file, target_file)
    #         os.makedirs(os.path.dirname(target_file), exist_ok=True)
    #         if not os.path.exists(target_file):
    #             # print('not exists', target_file)
    #             # get real path
    #             file = os.path.realpath(file)
    #             os.symlink(file, target_file)
    
    features = None
    for cohort_name in image_files.keys():
        print('extracting features for {}'.format(cohort_name))
        # cohort_dir = os.path.join(save_dir, 'tmp_data', cohort_name)
        cohort_dir = cohort_dirs[cohort_names.index(cohort_name)]
        # get current file path
        tmp_feature_dir = os.path.join(save_dir, 'tmp_data', cohort_name+'_extracted')
        res_path = f"{save_dir}/tmp_data/{cohort_name}_extracted/IQM.xlsx"
        if (not os.path.exists(tmp_feature_dir)) or (not os.path.exists(res_path)):
            run_qc(f'{save_dir}/tmp_data/{cohort_name}_extracted',cohort_dir)
        
        cohort_feature = pd.read_excel(res_path, dtype={'Name': str})
        cohort_feature['Cohort'] = cohort_name
        if features is None:
            features = cohort_feature
        else:
            features = pd.concat([features, cohort_feature], axis=0)
            features = features.reset_index(drop=True)

    # iterate columns, convert MFR column to category with codes
    if 'MFR' in features.columns:
        features['MFR'] = features['MFR'].astype('category')
        features['MFR'] = features['MFR'].cat.codes
        # features['MFR'] = features['MFR'].apply(lambda x: x.cat.codes)
    # nan to 0
    features = features.fillna(0)
    return features




    