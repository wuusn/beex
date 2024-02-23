import os
import pandas as pd

def get_radi_metrics(image_files, mask_files, save_dir, n_workers):
    """
    Get radiological metrics from medical images
    param: image_files: dict, {cohort_name: [image_file1, image_file2, ...]}
    param: mask_files: dict, {cohort_name: [mask_file1, mask_file2, ...]}
    param: n_workers: int, number of workers
    return: pandas dataframe
    """
    # create soft link for each cohort with files
    for cohort_name, files in image_files.items():
        cohort_dir = os.path.join(save_dir, 'tmp_data', cohort_name)
        if not os.path.exists(cohort_dir):
            os.makedirs(cohort_dir)
        for file in files:
            target_file = os.path.join(cohort_dir, os.path.basename(file))
            # print(file, target_file)
            if not os.path.exists(target_file):
                # print('not exists', target_file)
                os.symlink(file, target_file)
    
    features = None
    for cohort_name in image_files.keys():
        print('extracting features for {}'.format(cohort_name))
        cohort_dir = os.path.join(save_dir, 'tmp_data', cohort_name)
        # get current file path
        tmp_feature_dir = os.path.join(save_dir, 'tmp_data', cohort_name+'_extracted')
        if not os.path.exists(tmp_feature_dir):
            cwd = os.getcwd()
            cmd = f"python {cwd}/feature_extractor/QC.py '{save_dir}/tmp_data/{cohort_name}_extracted' '{cohort_dir}'"
            os.system(cmd)
        res_path = f"{save_dir}/tmp_data/{cohort_name}_extracted/IQM.xlsx"
        cohort_feature = pd.read_excel(res_path)
        cohort_feature['Cohort'] = cohort_name
        if features is None:
            features = cohort_feature
        else:
            features = pd.concat([features, cohort_feature], axis=0)
    
    return features




    