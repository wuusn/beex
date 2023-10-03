import os
import sys
import glob
import yaml
from multiprocessing import Pool
from preprocessor.util import get_paths
from analyzer import image_overview
from feature_extractor import extract_feature

if __name__ == '__main__':

    yaml_path = sys.argv[1]
    with open(yaml_path, 'r') as f:
        param_sets = yaml.safe_load(f)
    for set_name, param in param_sets.items():
        print(set_name)
        print(param)
        cohort_dirs = param.get('cohort_dir')
        assert isinstance(cohort_dirs, list), 'cohort_dir must be a list'
        cohort_names = param.get('cohort_name')
        assert isinstance(cohort_names, list), 'cohort_name must be a list'
        cohort_identifiers = param.get('cohort_identifier')
        image_exts = param.get('image_ext')
        save_dir = param.get('save_dir', 'save')
        n_workers = param.get('n_workers', 8)

        # load image files from different cohorts
        image_files = {}
        for cohort_dir, cohort_name, cohort_identifier in zip(cohort_dirs, cohort_names, cohort_identifiers):
            image_files[cohort_name] = get_paths(cohort_dir, cohort_identifier, image_exts)[:20]
        
        # create save dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # image overview
        # image_overview(image_files, save_dir)

        # feature extraction
        features = extract_feature(image_files, save_dir, n_workers)
        


       