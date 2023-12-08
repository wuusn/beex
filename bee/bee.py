import os
import sys
import glob
import yaml
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool
from preprocessor.util import get_paths
from analyzer import *
from feature_extractor import extract_feature

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

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
        cohort_identifiers = param.get('cohort_identifier', [None] * len(cohort_dirs))
        image_exts = param.get('image_ext')
        mask_exts = param.get('mask_ext', [None]*len(cohort_dirs))
        save_dir = param.get('save_dir', 'save')
        n_workers = param.get('n_workers', 8)
        clinical_data_paths = param.get('clinical_data', None)
        clinical_columns = param.get('clinical_column', None)
        mask_dirs = param.get('mask_dir', [None]*len(cohort_dirs))
        feature_path = param.get('feature_path', None)
        # random_seed = param.get('random_seed', None) Not Support yet
        # if random_seed is not None:
        #     np.random.seed(random_seed)
        #     random.seed(random_seed)

        # load image files from different cohorts
        image_files = {}
        mask_files = {}
        for cohort_dir, mask_dir, cohort_name, cohort_identifier in zip(cohort_dirs, mask_dirs, cohort_names, cohort_identifiers):
            image_files[cohort_name] = sorted(get_paths(cohort_dir, cohort_identifier, image_exts))
            if mask_dir is not None:
                mask_files[cohort_name] = sorted(get_paths(mask_dir, cohort_identifier, mask_exts))
            else:
                mask_files[cohort_name] = [None] * len(image_files[cohort_name])
            
            # check if image and mask files have the same length
            assert len(image_files[cohort_name]) == len(mask_files[cohort_name]), f'Number of image files and mask files are not equal for {cohort_name}'
            
        
        # create save dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # image overview
        # image_overview(image_files, n_workers, save_dir)

        # feature extraction
        if feature_path is None:
            features = extract_feature(image_files, mask_files, save_dir, n_workers)
        else:
            features = pd.read_excel(feature_path)
        # features = features[features['Cohort'] != 'SSPH']

        # UMAP
        umap_distribution_analysis(features, save_dir)

        # Violin Plots with Significant Tests
        violin_plots_distribution_analysis(features, save_dir)

        # Hierarchical Clustering
        hierarchical_clustering(features, save_dir)

        # Principal Variance Component Analysis (PVCA)
        if clinical_data_paths is not None:
            pvca_distribution_analysis(features, clinical_data_paths, clinical_columns, cohort_names, save_dir)



        
        


       