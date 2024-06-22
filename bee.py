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
import itertools

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
        feature_mode = param.get('feature_mode', 'path')
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
        skip = param.get('skip', [])
        seed = param.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # load image files from different cohorts
        image_files = {}
        mask_files = {}
        for cohort_dir, mask_dir, cohort_name, cohort_identifier in zip(cohort_dirs, mask_dirs, cohort_names, cohort_identifiers):
            if feature_path is None:
                image_files[cohort_name] = sorted(get_paths(cohort_dir, cohort_identifier, image_exts))
                if mask_dir is not None:
                    mask_files[cohort_name] = sorted(get_paths(mask_dir, cohort_identifier, mask_exts))
                else:
                    mask_files[cohort_name] = [None] * len(image_files[cohort_name])
            else:
                image_files[cohort_name] = []
                mask_files[cohort_name] = []
                features = pd.read_excel(feature_path)
                filenames = features[features['Cohort']==cohort_name]['Name'].tolist()
                ext = image_exts[cohort_names.index(cohort_name)] if len(image_exts) == len(cohort_names) else image_exts[0]
                msk_ext = mask_exts[cohort_names.index(cohort_name)] if len(mask_exts) == len(cohort_names) else mask_exts[0]
                if ext.startswith('.'):
                    ext = ext[1:]
                if msk_ext != None and msk_ext.startswith('.'):
                    msk_ext = msk_ext[1:]
                for filename in filenames:
                    filepath = os.path.join(cohort_dir, filename+'.'+ext)
                    if os.path.exists(filepath):
                        image_files[cohort_name].append(filepath)
                        if mask_dir is not None:
                            mask_files[cohort_name].append(os.path.join(mask_dir, filename+'.'+msk_ext))
                        else:
                            mask_files[cohort_name].append(None)
            
            # check if image and mask files have the same length
            assert len(image_files[cohort_name]) == len(mask_files[cohort_name]), f'Number of image files and mask files are not equal for {cohort_name}'
            
        # create save dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # feature extraction
        if feature_path is None:
            features = extract_feature(image_files, mask_files, feature_mode, cohort_dirs, save_dir, n_workers)
        else:
            features = pd.read_excel(feature_path)

        # image overview
        if image_files[cohort_names[0]] is not None and 'image_overview' not in skip:
            image_overview(image_files, feature_mode, n_workers, save_dir)
        else:
            print('Skipping image overviewer.')

        # UMAP
        umap_distribution_analysis(features, save_dir)

        # # Violin Plots with Significant Tests
        violin_plots_distribution_analysis(features, save_dir)

        # Hierarchical Clustering
        hierarchical_clustering(features, save_dir)

        # Principal Variance Component Analysis (PVCA)
        if clinical_data_paths is not None:
            pvca_distribution_analysis(features, clinical_data_paths, clinical_columns, cohort_names, save_dir)

        # Batch Effect Score
        overall_bes = BES(features, len(cohort_names))
        print(f'Overall BES: {round(overall_bes, 4)}')

        pairs = itertools.combinations(cohort_names, 2)
        for cohort1, cohort2 in pairs:
            pair_features = features[features['Cohort'].isin([cohort1, cohort2])]
            s = BES(pair_features, 2)
            print(f'{cohort1}-{cohort2} BES:', round(s,4))

        
        


       