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
from time import time
from pathlib import Path
import pickle
import resource

clinical_paths = [
    '/mnt/hd0/project_large_files/bee/VTMA_v3/QDUH/clinical.xlsx',
    '/mnt/hd0/project_large_files/bee/VTMA_v3/SHSU/clinical.xlsx',
    '/mnt/hd0/project_large_files/bee/VTMA_v3/SUQH/clinical.xlsx'
]

clinical_pool_df = pd.concat([pd.read_excel(clinical_path) for clinical_path in clinical_paths])
clinical_pool_df.reset_index(drop=True, inplace=True)
clinical_pool_path = '/mnt/hd0/project_large_files/bee/fake/clinical_pool.xlsx'
clinical_pool_df.to_excel(clinical_pool_path, index=False)


def fake_bee(image_files, n_center, n_image):
    save_dir = f'/mnt/hd0/project_large_files/bee/fake/{n_center}_{n_image}'

    if os.path.exists(f'{save_dir}/time_dict.pkl'):
        with open(f'{save_dir}/time_dict.pkl', 'rb') as file:
            return pickle.load(file)
        

    clinical_data_paths = []
    
    os.makedirs(save_dir, exist_ok=True)
    for cohort_name, files in image_files.items():
        clinical_data_paths.append(clinical_pool_path)

    time_dict = {}
    start = time()

    mask_files = {cohort_name: [None] * len(image_files[cohort_name]) for cohort_name in image_files.keys()}
    feature_mode = 'path'
    
    cohort_dirs = None
    n_workers = 8
    cohort_names = list(image_files.keys())
    clinical_columns = ['Invasion', 'Overgrade']

    restart = time()
    features = extract_feature(image_files, mask_files, feature_mode, cohort_dirs, save_dir, n_workers)
    # features = pd.read_excel('/mnt/hd0/project_large_files/bee/fake/2_50/features.xlsx')
    end = time()
    time_dict['extract_feature'] = end - restart

    restart = time()
    image_overview(image_files, feature_mode, n_workers, save_dir)
    end = time()
    time_dict['image_overview'] = end - restart

    restart = time()
    umap_distribution_analysis(features, save_dir)
    end = time()
    time_dict['umap_distribution_analysis'] = end - restart

    restart = time()
    violin_plots_distribution_analysis(features, save_dir)
    end = time()
    time_dict['violin_plots_distribution_analysis'] = end - restart

    restart = time()
    hierarchical_clustering(features, save_dir)
    end = time()
    time_dict['hierarchical_clustering'] = end - restart

    restart = time()
    pvca_distribution_analysis(features, clinical_data_paths, clinical_columns, cohort_names, save_dir)
    end = time()

    time_dict['pvca_distribution_analysis'] = end - restart
    time_dict['total'] = end - start

    # write to pickle
    with open(f'{save_dir}/time_dict.pkl', 'wb') as f:
        pickle.dump(time_dict, f)

    return time_dict





n_fake_centers = [2, 5, 10, 25, 50, ]#100]
n_fake_images = [50,100,200,300, 400, 500]
fake_images_pool = get_paths('/mnt/hd0/project_large_files/bee/VTMA_v3', exts=['png'])
times = {
    'n_fake_centers': [],
    'n_fake_images': [],
    'extract_feature': [],
    'image_overview': [],
    'umap_distribution_analysis': [],
    'violin_plots_distribution_analysis': [],
    'hierarchical_clustering': [],
    'pvca_distribution_analysis': [],
    'total': []
}
for n_fake_center in n_fake_centers:
    for n_fake_image in n_fake_images:
        cohort_names = [f'fake-{i}' for i in range(n_fake_center)]
        cohort_files = [np.array(fake_images_pool)[np.random.choice(len(fake_images_pool), n_fake_image, replace=True)] 
                        for _ in range(n_fake_center)]
        image_files = dict(zip(cohort_names, cohort_files))
        
        time_dict = fake_bee(image_files, n_fake_center, n_fake_image)
        print(n_fake_center, n_fake_image, time_dict)

        times['n_fake_centers'].append(n_fake_center)
        times['n_fake_images'].append(n_fake_image)
        times['extract_feature'].append(time_dict['extract_feature'])
        times['image_overview'].append(time_dict['image_overview'])
        times['umap_distribution_analysis'].append(time_dict['umap_distribution_analysis'])
        times['violin_plots_distribution_analysis'].append(time_dict['violin_plots_distribution_analysis'])
        times['hierarchical_clustering'].append(time_dict['hierarchical_clustering'])
        times['pvca_distribution_analysis'].append(time_dict['pvca_distribution_analysis'])
        times['total'].append(time_dict['total'])

times_df = pd.DataFrame(times)
times_df.to_excel('/mnt/hd0/project_large_files/bee/fake/times.xlsx', index=False)


        