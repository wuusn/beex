# Batch Effect Explorer for Medical Images
A tool to assess and validate batch effects in medical images.

![](docs/beex_overview.png)

# Installation
Step 1: make sure your system has install [OpenSlide](https://openslide.org/download/)

Step 2: clone this repo

Step 3: change dir to repo folder  

Step 4: automatically create a conda env using cmd below:
```
conda env create -f environment.yaml
```



# Run Example
```
conda activate bee
python bee.py config/example.yaml
```

# Configuration
To run BEEx, we need a `yaml` config file. In `config` dir, there are some example yaml files.

Below, we provide a detailed explanation of a config file (config/case_tma.yaml).

```
CaseTMA: # The config title, can be any string
    feature_mode: path # mode of feature extraction, 'path' for pathology and 'radi' for radiology
    n_workers: 8 # number of workers for parallel processing
    cohort_dir: # a list of directories of cohort data
        - '/mnt/hd0/project_large_files/bee/VTMA_v3/SUQH/imgs'
        - '/mnt/hd0/project_large_files/bee/VTMA_v3/QDUH/imgs'
        - '/mnt/hd0/project_large_files/bee/VTMA_v3/SHSU/imgs'
    image_ext: # a list of image file extensions, if not provided, all supported image files in the folder will be used
        - 'png'
    cohort_name: #  a list of cohort names used to identify cohorts, should be the same order as `cohort_dir`
        - 'VTMA-1'
        - 'VTMA-2'
        - 'VTMA-3'
    clinical_data: # a list of clinical data file paths, should be the same order as `cohort_dir`, this is optional
        - '/mnt/hd0/project_large_files/bee/VTMA_v3/SUQH/clinical.xlsx'
        - '/mnt/hd0/project_large_files/bee/VTMA_v3/QDUH/clinical.xlsx'
        - '/mnt/hd0/project_large_files/bee/VTMA_v3/SHSU/clinical.xlsx'
    clinical_column: # a list of clinical data column names, required if provide `clinical_data`
        - 'Invasion'
        - 'Overgrade'
    save_dir: '/data/project_large_files/bee/deploy_test/CaseVTMA' # the folder where we save results
```

