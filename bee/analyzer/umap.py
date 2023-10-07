import os
import matplotlib.pyplot as plt
import umap
import numpy as np
from sklearn.preprocessing import LabelEncoder

def umap_distribution_analysis(features, save_dir):
    """
    Draw UMAP Plots depends on Cohorts from features
    :param features: pandas dataframe, has column 'Cohort'
    :param save_dir: str, save directory
    :return: None
    """
    # delete column Name and Cohort
    x = features.drop(['Name', 'Cohort'], axis=1)

    # convert Cohort D1, D2 to 1, 2 etc.
    le = LabelEncoder()
    y = le.fit_transform(features['Cohort'])
    # get the mapping
    cohort_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(x)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    colorbar = plt.colorbar(boundaries=np.arange(0, max(cohort_mapping.values())+2)-0.5)
    colorbar.set_ticks(np.arange(0, max(cohort_mapping.values())+1), labels=cohort_mapping.keys())
    
    save_path = os.path.join(save_dir, 'umap.png')
    print(f'Saving UMAP plot to {save_path}')
    plt.savefig(save_path)
    plt.close()