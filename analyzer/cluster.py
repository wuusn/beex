import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def hierarchical_clustering(features, save_dir):
    """
    Hierarchical clustering analysis
    param: features: pd.DataFrame
    param: save_dir: str
    return: None
    """
    df_selected = features.drop(['Name', 'Cohort'], axis=1)
    # # drop duplicated rows
    # df_selected = df_selected[~df_selected.duplicated()]
    # print(df_selected[df_selected.duplicated()])
    
    # Scale the DataFrame
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)
    # Replace NaN values with 0
    scaled_df = scaled_df.fillna(0)

    # Compute the correlation matrix
    corr_mat = scaled_df.cov()

    scaled_df = scaled_df.transpose()

    # define column annotation colors according to cohort
    cohorts = features['Cohort'].unique()
    cohort_colors = {}
    for i, cohort in enumerate(cohorts):
        cohort_colors[cohort] = sns.color_palette('Paired')[i]
    col_colors = features['Cohort'].map(cohort_colors)

    # Define colormap and color breaks
    cmap = sns.color_palette('bwr', 31)
    norm = plt.Normalize(-10, 10)

    for method in ['complete', 'ward', 'average']:
        g = sns.clustermap(scaled_df, method=method,cmap=cmap, col_colors=col_colors, norm=norm, \
               yticklabels=True, xticklabels=False,standard_scale=None, row_cluster=False, col_cluster=True,\
                 cbar_pos=(0.05, .2, .03, .4))
        for label, color in cohort_colors.items():
            g.ax_row_dendrogram.bar(0, 0, color=color,
                                    label=label, linewidth=0)
        g.ax_row_dendrogram.legend(bbox_to_anchor=(0.6, 1.0))

        # save plots
        save_path = os.path.join(save_dir, f'hierarchical_clustering_{method}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved {save_path}')