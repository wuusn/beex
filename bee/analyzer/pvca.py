import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pymer4.models import Lmer
import itertools

def pvca_distribution_analysis(features, clinical_data_paths, clinical_columns, cohort_names, save_dir):
    """
    Principal Variance Component Analysis (PVCA)
    param: features: dataframe of features
    param: clinical_data_paths: list of paths to clinical data
    param: clinical_columns: list of clinical columns to be analyzed
    param: cohort_names: list of cohort names
    param: save_dir: save directory
    return: None
    """

    # load clinical data and concatenate
    clinical_data = pd.DataFrame()
    for clinical_data_path, cohort_name in zip(clinical_data_paths, cohort_names):
        df = pd.read_excel(clinical_data_path)
        df = df[['Name'] + clinical_columns]
        # drop rows which name is not in features
        df = df[df['Name'].isin(features['Name'])]
        # add cohort info
        df['Batch'] = cohort_name
        clinical_data = pd.concat([clinical_data, df], axis=0)

    # set as categorical
    cate_cols = clinical_columns + ['Batch']
    clinical_data[cate_cols] = clinical_data[cate_cols].astype('category')

    # save clinical data
    clinical_data.to_excel(os.path.join(save_dir, 'clinical_data.xlsx'), index=False)
    
    # duplicate df
    features = features[features['Name'].isin(clinical_data['Name'])]
    df_selected = features.copy()
    # print(len(df_selected), len(clinical_data))

    # drop column Name and Cohort
    df_selected.drop(['Name', 'Cohort'], axis=1, inplace=True)
    # Scale the DataFrame
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)
    # print(len(scaled_df))

    # Replace NaN values with 0
    scaled_df = scaled_df.fillna(0)

    # Compute the correlation matrix
    corr_mat = scaled_df.cov()

    scaled_df_T = scaled_df.T

    # Fit PCA
    pca = PCA()
    scaled_pca = pca.fit(scaled_df_T)
    # print(scaled_df_T.shape)
    

    # Get the principal components (rotation in R)
    prin_comps_scaled = scaled_pca.components_

    # Get the eigenvalues (sdev^2 in R)
    eigvalues_scaled = scaled_pca.explained_variance_

    # Compute the sum of eigenvalues
    eigvalues_sum_scaled = np.sum(eigvalues_scaled)

    # Compute the variance explained
    var_explained_scaled = eigvalues_scaled / eigvalues_sum_scaled

    # processing PVCA
    outs = []
    df = features.copy()
    num_pca = 5
    for i in range(num_pca):
        pc = prin_comps_scaled[:, i]
        # scaled_df_named = np.transpose(scaled_df)
        pc_proj = np.dot(scaled_df, pc)
        pc_proj = pd.DataFrame(pc_proj)
        pc_proj['Name'] = list(df['Name'])
        df_pd_pc = pd.merge(clinical_data, pc_proj, on='Name', how='outer')
        # print(df_pd_pc.shape)
        # df_pd_pc.to_excel(os.path.join(save_dir, f'df_pd_pc{i+1}.xlsx'), index=False)
        df_pd_pc = df_pd_pc.dropna()
        df_pd_pc.columns = ['Name'] + cate_cols + [f'PC{i+1}']

        # fit Lmer
        effects = ""
        for cate_col in cate_cols:
            if effects == "":
                effects += f"(1|{cate_col})"
            else:
                effects += f" + (1|{cate_col})"
        combinations = list(itertools.combinations(cate_cols, 2))
        for combination in combinations:
            effects += f" + (1|{combination[0]}:{combination[1]})"
        # effects = "(1|Batch)"
        print(effects) 

        # fit model
        # df_pd_pc.to_excel(os.path.join(save_dir, f'pvca_pc{i+1}.xlsx'), index=False)
        model = Lmer(f'PC{i+1} ~ {effects}', data=df_pd_pc)    
        model.fit()
        outs.append(model.ranef_var)
      
    variance_decompositions = np.column_stack([o['Var'] for o in outs]) # not vcov

    factor_variances = np.sum(variance_decompositions, axis=1)
    total_variance = np.sum(factor_variances)
    percentage_variance = (factor_variances / total_variance)

    percentage_variance_named = pd.DataFrame({
        'grp': list(outs[0].index),
        'percentage_variance': percentage_variance
    })

    # plot
    plt.figure()
    sns.barplot(x='percentage_variance', y='grp', data=percentage_variance_named, legend=False)
    # hide x and y label
    plt.xlabel('')
    plt.ylabel('')
    # save plot and xlsx
    plt.savefig(os.path.join(save_dir, 'pvca.png'), dpi=300)
    print('PVCA plot saved to', os.path.join(save_dir, 'pvca.png'))
    plt.close()
    percentage_variance_named.to_excel(os.path.join(save_dir, 'pvca.xlsx'), index=False)
    print('PVCA data saved to', os.path.join(save_dir, 'pvca.xlsx'))

