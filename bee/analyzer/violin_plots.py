import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
from scipy.stats import kruskal

def significant_analysis(features, save_dir):
    """
    Get significant features from features dataframe.
    Features can separate different cohorts.
    :param features: pandas dataframe
    :param save_dir: str, save directory
    return dict names of significant features
    """
    # duplicate df
    df = features
    df_selected = features.copy()
    # drop column Name and Cohort
    df_selected.drop(['Name', 'Cohort'], axis=1, inplace=True)
    # get feature names
    feature_names = list(df_selected.columns.values)

    # Scale the DataFrame
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)

    # Replace NaN values with 0
    scaled_df = scaled_df.fillna(0)

    # Compute the correlation matrix
    # corr_mat = scaled_df.cov()
    scaled_df = scaled_df.transpose()
    # scaled_df = scaled_df.reset_index(drop=True)

    # Define the column annotation colors
    Batch_anno = pd.DataFrame(data={'BatchName':df['Cohort']})
    Batch_anno = Batch_anno.reset_index(drop=True)

    # Transpose the dataframe
    scaled_df_kruskal = pd.DataFrame(scaled_df.T)
    # scaled_df_kruskal = scaled_df_kruskal.loc[:, (scaled_df_kruskal != 0).any(axis=0)]

    # Assign column names
    names = scaled_df_kruskal.columns

    # Add the 'Batch' column
    scaled_df_kruskal['Batch'] = Batch_anno['BatchName'].values
    # Create a list of dataframes, each dataframe is a batch
    batch_list = []
    for batch in scaled_df_kruskal['Batch'].unique():
        batch_list.append(scaled_df_kruskal[scaled_df_kruskal['Batch'] == batch])
    # Create an empty DataFrame with the specified number of rows and columns
    kruskal_p_values = pd.DataFrame(index=range(len(names)), columns=['var_name', 'p_value'])

    # Compute the Kruskal-Wallis H-test for each feature
    for i in range(len(names)):
        # Create a list of dataframes, each dataframe is a feature
        feature_list = []
        for batch in batch_list:
            feature_list.append(batch.iloc[:, i])
        # Compute the Kruskal-Wallis H-test
        kruskal_p_values.iloc[i, 0] = names[i]
        kruskal_p_values.iloc[i, 1] = kruskal(*feature_list)[1]
    # save to excel
    kruskal_p_values = kruskal_p_values.dropna().sort_values(by='p_value')
    kruskal_p_values.to_excel(os.path.join(save_dir, 'kruskal_p_values.xlsx'), index=False)
    print(f'Saving Kruskal p values to {os.path.join(save_dir, "kruskal_p_values.xlsx")}')
    # get significant features
    significant_features = kruskal_p_values[kruskal_p_values['p_value'] < 0.05]['var_name'].values
    return significant_features





def violin_plots_distribution_analysis(features, save_dir):
    """
    Draw violin plots for each feature of all cohorts.
    Plots will be highlighted if the feature is significant.
    :param features: pandas dataframe
    :param save_dir: str, save directory
    :return: None
    """
    # drop column has all values equal to 0
    features = features.loc[:, (features != 0).any(axis=0)]

    # get significant features
    significant_features = significant_analysis(features, save_dir)

    # get feature names (except Name and Cohort)
    feature_names = list(features.columns.values)
    feature_names.remove('Name')
    feature_names.remove('Cohort')

    # auto adjust figure size and font size according to the number of features, plot using subplot
    # get number of features
    num_features = len(feature_names)
    # get number of rows and columns of subplots
    # number of rows should be similar to columns
    num_rows = math.ceil(math.sqrt(num_features))
    num_cols = num_rows
    # set font size
    # plt.rcParams.update({'font.size': 20})
    # set figure size
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))

    for ax, feature in zip(axs.flatten(), feature_names):
        sns.violinplot(x='Cohort', y=feature, data=features, ax=ax, hue='Cohort', legend=False)
        # highlight significant features
        if feature in significant_features:
            ax.set_facecolor((1, 0, 0, 0.1))
        ax.set_xlabel('')
    
    # delete empty subplots
    for ax in axs.flatten():
        if not ax.get_ylabel():
            fig.delaxes(ax)
    
    # save figure
    save_path = os.path.join(save_dir, 'violin_plots.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saving violin plots to {save_path}')




