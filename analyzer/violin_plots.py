import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal, mannwhitneyu
import itertools

def significant_analysis(features, save_dir, level=0.05):
    """
    Get significant features from features dataframe.
    Features can separate different cohorts.
    :param features: pandas dataframe
    :param save_dir: str, save directory
    return dict names of significant features
    """
    features = features.loc[:, (features != 0).any(axis=0)]
    cohorts = features['Cohort'].unique()
    feature_names = features.columns.tolist()
    feature_names.remove('Cohort')
    feature_names.remove('Name')

    scaler = StandardScaler()
    fscaled = pd.DataFrame(scaler.fit_transform(features[feature_names]), columns=feature_names)
    fscaled['Cohort'] = features['Cohort']
    features = fscaled

    pvalues = {}

    for feature_name in feature_names:
        # print(feature_name)
        pairs = itertools.combinations(cohorts, 2)
        for cohort1, cohort2 in pairs:
            input_lists = [
                features[features['Cohort'] == cohort1][feature_name].tolist(),
                features[features['Cohort'] == cohort2][feature_name].tolist(),
            ]
            # min_len = min([len(input_list) for input_list in input_lists])
            # input_lists = [input_list[:min_len] for input_list in input_lists]
            try:
                # pvalue = kruskal(*input_lists).pvalue
                pvalue = mannwhitneyu(*input_lists).pvalue
            except:
                pvalue = 1
            # print(pvalue)
            # pvalue = wilcoxon(*input_lists).pvalue
            comb = f'{cohort1}-{cohort2}'
            if comb not in pvalues:
                pvalues[comb] = [pvalue]
            else:
                pvalues[comb].append(pvalue)

    pair_names = list(pvalues.keys())
    all_pvalues= []
    for pair_name in pair_names:
        all_pvalues.extend(pvalues[pair_name])
    # adjusted_pvalues = multipletests(all_pvalues, method='fdr_bh')[1]
    adjusted_pvalues = multipletests(all_pvalues, method='bonferroni')[1]
  
    for i in range(len(pair_names)):
        pvalues[pair_names[i]] = adjusted_pvalues[i*len(feature_names):(i+1)*len(feature_names)]

    data = {}
    data['feature'] = feature_names
    for pair_name in pair_names:
        data[pair_name] = pvalues[pair_name]
    # for each feature, all pairs's pvalue <0.05, this feature is significant
    significants = []
    for feature_name in feature_names:
        is_significant = True
        for pair_name in pair_names:
            if pvalues[pair_name][feature_names.index(feature_name)] >= level:
                is_significant = False
                break
        significants.append(is_significant)
    data['Significant'] = significants
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(save_dir, 'p_values.xlsx'))

    significant_features = df[df['Significant'] == True]['feature'].values

    print(f'Saving p values to {os.path.join(save_dir, "p_values.xlsx")}')
    return pvalues





def violin_plots_distribution_analysis(features, save_dir, level=0.05):
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
    pvalues = significant_analysis(features, save_dir, level)

    # get feature names (except Name and Cohort)
    feature_names = list(features.columns.values)
    cohorts = features['Cohort'].unique()
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

    for i, ax, feature in zip(range(num_features), axs.flatten(), feature_names):
        if i == num_cols-1:
            sns.violinplot(x='Cohort', y=feature, data=features, ax=ax, hue='Cohort', legend=True)
            # set legend size
            ax.legend(fontsize=30)
        else:
            sns.violinplot(x='Cohort', y=feature, data=features, ax=ax, hue='Cohort', legend=False)
        # highlight significant features
        for c_i, cohort in enumerate(cohorts):
        # ax.set_facecolor((1, 0, 0, 0.1))
            significant = True
            for k,v in pvalues.items():
                if cohort in k:
                    if v[i] >= level:
                        significant = False
            if significant:
                ax.axvspan(c_i-0.5, c_i+0.5, facecolor='r', alpha=0.1)
        # hide x-asis 
        # ax.get_xaxis().set_visible(False)
        ax.xaxis.set_ticklabels([])
        ax.set_xlabel(feature, fontsize=30)
        ax.set_ylabel('')
        # set y axis font size
        ax.tick_params(labelsize=20)
        # delete ax legend
        # ax.get_legend().remove()
        # set y axis font size
        
    for ax in axs.flatten():
        if not ax.get_xlabel():
            fig.delaxes(ax)
    fig.tight_layout()
    
    # save figure
    save_path = os.path.join(save_dir, 'violin_plots.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saving violin plots to {save_path}')




