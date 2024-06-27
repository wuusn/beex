from .consensusclustering.consensusclustering.consensus import ConsensusClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
# from consensusclustering import ConsensusClustering
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import pandas as pd
import umap
from sklearn.metrics import roc_auc_score

def BES(features, n_cohort, n_resample=100):
    # clustering_obj = AgglomerativeClustering(affinity='euclidean', linkage='ward')
    clustering_obj = KMeans(n_clusters=n_cohort)
    # clustering_obj = SpectralClustering(n_clusters=n_cohort, affinity='nearest_neighbors', n_neighbors=10)
    cc = ConsensusClustering(
        clustering_obj=clustering_obj,
        min_clusters=n_cohort,
        max_clusters=n_cohort,
        n_resamples=n_resample,
        resample_frac=0.8,
        k_param='n_clusters',
    )

    feature_names = features.columns.tolist()
    feature_names.remove('Cohort')
    feature_names.remove('Name')
    cohorts = features['Cohort'].values
    x = features[feature_names].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # convert to umap embedding
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(x)
    x = embedding

    # cc._fit_single_k(x, n_cohort)
    cc.fit(x)
    M = cc.consensus_k(n_cohort)
    
    n = x.shape[0]
    cohort_matrix = np.zeros((n, n), dtype="float")
    rows, cols = zip(*list(combinations(range(n), 2)))
    for i, j in zip(rows, cols):
        cohort_matrix[i, j] = cohort_matrix[j, i] = int(cohorts[i] == cohorts[j])
    for i in range(n):
        cohort_matrix[i, i] = 1
    
    score = 0
    count = 0
    cc = 0
    AD = np.zeros((n, n), dtype="float")
    P = M > 0.5
    P = P.astype('int8')
    count_ture = 0
    preds = []
    probs = []
    ys = []
    for i, j in zip(rows, cols):
        if i < j:
            AD[i,j] = abs(M[i, j] - cohort_matrix[i, j])
            score += abs(M[i, j] - cohort_matrix[i, j])
            cc += M[i,j]
            count+=1
            preds.append(P[i,j])
            probs.append(M[i,j])
            ys.append(cohort_matrix[i,j])

    auc  = roc_auc_score(ys, probs)

    return auc
