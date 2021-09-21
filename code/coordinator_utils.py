import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.linalg import block_diag
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import f1_score, silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

def generate_clustered_dataset(dimension,total_no_samples,no_of_cluster, random_state):
    clustered_dataset, true_label, centroids = make_blobs(n_samples=total_no_samples,
                                                          n_features=dimension, 
                                                          centers=no_of_cluster,
                                                          return_centers=True, 
                                                          random_state=random_state) #check randomstate =80
    return clustered_dataset, true_label, centroids

def perform_PCA(dimension, dataset):
    pca = PCA(n_components= dimension)
    return pca.fit_transform(dataset)

def perform_MDS(dimension, dataset):
    mds = MDS(n_components=dimension)
    return mds.fit_transform(dataset)

def calc_fed_euc_dist(sldm_array):
    combined_eucl = np.concatenate(sldm_array)
    # rows are s1,s2..sn while columns are datapoints
    print(combined_eucl)
    # computing the distance of distance (e.g.: meta-distance)
    # number of all samples * number of all samples
    return euclidean_distances(combined_eucl)

def construct_global_Mx_Cx_matrix(MxCx,dataset_len_array):
    Mi,Ci = np.split(np.array(MxCx),2,axis=1)
    arrayMi=Mi.flatten()
    arrayCi=Ci.flatten()
    print("arrayMi: ",arrayMi)
    print("arrayCi: ",arrayCi)
    
    Mi_avg=np.average(arrayMi)
    Ci_avg=np.average(arrayCi)
    print("Average of slopes: ", Mi_avg)
    print("Average of constants: ", Ci_avg)
    print("array of number of vectors in the datasets, i.e., shape array: \n",dataset_len_array)
    
    #Placing the respective Mi of each datapoints and getting Mx matrix
    global_Mx = block_diag(*[np.full((i, i), c) for c, i in zip(arrayMi, dataset_len_array)])
    #Placing the respective Ci of each datapoints and getting Cx matrix
    global_Cx = block_diag(*[np.full((i, i), c) for c, i in zip(arrayCi, dataset_len_array)])
    print("Average of slope or coefficients i.e. Mi's: ", global_Mx)
    print("Average of constants or intercepts i.e. Ci's: ", global_Cx)
    # The zeroes in global slopes and constants matrix are replaced by Mi_avg and Ci_avg respectively 
    # They are used to calculate the predicted distance for cross-sectional data
    # For example: distance(a1,b5) where a1 and b5 belongs to different datasets
    global_Mx[global_Mx == 0] = Mi_avg
    global_Cx[global_Cx == 0] = Ci_avg
    print("Global coefficient or slope matrix: \n", global_Mx)
    print("Global constant or intercept matrix: \n", global_Cx)
    return global_Mx, global_Cx

def calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx):
    PGDM=np.add(np.multiply(global_Mx, global_fed_euc_dist),global_Cx)
    #As distance between same points is 0
    np.fill_diagonal(PGDM,0)
    # print("Predicted Global Distance Matrix: \n",PGDM)
    return PGDM

def plotDistanceMatrix(distmat, title):
    ax = plt.axes()
    sns.heatmap(distmat, ax = ax)
    ax.set_title(title)
    plt.show()

def evaluation_scores(dist_matrix, dist_matrix_name, expected_label, actual_label, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=False, davies_bouldin=True):
    print(f"Adjusted similarity score of the clustering with {dist_matrix_name} in (%) :", adjusted_rand_score(expected_label, actual_label)*100) if adj_rand == True else 0
    print(f"Adjusted mutual info score of the clustering with {dist_matrix_name} in (%) :", adjusted_mutual_info_score(expected_label, actual_label)*100) if adj_mutual_info == True else 0
    print(f"F1 score after clustering with {dist_matrix_name}:",f1_score(expected_label, actual_label, average='micro')) if f1 == True else 0
    print(f"Silhouette score of {dist_matrix_name}:  ",silhouette_score(dist_matrix, actual_label, metric='precomputed')) if silhouette == True else 0
    print(f"Davies-Bouldin Score of {dist_matrix_name}: ", davies_bouldin_score(dist_matrix, actual_label)) if davies_bouldin == True else 0
    
def pearson_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist):
    print("Pearson correlation between true and predicted global matrices:", np.corrcoef(global_true_euc_dist.flatten(),global_pred_euc_dist.flatten())[0,1])
    print("Pearson correlation between true and federated global matrices:", np.corrcoef(global_true_euc_dist.flatten(),global_fed_euc_dist.flatten())[0,1])
    print("Pearson correlation between federated and predicted global matrices:", np.corrcoef(global_fed_euc_dist.flatten(),global_pred_euc_dist.flatten())[0,1])

